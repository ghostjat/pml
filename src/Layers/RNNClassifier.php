<?php

declare(strict_types=1);

namespace Pml\Layers;

use Pml\{Tensor, Ops, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  RNNClassifier — Sequence-to-label classifier backed by RNNCell or LSTMCell
//
//  Unrolls a recurrent cell over the time dimension T of an input sequence,
//  then maps the final hidden state h_T to class logits via a linear head.
//
//  ── Architecture ─────────────────────────────────────────────────────────
//
//    Input: X [B, T, I]  — batch of B sequences, each T steps, I features
//
//    For t = 0 … T−1:
//      h_t = cell.forward(X[:, t, :], h_{t-1})   (RNNCell)
//      h_t, c_t = cell.forward(X[:,t,:], h_{t-1}, c_{t-1})  (LSTMCell)
//
//    logits [B, n_classes] = h_T @ W_out^T + b_out
//
//  ── BPTT and Gradient Clipping ───────────────────────────────────────────
//
//  Full Backpropagation Through Time (BPTT):
//    1. Forward pass: store all T caches.
//    2. Backward from loss gradient dlogits [B, n_classes]:
//       a. dh_T [B, H] = dlogits @ W_out         (sgemm)
//       b. dW_out += dlogits^T @ h_T              (sgemm)
//       c. db_out += Σ_b dlogits                  (sgemv)
//    3. Unroll backward t = T−1 … 0:
//       (dx_t, dh_{t-1}) = cell.backward(dh_t, cache_t)    RNN
//       (dx_t, dh_{t-1}, dc_{t-1}) = cell.backward(...)    LSTM
//    4. Global gradient norm clipping:
//       total_norm = √(Σ_p ||p.grad||²)
//       if total_norm > max_norm: scale = max_norm / total_norm; sscal each grad
//
//  Clipping prevents exploding gradients, the central pathology of BPTT on
//  long sequences.  The global norm clipping (Pascanu et al., 2013) is used
//  rather than per-parameter clipping to preserve the relative directions of
//  different parameter gradients.
//
//  ── Head linear layer BLAS ───────────────────────────────────────────────
//
//  Forward:  logits [B, C] = h_T [B, H] @ W_out^T [H, C]
//              sgemm(NoTrans, Trans, B, C, H)
//  Backward: dW_out[C, H] += dlogits^T @ h_T
//              sgemm(Trans, NoTrans, C, H, B)
//            dh_T [B, H]  = dlogits @ W_out
//              sgemm(NoTrans, NoTrans, B, H, C)
//            db_out[C]    += Σ_b dlogits[b, :]
//              sgemv(Trans, B, C)
//
//  ── Complexity ────────────────────────────────────────────────────────────
//
//  Forward:  O(T · (B·I·H + B·H²))    T cell steps
//  Backward: O(T · (B·I·H + B·H²))    T BPTT steps + O(B·H·C) head backward
// ═══════════════════════════════════════════════════════════════════════════

final class RNNClassifier
{
    // ── Head parameters ───────────────────────────────────────────────────

    /** Output weight matrix [n_classes, hidden_size]. */
    public readonly Tensor $W_out;

    /** Output bias [n_classes]. */
    public readonly Tensor $b_out;

    // ── Architecture meta ─────────────────────────────────────────────────

    /** Whether the cell is an LSTMCell (needs c-state) or RNNCell. */
    private readonly bool $isLSTM;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param RNNCell|LSTMCell $cell       Recurrent cell to unroll.
     * @param int              $n_classes  Number of output classes.
     * @param float            $maxNorm    Gradient clipping threshold (global L2 norm).
     *                                    Set to INF to disable clipping.
     */
    public function __construct(
        private readonly RNNCell|LSTMCell $cell,
        private readonly int              $n_classes,
        private readonly float            $maxNorm = 5.0,
    ) {
        $H = $cell->hidden_size;
        $C = $n_classes;

        // Xavier init for the head
        $initStd       = sqrt(2.0 / ($H + $C));
        $this->W_out   = Tensor::randn([$C, $H], 0.0, $initStd);
        $this->b_out   = Tensor::zeros([$C]);

        $this->W_out->requiresGrad = true;
        $this->b_out->requiresGrad = true;

        $this->isLSTM = ($cell instanceof LSTMCell);
    }

    // ── Inference ─────────────────────────────────────────────────────────

    /**
     * Forward pass + BPTT backward + gradient clipping.
     *
     * Intended for training: returns the logits AND populates gradients on
     * all parameters.  Call $optimizer->step() and then ->zeroGrad() after.
     *
     * @param Tensor   $X       Input sequences — [batch_size, T, input_size].
     * @param Tensor   $dLogits Gradient of the loss w.r.t. logits — [batch_size, n_classes].
     *                          Typically: softmax(logits) − one_hot(y), divided by batch_size.
     * @return Tensor           Logits [batch_size, n_classes] from the forward pass.
     *
     * ── FFI boundary assertions ─────────────────────────────────────────
     * Checked before every sgemm call.
     */
    public function trainStep(Tensor $X, Tensor $dLogits): Tensor
    {
        [$logits, $caches, $hT, $hStates, $cStates] = $this->forwardWithCache($X);
        $this->backwardWithGradClip($dLogits, $caches, $hT, $hStates, $cStates, $X);
        return $logits;
    }

    /**
     * Inference-only forward pass.
     * Returns logits [B, n_classes] without storing caches or computing gradients.
     *
     * @param Tensor $X  [batch_size, T, input_size]
     * @return Tensor    [batch_size, n_classes]
     */
    public function predict(Tensor $X): Tensor
    {
        $this->assertInputShape($X);

        [$B, $T, $I] = $X->shape;
        $H           = $this->cell->hidden_size;

        // Initialise hidden (and cell) state
        [$h, $c] = $this->isLSTM
            ? $this->cell->zeroState($B)
            : [$this->cell->zeroState($B), null];

        // Unroll forward-only
        for ($t = 0; $t < $T; $t++) {
            $x_t = $this->extractTimestep($X, $t, $B, $I);

            if ($this->isLSTM) {
                [$h, $c] = $this->cell->forward($x_t, $h, $c);
            } else {
                [$h]  = $this->cell->forward($x_t, $h);
            }
        }

        return $this->headForward($h, $B, $H);
    }

    /**
     * Compute cross-entropy loss and its gradient w.r.t. logits in one call.
     *
     * Returns [logits, dLogits, loss] where dLogits = (softmax - one_hot) / B
     * for use with trainStep().
     *
     * @param Tensor  $X        [B, T, I]
     * @param int[]   $targets  Class index per sample, length B.
     * @return array{Tensor, Tensor, float}  [logits [B,C], dLogits [B,C], loss]
     */
    public function loss(Tensor $X, array $targets): array
    {
        $logits = $this->predict($X);
        [$dLogits, $loss] = $this->softmaxCrossEntropy($logits, $targets);
        return [$logits, $dLogits, $loss];
    }

    /**
     * Accuracy: fraction of samples where argmax(logits) == target.
     */
    public function accuracy(Tensor $X, array $targets): float
    {
        $logits = $this->predict($X);
        $B      = $logits->shape[0];
        $C      = $logits->shape[1];
        $ok     = 0;

        for ($b = 0; $b < $B; $b++) {
            $best = 0;
            $bv   = (float) $logits->buffer[$b * $C];
            for ($c = 1; $c < $C; $c++) {
                $v = (float) $logits->buffer[$b * $C + $c];
                if ($v > $bv) { $bv = $v; $best = $c; }
            }
            if ($best === $targets[$b]) { $ok++; }
        }

        return $ok / $B;
    }

    /**
     * Return all learnable parameters (cell + head).
     * @return Tensor[]
     */
    public function parameters(): array
    {
        return array_merge($this->cell->parameters(), [$this->W_out, $this->b_out]);
    }

    /**
     * Zero all parameter gradients.
     */
    public function zeroGrad(): void
    {
        foreach ($this->parameters() as $p) {
            $p->zeroGrad();
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * Forward pass that stores all T step caches and intermediate states.
     *
     * @return array  [logits, caches[], hT, hStates[], cStates[]]
     */
    private function forwardWithCache(Tensor $X): array
    {
        $this->assertInputShape($X);

        [$B, $T, $I] = $X->shape;
        $H           = $this->cell->hidden_size;

        if ($this->isLSTM) {
            [$h, $c] = $this->cell->zeroState($B);
        } else {
            $h = $this->cell->zeroState($B);
            $c = null;
        }

        $caches  = [];
        $hStates = [$h];  // hStates[t] = h_{t-1} going into step t
        $cStates = $this->isLSTM ? [$c] : [];

        for ($t = 0; $t < $T; $t++) {
            $x_t = $this->extractTimestep($X, $t, $B, $I);

            if ($this->isLSTM) {
                [$h, $c, $cache] = $this->cell->forward($x_t, $h, $c);
                $cStates[]       = $c;
            } else {
                [$h, $cache] = $this->cell->forward($x_t, $h);
            }

            $caches[]  = $cache;
            $hStates[] = $h;
        }

        $hT     = $h;  // h_T = final hidden state
        $logits = $this->headForward($hT, $B, $H);

        return [$logits, $caches, $hT, $hStates, $cStates];
    }

    /**
     * BPTT backward: head + T cell steps + gradient clipping.
     */
    private function backwardWithGradClip(
        Tensor $dLogits,
        array  $caches,
        Tensor $hT,
        array  $hStates,
        array  $cStates,
        Tensor $X,
    ): void {
        [$B, $T, $I] = $X->shape;
        $H           = $this->cell->hidden_size;
        $C           = $this->n_classes;

        $blas = BlasEngine::get()->ffi;

        // ── Dimension assertions ───────────────────────────────────────────
        if ($dLogits->shape !== [$B, $C]) {
            throw new \InvalidArgumentException(
                "RNNClassifier: dLogits must be [{$B}, {$C}], got ["
                . implode(', ', $dLogits->shape) . '].'
            );
        }

        // ── Head backward ─────────────────────────────────────────────────
        $this->W_out->initGrad();
        $this->b_out->initGrad();

        // dW_out [C, H] += dLogits^T @ hT  (sgemm Trans, NoTrans, C, H, B)
        $blas->cblas_sgemm(101, 112, 111, $C, $H, $B,
            1.0, $dLogits->buffer, $C, $hT->buffer, $H,
            1.0, $this->W_out->grad, $H);

        // db_out [C] += Σ_b dLogits  (sgemv Trans, B, C)
        $onesB = Tensor::ones([$B]);
        $blas->cblas_sgemv(101, 112, $B, $C,
            1.0, $dLogits->buffer, $C, $onesB->buffer, 1,
            1.0, $this->b_out->grad, 1);

        // dh_T [B, H] = dLogits @ W_out  (sgemm NoTrans, NoTrans, B, H, C)
        $dh = new Tensor([$B, $H]);
        $blas->cblas_sgemm(101, 111, 111, $B, $H, $C,
            1.0, $dLogits->buffer, $C, $this->W_out->buffer, $H,
            0.0, $dh->buffer, $H);

        // ── BPTT: unroll backward t = T−1 … 0 ────────────────────────────
        $dc = $this->isLSTM ? Tensor::zeros([$B, $H]) : null;

        for ($t = $T - 1; $t >= 0; $t--) {
            if ($this->isLSTM) {
                [$dx, $dh, $dc] = $this->cell->backward($dh, $dc, $caches[$t]);
            } else {
                [$dx, $dh] = $this->cell->backward($dh, $caches[$t]);
            }
            // $dx is discarded (we have no gradient w.r.t. inputs by default).
            // $dh flows backward as dh_{t-1} into the previous step.
        }

        // ── Global gradient norm clipping (Pascanu et al., 2013) ──────────
        //
        // total_norm = √(Σ_p ||p.grad||²)
        // If total_norm > maxNorm: scale each grad by maxNorm / total_norm
        if ($this->maxNorm < INF) {
            $normSq = 0.0;
            foreach ($this->parameters() as $p) {
                if ($p->grad !== null) {
                    $normSq += (float) $blas->cblas_sdot($p->size, $p->grad, 1, $p->grad, 1);
                }
            }
            $totalNorm = sqrt($normSq);

            if ($totalNorm > $this->maxNorm) {
                $scale = $this->maxNorm / $totalNorm;
                foreach ($this->parameters() as $p) {
                    if ($p->grad !== null) {
                        $blas->cblas_sscal($p->size, (float) $scale, $p->grad, 1);
                    }
                }
            }
        }
    }

    /**
     * Linear head forward: logits [B, C] = h [B, H] @ W_out^T + b_out.
     *
     * ── BLAS: sgemm(NoTrans, Trans, B, C, H) ─────────────────────────────
     */
    private function headForward(Tensor $h, int $B, int $H): Tensor
    {
        $C    = $this->n_classes;
        $blas = BlasEngine::get()->ffi;

        // Dimension assertion
        if ($h->shape !== [$B, $H]) {
            throw new \InvalidArgumentException(
                "RNNClassifier head: h must be [{$B}, {$H}], got ["
                . implode(', ', $h->shape) . '].'
            );
        }

        $logits = new Tensor([$B, $C]);
        $blas->cblas_sgemm(101, 111, 112, $B, $C, $H,
            1.0, $h->buffer, $H, $this->W_out->buffer, $H,
            0.0, $logits->buffer, $C);

        Ops::addBiasInPlace($logits, $this->b_out);

        return $logits;
    }

    /**
     * Extract time step $t from X [B, T, I] → x_t [B, I].
     *
     * Uses cblas_scopy with stride I over the T axis.
     * This is O(B·I) with one scopy per batch sample, which is the minimum
     * possible data movement for this non-contiguous slice.
     */
    private function extractTimestep(Tensor $X, int $t, int $B, int $I): Tensor
    {
        $T    = $X->shape[1];
        $blas = BlasEngine::get()->ffi;
        $x_t  = new Tensor([$B, $I]);

        for ($b = 0; $b < $B; $b++) {
            // X[b, t, :] starts at offset b*T*I + t*I
            $srcOff = $b * $T * $I + $t * $I;
            $src    = \FFI::cast('float*', \FFI::addr($X->buffer[$srcOff]));
            $dstOff = $b * $I;
            $dst    = \FFI::cast('float*', \FFI::addr($x_t->buffer[$dstOff]));
            $blas->cblas_scopy($I, $src, 1, $dst, 1);
        }

        return $x_t;
    }

    /**
     * Fused softmax cross-entropy gradient.
     *
     * Returns (dLogits [B, C], mean_loss) where:
     *   softmax_probs = softmax(logits)
     *   dLogits[b, c] = (softmax_probs[b, c] − 1{c == target[b]}) / B
     *   loss          = −(1/B) Σ_b log softmax_probs[b, target[b]]
     *
     * @param  Tensor  $logits  [B, C]
     * @param  int[]   $targets Length B
     * @return array{Tensor, float}
     */
    private function softmaxCrossEntropy(Tensor $logits, array $targets): array
    {
        $B = $logits->shape[0];
        $C = $logits->shape[1];

        $dLogits = new Tensor([$B, $C]);
        $loss    = 0.0;

        for ($b = 0; $b < $B; $b++) {
            $off = $b * $C;

            // Numerically stable softmax: subtract row max
            $maxV = (float) $logits->buffer[$off];
            for ($c = 1; $c < $C; $c++) {
                $v = (float) $logits->buffer[$off + $c];
                if ($v > $maxV) { $maxV = $v; }
            }

            $expSum = 0.0;
            for ($c = 0; $c < $C; $c++) {
                $e = exp((float) $logits->buffer[$off + $c] - $maxV);
                $dLogits->buffer[$off + $c] = $e;
                $expSum += $e;
            }

            $tgt = $targets[$b];
            $loss += -log(max(1e-15, (float) $dLogits->buffer[$off + $tgt] / $expSum));

            // Normalise to probabilities; subtract 1 for target; divide by B
            for ($c = 0; $c < $C; $c++) {
                $prob = (float) $dLogits->buffer[$off + $c] / $expSum;
                $dLogits->buffer[$off + $c] = ($prob - ($c === $tgt ? 1.0 : 0.0)) / $B;
            }
        }

        return [$dLogits, $loss / $B];
    }

    private function assertInputShape(Tensor $X): void
    {
        if (count($X->shape) !== 3 || $X->shape[2] !== $this->cell->input_size) {
            throw new \InvalidArgumentException(
                "RNNClassifier: X must be [B, T, {$this->cell->input_size}], got ["
                . implode(', ', $X->shape) . '].'
            );
        }
    }
}
