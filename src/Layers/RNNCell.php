<?php

declare(strict_types=1);

namespace Pml\Layers;

use Pml\{Tensor, Ops, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  RNNCell — single time-step Elman RNN
//
//  Computes one step of the recurrent update:
//
//    h_t = tanh( x_t W_{ih}^T + b_{ih}  +  h_{t-1} W_{hh}^T + b_{hh} )
//
//  ── Parameters ───────────────────────────────────────────────────────────
//
//    W_ih  [hidden_size, input_size]   input-to-hidden weight matrix
//    W_hh  [hidden_size, hidden_size]  hidden-to-hidden weight matrix
//    b_ih  [hidden_size]               input-to-hidden bias
//    b_hh  [hidden_size]               hidden-to-hidden bias
//
//  ── Batched Forward  (x: [B, I], h_prev: [B, H]) ─────────────────────────
//
//    pre_act [B, H]  = x @ W_ih^T          — sgemm(NoTrans, Trans, B, H, I)
//                    + h_prev @ W_hh^T     — sgemm(NoTrans, Trans, B, H, H)
//                    + b_ih                — addBiasInPlace
//                    + b_hh                — addBiasInPlace
//    h_t [B, H]      = tanh(pre_act)       — element-wise PHP loop
//
//  ── Explicit BPTT Backward  (dh_t: [B, H]) ───────────────────────────────
//
//  Given upstream gradient dh_t [B, H] from the next time step or the loss:
//
//    d_pre [B, H]  = dh_t ⊙ (1 − h_t²)       tanh backward (element-wise)
//
//    dW_ih [H, I] += d_pre^T @ x              sgemm(Trans, NoTrans, H, I, B)
//    dW_hh [H, H] += d_pre^T @ h_prev         sgemm(Trans, NoTrans, H, H, B)
//    db_ih [H]    += Σ_b d_pre[b, :]          sgemv(Trans, B, H)
//    db_hh [H]    += Σ_b d_pre[b, :]          sgemv(Trans, B, H)
//
//    dx     [B, I]  = d_pre @ W_ih            sgemm(NoTrans, NoTrans, B, I, H)
//    dh_prev[B, H]  = d_pre @ W_hh            sgemm(NoTrans, NoTrans, B, H, H)
//
//  ── Gradient Clipping ─────────────────────────────────────────────────────
//
//  Clipping is applied at the CLASSIFIER level (RNNClassifier) after the full
//  BPTT unroll, not inside the cell.  The cell only computes local gradients
//  and accumulates them; the classifier clips and steps the optimizer.
//
//  ── Complexity ────────────────────────────────────────────────────────────
//
//  Forward  per step: O(B·I·H + B·H²) — dominated by two sgemm calls.
//  Backward per step: O(B·I·H + B·H²) — same.
// ═══════════════════════════════════════════════════════════════════════════

final class RNNCell
{
    // ── Learnable parameters ──────────────────────────────────────────────

    /** Input-to-hidden weight matrix [hidden_size, input_size]. */
    public readonly Tensor $W_ih;

    /** Hidden-to-hidden weight matrix [hidden_size, hidden_size]. */
    public readonly Tensor $W_hh;

    /** Input-to-hidden bias [hidden_size]. */
    public readonly Tensor $b_ih;

    /** Hidden-to-hidden bias [hidden_size]. */
    public readonly Tensor $b_hh;

    // ── Dimensions ───────────────────────────────────────────────────────

    public readonly int $input_size;
    public readonly int $hidden_size;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int   $input_size   Dimension of x_t.
     * @param int   $hidden_size  Dimension of h_t.
     * @param float $initStd      Xavier-like stddev for weight initialization.
     *                            Default: sqrt(1 / hidden_size) (PyTorch RNNCell default).
     */
    public function __construct(
        int   $input_size,
        int   $hidden_size,
        float $initStd = 0.0,
    ) {
        $this->input_size  = $input_size;
        $this->hidden_size = $hidden_size;

        // Default stddev: 1/sqrt(hidden_size) — same as PyTorch RNN default
        if ($initStd <= 0.0) {
            $initStd = 1.0 / sqrt((float) $hidden_size);
        }

        $this->W_ih = Tensor::randn([$hidden_size, $input_size],  0.0, $initStd);
        $this->W_hh = Tensor::randn([$hidden_size, $hidden_size], 0.0, $initStd);
        $this->b_ih = Tensor::zeros([$hidden_size]);
        $this->b_hh = Tensor::zeros([$hidden_size]);

        // Mark parameters as learnable so gradient buffers will be allocated
        $this->W_ih->requiresGrad = true;
        $this->W_hh->requiresGrad = true;
        $this->b_ih->requiresGrad = true;
        $this->b_hh->requiresGrad = true;
    }

    // ── Forward pass ──────────────────────────────────────────────────────

    /**
     * Compute one RNN time step.
     *
     * @param Tensor $x      Input at this step — shape [batch_size, input_size].
     * @param Tensor $h_prev Previous hidden state — shape [batch_size, hidden_size].
     * @return array{Tensor, array}  [h_t [B,H], cache for BPTT backward]
     *
     * ── Dimension assertions ──────────────────────────────────────────────
     * Checked before every sgemm call to prevent FFI out-of-bounds segfaults.
     */
    public function forward(Tensor $x, Tensor $h_prev): array
    {
        $B = $x->shape[0];
        $I = $x->shape[1];
        $H = $this->hidden_size;

        // ── FFI boundary assertions ────────────────────────────────────────
        if (count($x->shape) !== 2 || $I !== $this->input_size) {
            throw new \InvalidArgumentException(
                "RNNCell::forward(): x must be [B, {$this->input_size}], got ["
                . implode(', ', $x->shape) . '].'
            );
        }
        if (count($h_prev->shape) !== 2 || $h_prev->shape[0] !== $B || $h_prev->shape[1] !== $H) {
            throw new \InvalidArgumentException(
                "RNNCell::forward(): h_prev must be [{$B}, {$H}], got ["
                . implode(', ', $h_prev->shape) . '].'
            );
        }

        $blas = BlasEngine::get()->ffi;

        // ── pre_act [B, H] = x @ W_ih^T  (sgemm NoTrans, Trans, B, H, I) ─
        $preAct = new Tensor([$B, $H]);
        $blas->cblas_sgemm(101, 111, 112, $B, $H, $I,
            1.0, $x->buffer, $I, $this->W_ih->buffer, $I,
            0.0, $preAct->buffer, $H);

        // ── pre_act += h_prev @ W_hh^T  (sgemm NoTrans, Trans, B, H, H) ──
        $blas->cblas_sgemm(101, 111, 112, $B, $H, $H,
            1.0, $h_prev->buffer, $H, $this->W_hh->buffer, $H,
            1.0, $preAct->buffer, $H);

        // ── pre_act += b_ih + b_hh  (broadcast over batch) ────────────────
        Ops::addBiasInPlace($preAct, $this->b_ih);
        Ops::addBiasInPlace($preAct, $this->b_hh);

        // ── h_t = tanh(pre_act)  (element-wise) ───────────────────────────
        $h = new Tensor([$B, $H]);
        for ($i = 0; $i < $B * $H; $i++) {
            $h->buffer[$i] = (float) tanh((float) $preAct->buffer[$i]);
        }

        // Cache stores what backward needs to recompute gradients
        $cache = [
            'x'       => $x,
            'h_prev'  => $h_prev,
            'pre_act' => $preAct,
            'h'       => $h,
            'B'       => $B,
            'I'       => $I,
            'H'       => $H,
        ];

        return [$h, $cache];
    }

    /**
     * Zero-state initializer: returns a [batch_size, hidden_size] zeros tensor.
     */
    public function zeroState(int $batchSize): Tensor
    {
        return Tensor::zeros([$batchSize, $this->hidden_size]);
    }

    // ── Explicit BPTT backward ────────────────────────────────────────────

    /**
     * Backpropagate through one time step.
     *
     * Accumulates gradients into W_ih->grad, W_hh->grad, b_ih->grad, b_hh->grad.
     * Returns (dx [B,I], dh_prev [B,H]) for propagation to prior steps and inputs.
     *
     * @param Tensor $dh    Upstream gradient of h_t — shape [batch_size, hidden_size].
     * @param array  $cache The cache array returned by forward().
     * @return array{Tensor, Tensor}  [dx [B,I], dh_prev [B,H]]
     */
    public function backward(Tensor $dh, array $cache): array
    {
        $x      = $cache['x'];
        $h_prev = $cache['h_prev'];
        $h      = $cache['h'];
        $B      = $cache['B'];
        $I      = $cache['I'];
        $H      = $cache['H'];

        $blas = BlasEngine::get()->ffi;

        // ── d_pre [B, H] = dh ⊙ (1 − h²)  (tanh backward) ────────────────
        $dPre = new Tensor([$B, $H]);
        for ($i = 0; $i < $B * $H; $i++) {
            $ht             = (float) $h->buffer[$i];
            $dPre->buffer[$i] = (float) $dh->buffer[$i] * (1.0 - $ht * $ht);
        }

        // ── Ensure gradient buffers exist ──────────────────────────────────
        $this->W_ih->initGrad();
        $this->W_hh->initGrad();
        $this->b_ih->initGrad();
        $this->b_hh->initGrad();

        // ── dW_ih [H, I] += d_pre^T @ x  (sgemm Trans, NoTrans, H, I, B) ─
        //   d_pre is [B×H], lda=H; x is [B×I], ldb=I
        $blas->cblas_sgemm(101, 112, 111, $H, $I, $B,
            1.0, $dPre->buffer, $H, $x->buffer, $I,
            1.0, $this->W_ih->grad, $I);

        // ── dW_hh [H, H] += d_pre^T @ h_prev  (sgemm Trans, NoTrans, H,H,B) ─
        $blas->cblas_sgemm(101, 112, 111, $H, $H, $B,
            1.0, $dPre->buffer, $H, $h_prev->buffer, $H,
            1.0, $this->W_hh->grad, $H);

        // ── db_ih [H] += Σ_b d_pre[b,:]  (sgemv Trans, B, H) ─────────────
        //   sgemv(RowMajor, Trans, M=B, N=H, 1.0, d_pre, H, ones_B, 1, 1.0, db, 1)
        $onesB = Tensor::ones([$B]);
        $blas->cblas_sgemv(101, 112, $B, $H,
            1.0, $dPre->buffer, $H, $onesB->buffer, 1,
            1.0, $this->b_ih->grad, 1);

        // ── db_hh [H] += same (b_hh sees the same pre_act signal) ─────────
        $blas->cblas_sgemv(101, 112, $B, $H,
            1.0, $dPre->buffer, $H, $onesB->buffer, 1,
            1.0, $this->b_hh->grad, 1);

        // ── dx [B, I] = d_pre @ W_ih  (sgemm NoTrans, NoTrans, B, I, H) ──
        $dx = new Tensor([$B, $I]);
        $blas->cblas_sgemm(101, 111, 111, $B, $I, $H,
            1.0, $dPre->buffer, $H, $this->W_ih->buffer, $I,
            0.0, $dx->buffer, $I);

        // ── dh_prev [B, H] = d_pre @ W_hh  (sgemm NoTrans, NoTrans, B, H, H) ─
        $dh_prev = new Tensor([$B, $H]);
        $blas->cblas_sgemm(101, 111, 111, $B, $H, $H,
            1.0, $dPre->buffer, $H, $this->W_hh->buffer, $H,
            0.0, $dh_prev->buffer, $H);

        return [$dx, $dh_prev];
    }

    /**
     * Return all learnable parameter tensors (for optimizer and gradient clipping).
     * @return Tensor[]
     */
    public function parameters(): array
    {
        return [$this->W_ih, $this->W_hh, $this->b_ih, $this->b_hh];
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
}
