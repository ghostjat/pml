<?php

declare(strict_types=1);

namespace Pml\Layers;

use Pml\{Tensor, Ops, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  LSTMCell — single time-step Long Short-Term Memory cell
//
//  Computes one LSTM step with four gated updates:
//
//    gates [B, 4H] = x W_{ih}^T + h_{t-1} W_{hh}^T + b_{ih} + b_{hh}
//
//    i_t = σ(gates[:, 0:H])       input gate
//    f_t = σ(gates[:, H:2H])      forget gate
//    g_t = tanh(gates[:, 2H:3H])  cell gate
//    o_t = σ(gates[:, 3H:4H])     output gate
//
//    c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t
//    h_t = o_t ⊙ tanh(c_t)
//
//  ── Packed Weight Layout ─────────────────────────────────────────────────
//
//  All four gate weights are PACKED into a single matrix, matching PyTorch's
//  layout.  This allows computing all four gates in a single sgemm call:
//
//    W_ih [4H, I]  row order: [W_i; W_f; W_g; W_o]
//    W_hh [4H, H]  row order: [W_i; W_f; W_g; W_o]
//    b_ih [4H]     bias order: [b_i; b_f; b_g; b_o]
//    b_hh [4H]     same
//
//  ── Forward BLAS ─────────────────────────────────────────────────────────
//
//    gates = x @ W_ih^T          sgemm(NoTrans, Trans, B, 4H, I)
//          + h_prev @ W_hh^T     sgemm(NoTrans, Trans, B, 4H, H)  (beta=1)
//    gates += b_ih                addBiasInPlace
//    gates += b_hh                addBiasInPlace
//
//  ── Explicit BPTT Backward ───────────────────────────────────────────────
//
//  Given upstream (dh_t [B,H], dc_t [B,H]) — dc_t is the gradient of the
//  cell state from the next time step (zero at the last step):
//
//    do      = dh_t ⊙ tanh(c_t)
//    d_tc    = dh_t ⊙ o_t ⊙ (1 − tanh(c_t)²)
//    dc_eff  = dc_t + d_tc                       combined cell-state gradient
//    df      = dc_eff ⊙ c_{t-1}
//    di      = dc_eff ⊙ g_t
//    dg      = dc_eff ⊙ i_t
//    dc_prev = dc_eff ⊙ f_t
//
//    d_gates [B, 4H] packed:
//      [:, 0:H]   = di ⊙ i_t ⊙ (1−i_t)     sigmoid backward for i
//      [:, H:2H]  = df ⊙ f_t ⊙ (1−f_t)     sigmoid backward for f
//      [:, 2H:3H] = dg ⊙ (1−g_t²)          tanh backward for g
//      [:, 3H:4H] = do ⊙ o_t ⊙ (1−o_t)     sigmoid backward for o
//
//    dW_ih  [4H, I] += d_gates^T @ x         sgemm(Trans, NoTrans, 4H, I, B)
//    dW_hh  [4H, H] += d_gates^T @ h_prev    sgemm(Trans, NoTrans, 4H, H, B)
//    db_ih  [4H]    += Σ_b d_gates           sgemv(Trans, B, 4H)
//    db_hh  [4H]    += same
//
//    dx     [B, I]  = d_gates @ W_ih         sgemm(NoTrans, NoTrans, B, I, 4H)
//    dh_prev[B, H]  = d_gates @ W_hh         sgemm(NoTrans, NoTrans, B, H, 4H)
//
//  Return: (dx [B,I], dh_prev [B,H], dc_prev [B,H])
//
//  ── Complexity ────────────────────────────────────────────────────────────
//
//  Forward per step:  O(B·I·4H + B·H·4H)   two sgemm calls
//  Backward per step: O(B·I·4H + B·H·4H)   two more sgemm calls
// ═══════════════════════════════════════════════════════════════════════════

final class LSTMCell
{
    // ── Learnable parameters ──────────────────────────────────────────────

    /** Packed input-to-hidden weights [4*hidden_size, input_size]. */
    public readonly Tensor $W_ih;

    /** Packed hidden-to-hidden weights [4*hidden_size, hidden_size]. */
    public readonly Tensor $W_hh;

    /** Packed input-to-hidden bias [4*hidden_size]. */
    public readonly Tensor $b_ih;

    /** Packed hidden-to-hidden bias [4*hidden_size]. */
    public readonly Tensor $b_hh;

    // ── Dimensions ───────────────────────────────────────────────────────

    public readonly int $input_size;
    public readonly int $hidden_size;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int   $input_size   Dimension of x_t.
     * @param int   $hidden_size  Dimension of h_t and c_t.
     * @param float $initStd      Weight init stddev. Default: 1/sqrt(hidden_size).
     */
    public function __construct(
        int   $input_size,
        int   $hidden_size,
        float $initStd = 0.0,
    ) {
        $this->input_size  = $input_size;
        $this->hidden_size = $hidden_size;

        if ($initStd <= 0.0) {
            $initStd = 1.0 / sqrt((float) $hidden_size);
        }

        $H4 = 4 * $hidden_size;

        $this->W_ih = Tensor::randn([$H4, $input_size],  0.0, $initStd);
        $this->W_hh = Tensor::randn([$H4, $hidden_size], 0.0, $initStd);
        $this->b_ih = Tensor::zeros([$H4]);
        $this->b_hh = Tensor::zeros([$H4]);

        // PyTorch-style forget gate bias initialisation (b_f += 1)
        // Helps prevent the cell from forgetting at the start of training.
        for ($k = $hidden_size; $k < 2 * $hidden_size; $k++) {
            $this->b_ih->buffer[$k] = 1.0;
        }

        foreach ([$this->W_ih, $this->W_hh, $this->b_ih, $this->b_hh] as $p) {
            $p->requiresGrad = true;
        }
    }

    // ── Forward pass ──────────────────────────────────────────────────────

    /**
     * Compute one LSTM time step.
     *
     * @param Tensor $x      Input — [batch_size, input_size].
     * @param Tensor $h_prev Previous hidden state — [batch_size, hidden_size].
     * @param Tensor $c_prev Previous cell state   — [batch_size, hidden_size].
     * @return array{Tensor, Tensor, array}  [h_t [B,H], c_t [B,H], cache]
     */
    public function forward(Tensor $x, Tensor $h_prev, Tensor $c_prev): array
    {
        $B  = $x->shape[0];
        $I  = $this->input_size;
        $H  = $this->hidden_size;
        $H4 = 4 * $H;

        // ── FFI boundary assertions ────────────────────────────────────────
        if (count($x->shape) !== 2 || $x->shape[1] !== $I) {
            throw new \InvalidArgumentException(
                "LSTMCell::forward(): x must be [B, {$I}], got ["
                . implode(', ', $x->shape) . '].'
            );
        }
        if ($h_prev->shape !== [$B, $H]) {
            throw new \InvalidArgumentException(
                "LSTMCell::forward(): h_prev must be [{$B}, {$H}], got ["
                . implode(', ', $h_prev->shape) . '].'
            );
        }
        if ($c_prev->shape !== [$B, $H]) {
            throw new \InvalidArgumentException(
                "LSTMCell::forward(): c_prev must be [{$B}, {$H}], got ["
                . implode(', ', $c_prev->shape) . '].'
            );
        }

        $blas = BlasEngine::get()->ffi;

        // ── gates [B, 4H] = x @ W_ih^T  (sgemm NoTrans, Trans, B, 4H, I) ─
        $gates = new Tensor([$B, $H4]);
        $blas->cblas_sgemm(101, 111, 112, $B, $H4, $I,
            1.0, $x->buffer, $I, $this->W_ih->buffer, $I,
            0.0, $gates->buffer, $H4);

        // ── gates += h_prev @ W_hh^T  (sgemm NoTrans, Trans, B, 4H, H) ───
        $blas->cblas_sgemm(101, 111, 112, $B, $H4, $H,
            1.0, $h_prev->buffer, $H, $this->W_hh->buffer, $H,
            1.0, $gates->buffer, $H4);

        // ── gates += b_ih + b_hh ──────────────────────────────────────────
        Ops::addBiasInPlace($gates, $this->b_ih);
        Ops::addBiasInPlace($gates, $this->b_hh);

        // ── Split and activate: i, f, g, o gates ──────────────────────────
        //
        // Gate order within each row of gates [4H]:
        //   [0 : H)     → input gate   (sigmoid)
        //   [H : 2H)    → forget gate  (sigmoid)
        //   [2H : 3H)   → cell gate    (tanh)
        //   [3H : 4H)   → output gate  (sigmoid)
        $i_gate = new Tensor([$B, $H]);
        $f_gate = new Tensor([$B, $H]);
        $g_gate = new Tensor([$B, $H]);
        $o_gate = new Tensor([$B, $H]);

        for ($b = 0; $b < $B; $b++) {
            $off = $b * $H4;  // offset into gates buffer for batch b

            for ($j = 0; $j < $H; $j++) {
                $rowBase = $b * $H + $j;
                $v_i = (float) $gates->buffer[$off + $j];
                $v_f = (float) $gates->buffer[$off + $H + $j];
                $v_g = (float) $gates->buffer[$off + 2 * $H + $j];
                $v_o = (float) $gates->buffer[$off + 3 * $H + $j];

                $i_gate->buffer[$rowBase] = 1.0 / (1.0 + exp(-$v_i));  // sigmoid
                $f_gate->buffer[$rowBase] = 1.0 / (1.0 + exp(-$v_f));  // sigmoid
                $g_gate->buffer[$rowBase] = (float) tanh($v_g);         // tanh
                $o_gate->buffer[$rowBase] = 1.0 / (1.0 + exp(-$v_o));  // sigmoid
            }
        }

        // ── c_t = f ⊙ c_prev + i ⊙ g ─────────────────────────────────────
        $c = new Tensor([$B, $H]);
        for ($k = 0; $k < $B * $H; $k++) {
            $c->buffer[$k] =
                (float) $f_gate->buffer[$k] * (float) $c_prev->buffer[$k]
                + (float) $i_gate->buffer[$k] * (float) $g_gate->buffer[$k];
        }

        // ── tanh(c_t) ─────────────────────────────────────────────────────
        $tanh_c = new Tensor([$B, $H]);
        for ($k = 0; $k < $B * $H; $k++) {
            $tanh_c->buffer[$k] = (float) tanh((float) $c->buffer[$k]);
        }

        // ── h_t = o ⊙ tanh(c_t) ───────────────────────────────────────────
        $h = new Tensor([$B, $H]);
        for ($k = 0; $k < $B * $H; $k++) {
            $h->buffer[$k] = (float) $o_gate->buffer[$k] * (float) $tanh_c->buffer[$k];
        }

        $cache = [
            'x'      => $x,
            'h_prev' => $h_prev,
            'c_prev' => $c_prev,
            'gates'  => $gates,
            'i'      => $i_gate,
            'f'      => $f_gate,
            'g'      => $g_gate,
            'o'      => $o_gate,
            'c'      => $c,
            'tanh_c' => $tanh_c,
            'h'      => $h,
            'B'      => $B,
            'I'      => $I,
            'H'      => $H,
            'H4'     => $H4,
        ];

        return [$h, $c, $cache];
    }

    /**
     * Zero-state initializer: returns [h0 [B,H], c0 [B,H]].
     *
     * @return array{Tensor, Tensor}
     */
    public function zeroState(int $batchSize): array
    {
        return [
            Tensor::zeros([$batchSize, $this->hidden_size]),
            Tensor::zeros([$batchSize, $this->hidden_size]),
        ];
    }

    // ── Explicit BPTT backward ────────────────────────────────────────────

    /**
     * Backpropagate through one LSTM time step.
     *
     * Accumulates gradients into W_ih, W_hh, b_ih, b_hh.
     * Returns (dx, dh_prev, dc_prev) for propagation to earlier steps.
     *
     * @param Tensor $dh    Upstream gradient of h_t — [B, H].
     * @param Tensor $dc    Upstream gradient of c_t — [B, H].
     *                      Zero for the final step in an unrolled sequence.
     * @param array  $cache The cache array returned by forward().
     * @return array{Tensor, Tensor, Tensor}  [dx [B,I], dh_prev [B,H], dc_prev [B,H]]
     */
    public function backward(Tensor $dh, Tensor $dc, array $cache): array
    {
        $x      = $cache['x'];
        $h_prev = $cache['h_prev'];
        $c_prev = $cache['c_prev'];
        $i      = $cache['i'];
        $f      = $cache['f'];
        $g      = $cache['g'];
        $o      = $cache['o'];
        $c      = $cache['c'];
        $tanh_c = $cache['tanh_c'];
        $B      = $cache['B'];
        $I      = $cache['I'];
        $H      = $cache['H'];
        $H4     = $cache['H4'];
        $blas   = BlasEngine::get()->ffi;

        // ── Unpack the output-gate gradient ───────────────────────────────
        //
        // h_t = o ⊙ tanh(c_t)
        //   do      = dh ⊙ tanh(c_t)
        //   d_tanh_c = dh ⊙ o
        //
        // dc_eff = dc  +  dh ⊙ o ⊙ (1 − tanh²(c_t))   combined cell grad

        $do      = new Tensor([$B, $H]);
        $dc_eff  = new Tensor([$B, $H]);
        $dc_prev = new Tensor([$B, $H]);

        for ($k = 0; $k < $B * $H; $k++) {
            $ov      = (float) $o->buffer[$k];
            $tv      = (float) $tanh_c->buffer[$k];
            $dhv     = (float) $dh->buffer[$k];
            $dcv     = (float) $dc->buffer[$k];

            $do->buffer[$k]     = $dhv * $tv;
            $dc_eff->buffer[$k] = $dcv + $dhv * $ov * (1.0 - $tv * $tv);
        }

        // ── Gate-state gradients ───────────────────────────────────────────
        //
        // c_t = f ⊙ c_{t-1} + i ⊙ g
        //   df      = dc_eff ⊙ c_{t-1}
        //   di      = dc_eff ⊙ g
        //   dg      = dc_eff ⊙ i
        //   dc_prev = dc_eff ⊙ f
        //
        // d_gates (pre-activation) [B, 4H] — packed [di; df; dg; do]:
        //   d_gates[:, 0:H]   = di ⊙ i ⊙ (1−i)    sigmoid backward
        //   d_gates[:, H:2H]  = df ⊙ f ⊙ (1−f)    sigmoid backward
        //   d_gates[:, 2H:3H] = dg ⊙ (1−g²)        tanh backward
        //   d_gates[:, 3H:4H] = do ⊙ o ⊙ (1−o)    sigmoid backward

        $dGates = new Tensor([$B, $H4]);

        for ($b = 0; $b < $B; $b++) {
            $off = $b * $H4;
            for ($j = 0; $j < $H; $j++) {
                $k = $b * $H + $j;

                $iv  = (float) $i->buffer[$k];
                $fv  = (float) $f->buffer[$k];
                $gv  = (float) $g->buffer[$k];
                $ov  = (float) $o->buffer[$k];
                $cpv = (float) $c_prev->buffer[$k];
                $dev = (float) $dc_eff->buffer[$k];

                $di_v = $dev * $gv;
                $df_v = $dev * $cpv;
                $dg_v = $dev * $iv;

                $dc_prev->buffer[$k] = $dev * $fv;

                // Pre-activation gradients (chain rule through activation)
                $dGates->buffer[$off + $j]           = $di_v * $iv * (1.0 - $iv);
                $dGates->buffer[$off + $H + $j]      = $df_v * $fv * (1.0 - $fv);
                $dGates->buffer[$off + 2 * $H + $j]  = $dg_v * (1.0 - $gv * $gv);
                $dGates->buffer[$off + 3 * $H + $j]  = (float) $do->buffer[$k] * $ov * (1.0 - $ov);
            }
        }

        // ── Ensure parameter gradient buffers exist ────────────────────────
        $this->W_ih->initGrad();
        $this->W_hh->initGrad();
        $this->b_ih->initGrad();
        $this->b_hh->initGrad();

        // ── dW_ih [4H, I] += d_gates^T @ x  (sgemm Trans, NoTrans, 4H, I, B) ─
        //   d_gates [B×4H], lda=4H; x [B×I], ldb=I
        $blas->cblas_sgemm(101, 112, 111, $H4, $I, $B,
            1.0, $dGates->buffer, $H4, $x->buffer, $I,
            1.0, $this->W_ih->grad, $I);

        // ── dW_hh [4H, H] += d_gates^T @ h_prev ──────────────────────────
        $blas->cblas_sgemm(101, 112, 111, $H4, $H, $B,
            1.0, $dGates->buffer, $H4, $h_prev->buffer, $H,
            1.0, $this->W_hh->grad, $H);

        // ── db_ih [4H] += Σ_b d_gates  (sgemv Trans, B, 4H) ─────────────
        $onesB = Tensor::ones([$B]);
        $blas->cblas_sgemv(101, 112, $B, $H4,
            1.0, $dGates->buffer, $H4, $onesB->buffer, 1,
            1.0, $this->b_ih->grad, 1);

        // ── db_hh [4H] += same ────────────────────────────────────────────
        $blas->cblas_sgemv(101, 112, $B, $H4,
            1.0, $dGates->buffer, $H4, $onesB->buffer, 1,
            1.0, $this->b_hh->grad, 1);

        // ── dx [B, I] = d_gates @ W_ih  (sgemm NoTrans, NoTrans, B, I, 4H) ─
        $dx = new Tensor([$B, $I]);
        $blas->cblas_sgemm(101, 111, 111, $B, $I, $H4,
            1.0, $dGates->buffer, $H4, $this->W_ih->buffer, $I,
            0.0, $dx->buffer, $I);

        // ── dh_prev [B, H] = d_gates @ W_hh  (sgemm NoTrans, NoTrans, B, H, 4H) ─
        $dh_prev = new Tensor([$B, $H]);
        $blas->cblas_sgemm(101, 111, 111, $B, $H, $H4,
            1.0, $dGates->buffer, $H4, $this->W_hh->buffer, $H,
            0.0, $dh_prev->buffer, $H);

        return [$dx, $dh_prev, $dc_prev];
    }

    /**
     * Return all learnable parameter tensors.
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
