<?php

declare(strict_types=1);

namespace Pml\Training;

use Pml\{Tensor, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  CrossEntropyLoss
//
//  Fused Softmax + Negative Log-Likelihood loss with an analytically-derived
//  backward pass.
//
//  Why fuse softmax and cross-entropy?
//  ─────────────────────────────────────
//  If you wrote them as two separate differentiable ops, the backward through
//  full Softmax Jacobian costs O(C²) per sample.  The combined derivative
//  collapses to a beautifully simple O(C) expression:
//
//      dL/dz_i  =  p_i  −  1{i == target}
//
//  where p = softmax(z) and 1{·} is the Kronecker delta (1 at target, 0 elsewhere).
//
//  Derivation (one sample):
//      L = −log p_t   where p_t = exp(z_t) / Σ exp(z_j)
//      ∂L/∂z_i = p_i − 1{i == t}
//  (Standard result; see Bishop PRML §4.3.4 or Goodfellow DL §6.2.2.)
//
//  Batched (seqLen samples, average loss):
//      L = (1/T) Σ_t −log p_{t, target_t}
//      ∂L/∂z_{t,i} = (1/T) (p_{t,i} − 1{i == target_t})
//
//  Supported logit shapes
//  ───────────────────────
//    • 1D [vocabSize]  + int  $target  → scalar loss
//    • 2D [seqLen, vocabSize] + int[] $targets → scalar loss (mean over seq)
// ═══════════════════════════════════════════════════════════════════════════

class CrossEntropyLoss
{
    /**
     * Compute the cross-entropy loss and register a backward closure.
     *
     * The returned scalar Tensor (size=1) is the mean negative log-likelihood.
     * Call ->backward() on it to populate $logits->grad with dL/dLogits.
     *
     * @param Tensor    $logits  [vocabSize] OR [seqLen, vocabSize] — raw scores
     *                           (pre-softmax; float32).  Should have
     *                           requiresGrad=true so the backward is useful.
     * @param int|int[] $targets Single class index OR array of class indices
     *                           (one per row when logits is 2D).
     * @return Tensor            Scalar loss tensor (size=1, requiresGrad=true).
     */
    public function forward(Tensor $logits, int|array $targets): Tensor
    {
        // ── Normalise inputs to 2D ─────────────────────────────────────────
        // IMPORTANT: keep $origLogits pointing to the Tensor object the caller
        // holds.  reshape() creates a NEW Tensor object sharing the same C
        // buffer; if we overwrote $logits and later called initGrad() on the
        // reshape view, the grad would live on that shadow object — invisible
        // to the caller.  All gradient I/O goes through $origLogits.
        $origLogits = $logits;
        $is1D       = (count($logits->shape) === 1);
        if ($is1D) {
            // View the 1D logit vector as a single-row 2D batch
            $logits  = $origLogits->reshape([1, $origLogits->size]);
            $targets = [$targets];
        }

        [$seqLen, $vocabSize] = $logits->shape;

        if (count($targets) !== $seqLen) {
            throw new \InvalidArgumentException(
                "CrossEntropyLoss: targets count ({$seqLen}) must match logits rows "
                . '(' . count($targets) . ').'
            );
        }

        // ── Forward: numerically stable per-row softmax ────────────────────
        //
        // For each row t:
        //   1. max_t = max(z_{t,·})           — numerical stability
        //   2. e_{t,i} = exp(z_{t,i} − max_t) — shifted exp
        //   3. p_{t,i} = e_{t,i} / Σ_j e_{t,j}  — normalise
        //   4. L_t = −log(p_{t, target_t})    — NLL for this token

        // We store the full probability matrix [seqLen, vocabSize] in a
        // GC-owned F32 buffer — the backward closure captures it to compute
        // the gradient without re-running the forward.
        $probs = Tensor::zeros([$seqLen, $vocabSize]);
        $ffi   = BlasEngine::get()->ffi;
        $loss  = 0.0;

        for ($t = 0; $t < $seqLen; $t++) {
            $offset = $t * $vocabSize;

            // ── Step 1: find row max ──────────────────────────────────────
            $maxVal = (float) $logits->buffer[$offset];
            for ($i = 1; $i < $vocabSize; $i++) {
                $v = (float) $logits->buffer[$offset + $i];
                if ($v > $maxVal) { $maxVal = $v; }
            }

            // ── Step 2 & 3: exp(z − max) then normalise ───────────────────
            $sum = 0.0;
            for ($i = 0; $i < $vocabSize; $i++) {
                $e = exp((float) $logits->buffer[$offset + $i] - $maxVal);
                $probs->buffer[$offset + $i] = $e;
                $sum += $e;
            }
            // cblas_sscal: probs[row] *= (1/sum)  — no PHP loop over floats
            $rowPtr = \FFI::cast('float*', \FFI::addr($probs->buffer[$offset]));
            $ffi->cblas_sscal($vocabSize, 1.0 / $sum, $rowPtr, 1);

            // ── Step 4: NLL ───────────────────────────────────────────────
            $tgt  = (int) $targets[$t];
            $prob = (float) $probs->buffer[$offset + $tgt];
            // Clamp to avoid log(0); in practice softmax output is > 1e-37
            $loss += -log(max($prob, 1e-37));
        }

        // Mean over the sequence length
        $loss /= $seqLen;

        // ── Build the output scalar Tensor ─────────────────────────────────
        $lossT = Tensor::full([1], $loss);
        $lossT->requiresGrad = true;
        $lossT->_prev        = [$origLogits]; // original logits tensor — grad accumulates here

        // ── Register backward closure ──────────────────────────────────────
        //
        // Combined Softmax + Cross-Entropy gradient:
        //
        //   dL/dz_{t,i} = (1/T) * (p_{t,i} − 1{i == target_t})
        //
        // Scaled by the upstream gradient ($lossT->grad[0]) so this integrates
        // correctly when the loss is itself inside a larger graph (e.g. if you
        // average losses across micro-batches).
        //
        // Implementation: one PHP loop over T*C elements — permitted by the
        // architecture contract because there is no BLAS primitive for this
        // fused gradient computation.
        $lossT->_backward = static function()
            use ($origLogits, $lossT, $probs, $targets, $seqLen, $vocabSize): void
        {
            if (!$origLogits->requiresGrad) {
                return;
            }

            $origLogits->initGrad();

            // Upstream gradient scale.  For a standalone loss, grad[0] = 1.0.
            // Multiply by 1/seqLen for the mean-reduction.
            $scale = (float) $lossT->grad[0] / $seqLen;

            for ($t = 0; $t < $seqLen; $t++) {
                $offset = $t * $vocabSize;
                $tgt    = (int) $targets[$t];

                for ($i = 0; $i < $vocabSize; $i++) {
                    // dL/dz_{t,i} += scale * (p_{t,i} − 1{i == tgt})
                    $g = (float) $probs->buffer[$offset + $i];
                    if ($i === $tgt) { $g -= 1.0; }
                    $origLogits->grad[$offset + $i] = (float) $origLogits->grad[$offset + $i] + $g * $scale;
                }
            }
        };

        return $lossT;
    }
}
