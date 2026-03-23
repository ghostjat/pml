<?php

declare(strict_types=1);

namespace Pml\Training;

use Pml\{Tensor, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  AdamW Optimizer
//
//  Implements the AdamW algorithm (Loshchilov & Hutter, 2019,
//  "Decoupled Weight Decay Regularization") as used by GPT-2/3 and nearly
//  all modern language model training runs.
//
//  Update rule per parameter θ, gradient g:
//
//    m_t  = β1 * m_{t-1}  +  (1 − β1) * g_t              (1st moment)
//    v_t  = β2 * v_{t-1}  +  (1 − β2) * g_t²             (2nd moment)
//    m̂_t  = m_t  / (1 − β1^t)                            (bias correction)
//    v̂_t  = v_t  / (1 − β2^t)                            (bias correction)
//    θ_t  = (1 − lr * λ) * θ_{t-1}  −  lr * m̂_t / (√v̂_t + ε)
//
//  The crucial difference vs Adam+L2:
//    AdamW: weight decay is applied DIRECTLY to the weights, multiplicatively.
//    Adam+L2: weight decay is added to the GRADIENT, which distorts the
//             adaptive learning rate and makes λ scale-dependent.
//  AdamW's decoupled form makes λ interpretable and hyperparameter-stable.
//
//  Memory:
//    Two additional float32 buffers per parameter (m and v), allocated once
//    at construction and GC-freed when the optimizer goes out of scope.
//    For a 3M-parameter model: 2 × 3M × 4 bytes = 24 MB overhead.
//
//  PHP loop policy:
//    The AdamW update involves a per-element nonlinear operation (sqrt, divide
//    by adaptive term) that BLAS cannot express.  The inner loop over FFI
//    CData indices is therefore the only permitted PHP loop here, per the
//    architecture contract.  Element access via $buf[$i] goes through the FFI
//    boundary but avoids any PHP-level array allocation.
// ═══════════════════════════════════════════════════════════════════════════

final class AdamW
{
    /** Step counter — used for bias correction. Starts at 0, incremented before each update. */
    private int $t = 0;

    /**
     * Per-parameter moment buffers.
     * Indexed the same as $params.
     *
     * @var array<int, array{m: \FFI\CData, v: \FFI\CData}>
     */
    private array $state = [];

    /**
     * @param Tensor[] $params      Learnable parameter tensors (those with requiresGrad=true).
     *                              Pass them in the same order every time — the optimizer
     *                              identifies them by their position in this array.
     * @param float    $lr          Learning rate η.  GPT-2 uses 3e-4; NanoGPT uses 3e-4–1e-3.
     * @param float    $beta1       Decay for first moment  (running mean of gradients).
     * @param float    $beta2       Decay for second moment (running variance of gradients).
     * @param float    $eps         Denominator stability term.  1e-8 is standard.
     * @param float    $weightDecay Decoupled weight decay coefficient λ.
     *                              0.1 for GPT-2 (applied only to non-bias, non-norm params).
     */
    public function __construct(
        private readonly array $params,
        private readonly float $lr          = 3e-4,
        private readonly float $beta1       = 0.9,
        private readonly float $beta2       = 0.999,
        private readonly float $eps         = 1e-8,
        private readonly float $weightDecay = 0.1
    ) {
        $blas = BlasEngine::get();

        foreach ($this->params as $i => $param) {
            if (!$param instanceof Tensor) {
                throw new \InvalidArgumentException(
                    "AdamW: params[{$i}] must be a Tensor, got " . gettype($param)
                );
            }
            // Allocate zeroed moment buffers — same element count as the weight.
            // FFI::new zero-initialises, so m_0 = v_0 = 0 as the algorithm requires.
            $this->state[$i] = [
                'm' => $blas->allocFloat($param->size, true), // 1st moment (mean)
                'v' => $blas->allocFloat($param->size, true), // 2nd moment (variance)
            ];
        }
    }

    /**
     * Perform one AdamW parameter update step.
     *
     * Call this AFTER loss->backward() has populated all $param->grad buffers
     * and BEFORE zeroGrad().  The weight update is applied in-place to
     * each parameter's $buffer.
     *
     * Steps:
     *   1. Increment step counter t (for bias correction).
     *   2. Pre-compute bias correction denominators (scalar, not per-element).
     *   3. For each parameter with a non-null grad buffer:
     *        a. Update m and v moment estimates.
     *        b. Apply bias correction.
     *        c. Apply AdamW weight update (weight decay + adaptive gradient step).
     */
    public function step(): void
    {
        // ── 1. Increment step counter ────────────────────────────────────
        $this->t++;

        // ── 2. Bias correction scalars ───────────────────────────────────
        // These cancel the zero-initialisation bias in m and v at early steps.
        // At t=1, bc1 = 1−β1, bc2 = 1−β2; they approach 1.0 as t → ∞.
        $bc1 = 1.0 - ($this->beta1 ** $this->t);
        $bc2 = 1.0 - ($this->beta2 ** $this->t);

        // Cache hyperparams as locals to avoid $this dereference in the loop
        $lr    = $this->lr;
        $beta1 = $this->beta1;
        $beta2 = $this->beta2;
        $eps   = $this->eps;
        $wd    = $this->weightDecay;

        // ── 3. Per-parameter update ──────────────────────────────────────
        foreach ($this->params as $i => $param) {
            if ($param->grad === null) {
                // This parameter received no gradient (e.g. unused layer).
                continue;
            }

            $w = $param->buffer; // weight buffer (float[n])
            $g = $param->grad;   // gradient buffer (float[n])
            $m = $this->state[$i]['m']; // 1st moment (float[n])
            $v = $this->state[$i]['v']; // 2nd moment (float[n])
            $n = $param->size;

            // ── Inner loop: AdamW math per element ───────────────────────
            //
            // PHP loop over FFI CData indices is permitted here because:
            //   (a) The update is element-wise nonlinear (sqrt, /).
            //   (b) BLAS has no primitive for this operation.
            //   (c) The loop body is purely arithmetic — no PHP allocations.
            //
            // Each iteration:
            //   g_j = grad[j]
            //   m_j = β1·m_j + (1−β1)·g_j       → biased 1st moment
            //   v_j = β2·v_j + (1−β2)·g_j²       → biased 2nd moment
            //   m̂_j = m_j / bc1                  → bias-corrected 1st moment
            //   v̂_j = v_j / bc2                  → bias-corrected 2nd moment
            //   w_j = (1 − lr·λ)·w_j − lr·m̂_j/(√v̂_j + ε)
            //          └─── weight decay ───┘   └─── Adam step ────────┘
            for ($j = 0; $j < $n; $j++) {
                $gj = (float) $g[$j];

                // ── 1st moment: exponential moving average of gradient ────
                $mj = $beta1 * (float) $m[$j] + (1.0 - $beta1) * $gj;
                $m[$j] = $mj;

                // ── 2nd moment: EMA of squared gradient ──────────────────
                $vj = $beta2 * (float) $v[$j] + (1.0 - $beta2) * $gj * $gj;
                $v[$j] = $vj;

                // ── Bias-corrected estimates ──────────────────────────────
                $mHat = $mj / $bc1; // approaches true mean as t grows
                $vHat = $vj / $bc2; // approaches true variance as t grows

                // ── AdamW weight update ───────────────────────────────────
                // The (1 − lr·λ) factor is decoupled weight decay: it shrinks
                // the weight toward zero independently of the gradient magnitude,
                // which is the key advantage of AdamW over Adam + L2.
                $w[$j] = (1.0 - $lr * $wd) * (float) $w[$j]
                       - $lr * $mHat / (sqrt($vHat) + $eps);
            }
        }
    }

    /**
     * Zero all parameter gradient buffers.
     *
     * Call this AFTER step() so accumulated gradients do not bleed into the
     * next forward–backward pass.  Internally uses FFI memset (byte-zero =
     * IEEE 754 0.0f) for maximum throughput.
     */
    public function zeroGrad(): void
    {
        foreach ($this->params as $param) {
            $param->zeroGrad();
        }
    }

    /**
     * Reset the optimizer state (moments + step counter).
     *
     * Useful when restarting training from a checkpoint with a new learning
     * rate schedule — warm-starting the moments from zero avoids the stale
     * gradient bias from the previous run.
     */
    public function resetState(): void
    {
        $this->t = 0;
        foreach ($this->state as $state) {
            \FFI::memset($state['m'], 0, \FFI::sizeof($state['m']));
            \FFI::memset($state['v'], 0, \FFI::sizeof($state['v']));
        }
    }

    /** Current step count (useful for learning rate scheduling). */
    public function stepCount(): int { return $this->t; }

    /**
     * Force-set the step counter, e.g. when resuming from a checkpoint.
     * This restores the correct bias-correction denominators (β1^t, β2^t)
     * so AdamW doesn't re-run the warm-up phase after a resume.
     */
    public function setStep(int $step): void { $this->t = $step; }

    // ── Global gradient norm clipping ────────────────────────────────────

    /**
     * Clip the global gradient L2-norm to $maxNorm (in-place on all grad buffers).
     *
     * Algorithm:
     *   totalNorm = √(Σ_p ||grad_p||²)           (global L2 norm)
     *   if totalNorm > maxNorm:
     *       scale = maxNorm / totalNorm
     *       grad_p *= scale  for each parameter p
     *
     * Why global clipping (not per-parameter)?
     *   Per-parameter clipping changes gradient DIRECTION.  Global clipping
     *   only changes MAGNITUDE while preserving the direction of the full
     *   gradient vector — mathematically equivalent to projecting onto the
     *   ball of radius maxNorm in parameter space.
     *
     * Call immediately BEFORE step() and AFTER the last backward() call.
     *
     * @param Tensor[] $parameters   All trainable parameters (same list passed to __construct).
     * @param float    $maxNorm      Clip threshold.  1.0 is a safe default for transformers.
     * @return float                 The pre-clipping global norm (useful for logging).
     */
    public static function clipGradNorm(array $parameters, float $maxNorm = 1.0): float
    {
        $ffi         = BlasEngine::get()->ffi;
        $totalNormSq = 0.0;

        // ── Pass 1: accumulate sum of squared L2 norms ────────────────────
        //
        // cblas_sdot(n, x, 1, x, 1) = x·x = Σ x[i]²
        // This is equivalent to ||x||² and avoids a separate cblas_snrm2 call,
        // matching the pattern already used in the codebase.
        foreach ($parameters as $param) {
            if ($param->grad === null) {
                continue;
            }
            // sdot of a vector with itself = sum of squares = ||grad||²
            $totalNormSq += (float) $ffi->cblas_sdot(
                $param->size,
                $param->grad, 1,
                $param->grad, 1
            );
        }

        $totalNorm = sqrt($totalNormSq);

        // ── Pass 2: scale all grad buffers if over the threshold ──────────
        //
        // We only scale when strictly over maxNorm — avoids a no-op sscal
        // on every step when gradients are already small.
        if ($totalNorm > $maxNorm) {
            $scale = $maxNorm / $totalNorm;

            foreach ($parameters as $param) {
                if ($param->grad === null) {
                    continue;
                }
                // cblas_sscal(n, scale, x, 1): x *= scale  (in-place)
                $ffi->cblas_sscal($param->size, $scale, $param->grad, 1);
            }
        }

        return $totalNorm; // caller may log this
    }

    // ── Checkpoint state serialisation ───────────────────────────────────

    /**
     * Export the optimizer's m/v moment buffers as named Tensor views.
     *
     * The returned Tensors are VIEWS over the optimizer's internal CData
     * buffers — they share memory and must not outlive this AdamW instance.
     * Pass the result directly to SafetensorsWriter::write(); the buffers
     * are copied to disk before the Tensors go out of scope.
     *
     * Naming convention:
     *   "__opt_m__{paramName}"  — 1st moment (mean) buffer for $paramName
     *   "__opt_v__{paramName}"  — 2nd moment (variance) buffer for $paramName
     *
     * @param array<string, Tensor> $namedParams  Named parameter map from model->namedParams().
     *                                             Must be in the same order as the params array
     *                                             passed to the AdamW constructor.
     * @return array<string, Tensor>               Named 1D Tensor views of the m/v buffers.
     */
    public function getNamedState(array $namedParams): array
    {
        $state = [];
        $i     = 0;

        foreach ($namedParams as $name => $param) {
            if (isset($this->state[$i])) {
                // Wrap the raw CData float[] as a 1D Tensor view.
                // Tensor constructor accepts an existing CData buffer and does NOT
                // take ownership (the optimizer's $state array still holds the ref).
                $state["__opt_m__{$name}"] = new Tensor([$param->size], $this->state[$i]['m']);
                $state["__opt_v__{$name}"] = new Tensor([$param->size], $this->state[$i]['v']);
            }
            $i++;
        }

        return $state;
    }

    /**
     * Restore the optimizer's m/v moment buffers from previously loaded Tensors.
     *
     * Call this after SafetensorsLoader::load() has loaded a checkpoint and you
     * have identified the optimizer state tensors (those named "__opt_m__*" /
     * "__opt_v__*").  Each found tensor is copied element-by-element into the
     * corresponding internal buffer via cblas_scopy.
     *
     * Missing entries (e.g. for new parameters added since the checkpoint) are
     * left at zero — equivalent to a fresh warm-start for those parameters.
     *
     * @param array<string, Tensor> $loadedTensors  Full tensor map from SafetensorsLoader::load().
     * @param array<string, Tensor> $namedParams    Named parameter map from model->namedParams().
     */
    public function loadNamedState(array $loadedTensors, array $namedParams): void
    {
        $ffi = BlasEngine::get()->ffi;
        $i   = 0;

        foreach ($namedParams as $name => $param) {
            $mKey = "__opt_m__{$name}";
            $vKey = "__opt_v__{$name}";

            if (isset($loadedTensors[$mKey], $this->state[$i])) {
                // cblas_scopy: copy $param->size floats from loaded buffer → internal m buffer
                $ffi->cblas_scopy(
                    $param->size,
                    $loadedTensors[$mKey]->buffer, 1,
                    $this->state[$i]['m'],          1
                );
            }

            if (isset($loadedTensors[$vKey], $this->state[$i])) {
                $ffi->cblas_scopy(
                    $param->size,
                    $loadedTensors[$vKey]->buffer, 1,
                    $this->state[$i]['v'],          1
                );
            }

            $i++;
        }
    }
}
