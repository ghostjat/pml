<?php

declare(strict_types=1);

namespace Pml\Classic\SVM;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  SVR — sklearn.svm.SVR
//
//  ε-Support Vector Regression backed by libsvm (via LibSVMBridge FFI).
//
//  ── Algorithm Overview ───────────────────────────────────────────────────
//
//  SVR fits a function f(x) = w^T φ(x) + b such that most training outputs
//  y_i lie within an ε-tube around f(x_i):
//
//    |y_i − f(x_i)| ≤ ε  (ε-insensitive loss)
//
//  Samples outside the tube are penalised proportionally to their excess
//  deviation, controlled by C (regularisation strength).
//
//  The primal problem is:
//    min  ½||w||² + C · Σ_i (ξ_i + ξ_i*)
//    s.t. y_i − f(x_i) ≤ ε + ξ_i
//         f(x_i) − y_i ≤ ε + ξ_i*
//         ξ_i, ξ_i* ≥ 0
//
//  libsvm solves the dual, using the EPSILON_SVR svm_type.  The ε-tube
//  half-width is stored in svm_parameter.p.
//
//  ── Relationship to SVC ──────────────────────────────────────────────────
//
//  SVR differs from SVC only in:
//    1. svm_type = EPSILON_SVR (3) instead of C_SVC (0)
//    2. svm_parameter.p = epsilon  (ε-tube half-width)
//    3. No class discovery — y is a continuous regression target
//    4. svm_predict() returns a regression value, not a class label
//
//  Everything else (kernel choices, gamma computation, data marshalling,
//  model lifecycle, memory safety) is identical to SVC.
//
//  See SVC.php for detailed comments on the shared mechanics.
// ═══════════════════════════════════════════════════════════════════════════

final class SVR implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    public readonly int   $n_features_in_;
    public readonly float $gamma_;

    // ── Internal model handle ─────────────────────────────────────────────

    /**
     * void*[1] box for the svm_model* handle.
     * See SVC.php for the rationale.
     */
    private \FFI\CData $modelBox;
    private bool $fitted = false;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param float            $C          Regularisation strength.
     *                                     Smaller C allows more deviation beyond ε.
     * @param float            $epsilon    ε-tube half-width.
     *                                     Predictions within ε of the target are
     *                                     penalty-free (ε-insensitive loss).
     * @param string           $kernel     'rbf' | 'linear' | 'poly' | 'sigmoid'
     * @param int|float|string $gamma      RBF/poly/sigmoid coefficient.
     *                                     'scale' | 'auto' | float
     * @param int              $degree     Polynomial degree (poly kernel only).
     * @param float            $coef0      Intercept for poly/sigmoid kernels.
     * @param float            $tol        SMO stopping tolerance.
     * @param float            $cache_size Kernel cache in MB.
     * @param int              $max_iter   Maximum solver iterations (−1 = no limit).
     * @param bool             $shrinking  Whether to use shrinking heuristics.
     */
    public function __construct(
        private readonly float            $C          = 1.0,
        private readonly float            $epsilon    = 0.1,
        private readonly string           $kernel     = 'rbf',
        private readonly int|float|string $gamma      = 'scale',
        private readonly int              $degree     = 3,
        private readonly float            $coef0      = 0.0,
        private readonly float            $tol        = 1e-3,
        private readonly float            $cache_size = 200.0,
        private readonly int              $max_iter   = -1,
        private readonly bool             $shrinking  = true,
    ) {
        if ($C <= 0.0) {
            throw new \InvalidArgumentException('SVR: C must be > 0.');
        }
        if ($epsilon < 0.0) {
            throw new \InvalidArgumentException('SVR: epsilon must be >= 0.');
        }
        if (!array_key_exists($kernel, LibSVMBridge::KERNEL_MAP)) {
            throw new \InvalidArgumentException(
                "SVR: unknown kernel '{$kernel}'. "
                . "Valid: " . implode(', ', array_keys(LibSVMBridge::KERNEL_MAP))
            );
        }
        if (is_string($gamma) && !in_array($gamma, ['scale', 'auto'], true)) {
            throw new \InvalidArgumentException("SVR: gamma must be 'scale', 'auto', or a float.");
        }
    }

    // ── Destructor ────────────────────────────────────────────────────────

    public function __destruct()
    {
        if ($this->fitted) {
            LibSVMBridge::get()->ffi->svm_free_and_destroy_model(
                \FFI::addr($this->modelBox[0])
            );
            $this->fitted = false;
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Fit ε-SVR to training data.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  [n_samples] — continuous regression targets
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('SVR: y must be provided.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('SVR: X must be 2-D [n_samples, n_features].');
        }

        [$n, $d]              = $X->shape;
        $this->n_features_in_ = $d;

        $ffi = LibSVMBridge::get()->ffi;

        // ── Resolve gamma (same logic as SVC) ─────────────────────────
        if (is_string($this->gamma)) {
            if ($this->gamma === 'auto') {
                $gammaVal = 1.0 / $d;
            } else {
                // 'scale': γ = 1 / (d · Var(X))
                $blas  = BlasEngine::get()->ffi;
                $nd    = $n * $d;
                $sumSq = (float)$blas->cblas_sdot($nd, $X->buffer, 1, $X->buffer, 1) / $nd;
                $sum   = 0.0;
                for ($k = 0; $k < $nd; $k++) {
                    $sum += (float)$X->buffer[$k];
                }
                $meanX    = $sum / $nd;
                $varX     = $sumSq - $meanX * $meanX;
                $gammaVal = $varX > 1e-10 ? 1.0 / ($d * $varX) : 1.0 / $d;
            }
        } else {
            $gammaVal = (float)$this->gamma;
        }
        $this->gamma_ = $gammaVal;

        // ── Marshal Tensor $X → svm_node[][] ──────────────────────────
        //
        // Identical to SVC marshalling; see SVC.php for detailed comments.
        $nodeArrays = [];
        for ($i = 0; $i < $n; $i++) {
            $nodes = $ffi->new('svm_node[' . ($d + 1) . ']');
            for ($j = 0; $j < $d; $j++) {
                $nodes[$j]->index = $j + 1;
                $nodes[$j]->value = (float)$X->buffer[$i * $d + $j];
            }
            $nodes[$d]->index = -1;
            $nodes[$d]->value = 0.0;
            $nodeArrays[$i] = $nodes;
        }

        $xPtrs = $ffi->new("svm_node*[{$n}]");
        for ($i = 0; $i < $n; $i++) {
            $xPtrs[$i] = \FFI::addr($nodeArrays[$i][0]);
        }

        $yArr = $ffi->new("double[{$n}]");
        for ($i = 0; $i < $n; $i++) {
            $yArr[$i] = (float)$y->buffer[$i];
        }

        // ── Build svm_problem ─────────────────────────────────────────
        $prob    = $ffi->new('svm_problem');
        $prob->l = $n;
        $prob->y = \FFI::addr($yArr[0]);
        $prob->x = \FFI::addr($xPtrs[0]);

        // ── Build svm_parameter ───────────────────────────────────────
        //
        // SVR-specific:
        //   svm_type = EPSILON_SVR (3)  — ε-insensitive regression loss
        //   p = epsilon                 — half-width of the ε-insensitive tube
        //
        // The solver minimises: ½||w||² + C·Σ(ξ_i + ξ_i*)
        // Points within ε of the prediction incur zero loss.
        $param              = $ffi->new('svm_parameter');
        $param->svm_type    = LibSVMBridge::EPSILON_SVR;   // ← SVR-specific
        $param->kernel_type = LibSVMBridge::KERNEL_MAP[$this->kernel];
        $param->degree      = $this->degree;
        $param->gamma       = $gammaVal;
        $param->coef0       = $this->coef0;
        $param->cache_size  = $this->cache_size;
        $param->eps         = $this->tol;
        $param->C           = $this->C;
        $param->p           = $this->epsilon;   // ← ε-tube half-width
        $param->nr_weight   = 0;
        $param->shrinking   = $this->shrinking ? 1 : 0;
        $param->probability = 0;

        // ── Train ─────────────────────────────────────────────────────
        $this->modelBox    = $ffi->new('void*[1]');
        $this->modelBox[0] = $ffi->svm_train(\FFI::addr($prob), \FFI::addr($param));

        if ($this->modelBox[0] === null) {
            throw new \RuntimeException('SVR: svm_train() returned a null model pointer.');
        }

        $this->fitted = true;
        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict continuous target values for $X.
     *
     * svm_predict() in EPSILON_SVR mode returns a regression float, not a
     * class label.  We store it directly (no argmax, no rounding).
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples] — predicted regression values
     */
    public function predict(Tensor $X): Tensor
    {
        if (!$this->fitted) {
            throw new \RuntimeException('SVR is not fitted. Call fit() first.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('SVR::predict() requires a 2-D tensor.');
        }

        [$m, $d] = $X->shape;

        if ($d !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "SVR: expected {$this->n_features_in_} features, got {$d}."
            );
        }

        $ffi = LibSVMBridge::get()->ffi;
        $out = new Tensor([$m]);

        // Reuse a single svm_node buffer across all test samples.
        $nodes = $ffi->new('svm_node[' . ($d + 1) . ']');
        $nodes[$d]->index = -1;
        $nodes[$d]->value = 0.0;

        for ($i = 0; $i < $m; $i++) {
            for ($j = 0; $j < $d; $j++) {
                $nodes[$j]->index = $j + 1;
                $nodes[$j]->value = (float)$X->buffer[$i * $d + $j];
            }

            // SVR: svm_predict returns a regression value (double)
            $out->buffer[$i] = (float)$ffi->svm_predict(
                $this->modelBox[0],
                \FFI::addr($nodes[0])
            );
        }

        return $out;
    }
}
