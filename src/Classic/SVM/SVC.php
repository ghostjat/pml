<?php

declare(strict_types=1);

namespace Pml\Classic\SVM;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  SVC — sklearn.svm.SVC
//
//  C-Support Vector Classifier backed by libsvm (via LibSVMBridge FFI).
//
//  ── Algorithm Overview ───────────────────────────────────────────────────
//
//  Finds the maximum-margin hyperplane separating classes by solving the
//  dual QP:
//
//    min  ½ α^T Q α − e^T α
//    s.t. y^T α = 0,   0 ≤ α_i ≤ C
//
//  where Q[i,j] = y_i y_j K(x_i, x_j)  and  K is the chosen kernel.
//
//  Multi-class is handled by libsvm's built-in one-vs-one (OvO) strategy:
//  k(k−1)/2 binary classifiers are trained, predictions are resolved via
//  voting.
//
//  ── Kernel Choices ───────────────────────────────────────────────────────
//
//    'linear'  → K(u,v) = u^T v
//    'poly'    → K(u,v) = (γ·u^T v + coef0)^degree
//    'rbf'     → K(u,v) = exp(−γ·||u−v||²)   ← default
//    'sigmoid' → K(u,v) = tanh(γ·u^T v + coef0)
//
//  ── Gamma Computation ────────────────────────────────────────────────────
//
//    gamma='scale' (default):
//      γ = 1 / (n_features · Var(X_train))
//      where Var(X_train) = E[X²] − E[X]²  (global variance over all elements).
//      This normalises for the scale of the training data — sklearn's default
//      since v0.22.
//
//    gamma='auto':
//      γ = 1 / n_features  (sklearn pre-0.22 default)
//
//    gamma=float:
//      Use the supplied value directly.
//
//  ── Data Marshalling (Tensor → svm_node[]) ───────────────────────────────
//
//  libsvm expects each sample as a NULL-terminated array of (index, value)
//  pairs.  For a dense [n, d] Tensor:
//
//    For sample i:
//      nodes[j].index = j + 1          ← 1-based feature index
//      nodes[j].value = X[i, j]        ← widened float→double
//      nodes[d].index = -1             ← sentinel (libsvm convention)
//      nodes[d].value = 0.0
//
//  These svm_node arrays are PHP-GC-managed temporaries; libsvm copies
//  all training data into the model during svm_train(), so they can be
//  freed immediately after fit() returns.
//
//  ── Memory Safety ────────────────────────────────────────────────────────
//
//  The svm_model* returned by svm_train() is heap-allocated by libsvm.
//  It is stored in a void*[1] "box" ($this->modelBox) so that
//  \FFI::addr($this->modelBox[0]) yields the void** that
//  svm_free_and_destroy_model() requires.  __destruct() ensures the model
//  is freed exactly once when the PHP object is garbage-collected.
// ═══════════════════════════════════════════════════════════════════════════

final class SVC implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** @var int[]  Distinct class labels discovered during fit(), sorted. */
    public readonly array $classes_;

    public readonly int $n_classes_;
    public readonly int $n_features_in_;

    /** Computed gamma value (resolved from 'scale'/'auto'/float). */
    public readonly float $gamma_;

    // ── Internal model handle ─────────────────────────────────────────────

    /**
     * void*[1] box holding the svm_model* returned by svm_train().
     *
     * Using a one-element array makes the pointer addressable via
     * \FFI::addr($this->modelBox[0]) → void** for svm_free_and_destroy_model.
     */
    private \FFI\CData $modelBox;
    private bool $fitted = false;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param float       $C          Regularisation strength.  Smaller C → wider
     *                                margin with more misclassification allowed.
     * @param string      $kernel     'rbf' | 'linear' | 'poly' | 'sigmoid'
     * @param int|float|string $gamma RBF/poly/sigmoid coefficient.
     *                                'scale' = 1/(n_features·Var(X))
     *                                'auto'  = 1/n_features
     *                                float   = use directly
     * @param int         $degree     Degree for the polynomial kernel.
     * @param float       $coef0      Intercept for poly/sigmoid kernels.
     * @param float       $tol        Stopping tolerance for the SMO solver.
     * @param float       $cache_size Kernel cache in MB.
     * @param int         $max_iter   Maximum solver iterations (−1 = no limit).
     * @param bool        $shrinking  Whether to use shrinking heuristics.
     */
    public function __construct(
        private readonly float            $C          = 1.0,
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
            throw new \InvalidArgumentException('SVC: C must be > 0.');
        }
        if (!array_key_exists($kernel, LibSVMBridge::KERNEL_MAP)) {
            throw new \InvalidArgumentException(
                "SVC: unknown kernel '{$kernel}'. "
                . "Valid: " . implode(', ', array_keys(LibSVMBridge::KERNEL_MAP))
            );
        }
        if (is_string($gamma) && !in_array($gamma, ['scale', 'auto'], true)) {
            throw new \InvalidArgumentException("SVC: gamma must be 'scale', 'auto', or a float.");
        }
    }

    // ── Destructor: free the C-heap model ────────────────────────────────

    public function __destruct()
    {
        if ($this->fitted) {
            // svm_free_and_destroy_model(&model_ptr):
            //   – frees all libsvm internal allocations
            //   – sets *model_ptr = NULL  (writes into our box)
            LibSVMBridge::get()->ffi->svm_free_and_destroy_model(
                \FFI::addr($this->modelBox[0])
            );
            $this->fitted = false;
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Train the SVM classifier.
     *
     * Steps:
     *   1. Discover and encode class labels.
     *   2. Resolve gamma (scale/auto/float).
     *   3. Marshal Tensor $X into libsvm's svm_node[][] sparse format.
     *   4. Build svm_problem and svm_parameter structs on the C stack.
     *   5. Call svm_train() → store the returned model pointer.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  [n_samples] — integer class labels
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('SVC: y must be provided.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('SVC: X must be 2-D [n_samples, n_features].');
        }

        [$n, $d]              = $X->shape;
        $this->n_features_in_ = $d;

        // ── Discover class labels ──────────────────────────────────────
        $labelSet = [];
        for ($i = 0; $i < $n; $i++) {
            $labelSet[(int)(float)$y->buffer[$i]] = true;
        }
        $classArr = array_keys($labelSet);
        sort($classArr);
        $this->classes_   = $classArr;
        $this->n_classes_ = count($classArr);

        $ffi = LibSVMBridge::get()->ffi;

        // ── Resolve gamma ──────────────────────────────────────────────
        //
        // gamma='scale': γ = 1 / (d · Var(X))
        //   Var(X) = E[X²] - E[X]²  (global variance over all n·d elements)
        //   We use BLAS sdot for the E[X²] sum, PHP loop for E[X] sum
        //   (cblas_sasum gives Σ|x|, not Σx).
        //
        // gamma='auto':  γ = 1 / d
        if (is_string($this->gamma)) {
            if ($this->gamma === 'auto') {
                $gammaVal = 1.0 / $d;
            } else {
                // 'scale'
                $blas     = BlasEngine::get()->ffi;
                $nd       = $n * $d;
                // E[X²] via sdot(n*d, X, 1, X, 1) / (n*d)
                $sumSq    = (float)$blas->cblas_sdot($nd, $X->buffer, 1, $X->buffer, 1) / $nd;
                // E[X] via PHP loop (no direct BLAS sum with sign)
                $sum      = 0.0;
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

        // ── Marshal X into svm_node arrays ────────────────────────────
        //
        // Each sample i becomes an svm_node[d+1] array:
        //   nodes[j] = { index=j+1, value=(double)X[i,j] }  j∈[0,d)
        //   nodes[d] = { index=-1,  value=0.0 }              ← sentinel
        //
        // These are PHP-GC temporaries: libsvm copies all data during
        // svm_train(), so we don't need to keep them alive after the call.
        $nodeArrays = [];     // float32 → double widening happens here
        for ($i = 0; $i < $n; $i++) {
            $nodes = $ffi->new('svm_node[' . ($d + 1) . ']');
            for ($j = 0; $j < $d; $j++) {
                $nodes[$j]->index = $j + 1;
                $nodes[$j]->value = (float)$X->buffer[$i * $d + $j];
            }
            $nodes[$d]->index = -1;  // sentinel: marks end of sparse vector
            $nodes[$d]->value = 0.0;
            $nodeArrays[$i] = $nodes;
        }

        // ── Build svm_node** (pointer array) ─────────────────────────
        //
        // svm_problem.x is svm_node**: an array of N pointers, one per sample.
        // $xPtrs[$i] = address of the first node of sample i.
        $xPtrs = $ffi->new("svm_node*[{$n}]");
        for ($i = 0; $i < $n; $i++) {
            $xPtrs[$i] = \FFI::addr($nodeArrays[$i][0]);
        }

        // ── Build double[] label array ────────────────────────────────
        //
        // libsvm uses double for labels (supports regression and multi-class).
        // We widen from our float32 Tensor.
        $yArr = $ffi->new("double[{$n}]");
        for ($i = 0; $i < $n; $i++) {
            $yArr[$i] = (float)$y->buffer[$i];
        }

        // ── Build svm_problem ─────────────────────────────────────────
        $prob    = $ffi->new('svm_problem');
        $prob->l = $n;
        $prob->y = \FFI::addr($yArr[0]);     // double* → $yArr[0] is a double lvalue
        $prob->x = \FFI::addr($xPtrs[0]);   // svm_node** → pointer to pointer array

        // ── Build svm_parameter ───────────────────────────────────────
        $param              = $ffi->new('svm_parameter');
        $param->svm_type    = LibSVMBridge::C_SVC;
        $param->kernel_type = LibSVMBridge::KERNEL_MAP[$this->kernel];
        $param->degree      = $this->degree;
        $param->gamma       = $gammaVal;
        $param->coef0       = $this->coef0;
        $param->cache_size  = $this->cache_size;
        $param->eps         = $this->tol;
        $param->C           = $this->C;
        $param->nr_weight   = 0;      // no per-class weight overrides
        $param->shrinking   = $this->shrinking ? 1 : 0;
        $param->probability = 0;
        // weight_label and weight remain zero-initialised (null pointers)
        // because nr_weight = 0; libsvm will not dereference them.

        // ── Train: svm_train() copies all data into the model ─────────
        $this->modelBox    = $ffi->new('void*[1]');
        $this->modelBox[0] = $ffi->svm_train(\FFI::addr($prob), \FFI::addr($param));

        if ($this->modelBox[0] === null) {
            throw new \RuntimeException('SVC: svm_train() returned a null model pointer.');
        }

        $this->fitted = true;

        // $prob, $xPtrs, $yArr, $nodeArrays all go out of scope here and are
        // freed by PHP's GC — safe because svm_train copied everything.
        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Classify samples in $X.
     *
     * For each test sample, builds a temporary svm_node[d+1] array and calls
     * svm_predict().  libsvm's OvO voting resolves to the winning class label.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples] — float32 encoding of integer class labels
     */
    public function predict(Tensor $X): Tensor
    {
        if (!$this->fitted) {
            throw new \RuntimeException('SVC is not fitted. Call fit() first.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('SVC::predict() requires a 2-D tensor.');
        }

        [$m, $d] = $X->shape;

        if ($d !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "SVC: expected {$this->n_features_in_} features, got {$d}."
            );
        }

        $ffi = LibSVMBridge::get()->ffi;
        $out = new Tensor([$m]);

        // ── Per-sample prediction ──────────────────────────────────────
        //
        // Reuse a single svm_node[d+1] buffer across all samples to avoid
        // N separate FFI allocations.  We overwrite it each iteration.
        $nodes = $ffi->new('svm_node[' . ($d + 1) . ']');
        $nodes[$d]->index = -1;   // sentinel stays fixed
        $nodes[$d]->value = 0.0;

        for ($i = 0; $i < $m; $i++) {
            // Fill feature nodes for sample i
            for ($j = 0; $j < $d; $j++) {
                $nodes[$j]->index = $j + 1;
                $nodes[$j]->value = (float)$X->buffer[$i * $d + $j];
            }

            // svm_predict returns the predicted class label as a double
            $label           = $ffi->svm_predict($this->modelBox[0], \FFI::addr($nodes[0]));
            $out->buffer[$i] = (float)$label;
        }

        return $out;
    }
}
