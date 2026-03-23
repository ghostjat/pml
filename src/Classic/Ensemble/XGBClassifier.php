<?php

declare(strict_types=1);

namespace Pml\Classic\Ensemble;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  XGBClassifier — XGBoost gradient-boosted tree classifier
//
//  Wraps the XGBoost C API (via XGBoostBridge FFI) with a scikit-learn
//  compatible interface.
//
//  ── Gradient Boosting Overview ───────────────────────────────────────────
//
//  XGBoost builds an additive ensemble of CART trees:
//
//    F_T(x) = Σ_{t=1}^{T}  η · f_t(x)
//
//  where f_t is the t-th tree and η is the learning rate (eta).
//  Each tree minimises the second-order Taylor expansion of the loss
//  with an L2 leaf-weight regularisation term.
//
//  ── Objective Functions ──────────────────────────────────────────────────
//
//    'binary:logistic'  — binary cross-entropy; output = sigmoid probability
//    'multi:softmax'    — multi-class CE (argmax output, requires num_class)
//    'multi:softprob'   — multi-class CE (full probability matrix output)
//
//  When objective='auto' (the default), XGBClassifier selects:
//    binary  (n_classes=2) → 'binary:logistic'
//    multi   (n_classes>2) → 'multi:softprob'
//
//  ── Zero-Copy DMatrix Creation ───────────────────────────────────────────
//
//  XGBoost's XGDMatrixCreateFromMat accepts a raw const float* and reads
//  directly from the Tensor's FFI buffer — no PHP-heap copy, no temp array.
//  XGBoost immediately copies the data into its own compressed internal
//  representation, so our buffer is safe to release after the call.
//
//  ── Handle Lifecycle ─────────────────────────────────────────────────────
//
//  After fit():
//    $this->booster   — the trained BoosterHandle (void*)
//  After predict()/predict_proba():
//    temporary DMatrix handles are created, used, and freed inline.
//  After __destruct():
//    XGBoosterFree($this->booster)
//    PHP GC frees all void*[1] boxes.
//
//  ── Prediction Output Layout ─────────────────────────────────────────────
//
//  XGBoosterPredict writes into an XGBoost-internal float buffer:
//
//    binary:logistic  → float[n_samples]        — P(class=1) per sample
//    multi:softprob   → float[n_samples × K]    — P(class=k) per sample, row-major
//    multi:softmax    → float[n_samples]         — predicted class index per sample
//
//  We immediately copy (memcpy) from this buffer into a new Pml Tensor
//  because the buffer may be invalidated by the next predict call.
// ═══════════════════════════════════════════════════════════════════════════

final class XGBClassifier implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** @var int[]  Sorted distinct class labels from training set. */
    public readonly array $classes_;

    public readonly int $n_classes_;
    public readonly int $n_features_in_;

    /** Resolved XGBoost objective string (e.g. 'binary:logistic'). */
    public readonly string $objective_;

    // ── Internal XGBoost handles ─────────────────────────────────────────

    /**
     * void*[1] box holding the BoosterHandle.
     * The [1] array makes the pointer addressable for XGBoosterCreate's void**.
     */
    private \FFI\CData $boosterBox;
    private bool $fitted = false;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int    $n_estimators     Number of boosting rounds (trees).
     * @param int    $max_depth        Maximum tree depth.  Typical: 3–10.
     * @param float  $learning_rate    Shrinkage applied to each tree (η).
     *                                 Lower values need more trees but generalise better.
     * @param float  $subsample        Row subsampling ratio per tree (0,1].
     * @param float  $colsample_bytree Column subsampling ratio per tree (0,1].
     * @param float  $reg_lambda       L2 regularisation on leaf weights.
     * @param float  $reg_alpha        L1 regularisation on leaf weights.
     * @param float  $min_child_weight Minimum sum of instance weights per leaf.
     *                                 Higher values prevent overfitting to rare samples.
     * @param string $objective        XGBoost objective, or 'auto' to infer from
     *                                 the number of classes discovered at fit() time.
     * @param int    $n_jobs           Number of parallel threads (0 = all CPUs).
     * @param int    $random_state     RNG seed.
     */
    public function __construct(
        private readonly int    $n_estimators     = 100,
        private readonly int    $max_depth        = 6,
        private readonly float  $learning_rate    = 0.3,
        private readonly float  $subsample        = 1.0,
        private readonly float  $colsample_bytree = 1.0,
        private readonly float  $reg_lambda       = 1.0,
        private readonly float  $reg_alpha        = 0.0,
        private readonly float  $min_child_weight = 1.0,
        private readonly string $objective        = 'auto',
        private readonly int    $n_jobs           = 0,
        private readonly int    $random_state     = 0,
    ) {}

    // ── Destructor ────────────────────────────────────────────────────────

    public function __destruct()
    {
        if ($this->fitted) {
            // Free the booster handle — releases all tree structures
            XGBoostBridge::get()->ffi->XGBoosterFree($this->boosterBox[0]);
            $this->fitted = false;
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Train the XGBoost classifier.
     *
     * Workflow:
     *   1. Discover class labels.
     *   2. Resolve objective ('auto' → 'binary:logistic' or 'multi:softprob').
     *   3. XGDMatrixCreateFromMat: pass $X->buffer directly (zero-copy read).
     *   4. XGDMatrixSetFloatInfo: bind $y labels to the DMatrix.
     *   5. XGBoosterCreate: allocate the booster, bind to training DMatrix.
     *   6. XGBoosterSetParam: send hyperparameters as key=value strings.
     *   7. Loop XGBoosterUpdateOneIter for n_estimators rounds.
     *   8. Free training DMatrix (booster keeps its own reference).
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  [n_samples] — integer class labels
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('XGBClassifier: y must be provided.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('XGBClassifier: X must be 2-D [n_samples, n_features].');
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
        $K                = count($classArr);
        $this->n_classes_ = $K;

        // ── Resolve objective ──────────────────────────────────────────
        //
        // 'auto' selects the standard XGBoost objective for the task:
        //   K = 2  → binary:logistic  (outputs sigmoid probability of class 1)
        //   K > 2  → multi:softprob   (outputs K probabilities per sample)
        //
        // For multi-class we use softprob (not softmax) so that predict_proba()
        // can return full probability distributions.  predict() applies argmax.
        if ($this->objective === 'auto') {
            $obj = ($K === 2) ? 'binary:logistic' : 'multi:softprob';
        } else {
            $obj = $this->objective;
        }
        $this->objective_ = $obj;

        $bridge = XGBoostBridge::get();
        $ffi    = $bridge->ffi;

        // ── Step 3: Create DMatrix from Tensor buffer (zero-copy read) ─
        //
        // XGDMatrixCreateFromMat reads from the float* directly:
        //   data    = $X->buffer  — flat float[n*d] row-major matrix
        //   nrow    = n           — number of samples
        //   ncol    = d           — number of features
        //   missing = NAN         — no missing value sentinel
        //   out     → dmatBox[0]  — receives the DMatrixHandle
        //
        // XGBoost immediately copies into its own internal format, so
        // $X->buffer is safe to release after this call.
        $dmatBox = $ffi->new('void*[1]');
        $bridge->check(
            $ffi->XGDMatrixCreateFromMat($X->buffer, $n, $d, NAN, \FFI::addr($dmatBox[0])),
            'XGDMatrixCreateFromMat'
        );

        // ── Step 4: Bind labels ────────────────────────────────────────
        //
        // XGDMatrixSetFloatInfo("label", float_array, n):
        //   Attaches training labels.  XGBoost needs a float* even though labels
        //   are integers — our Tensor buffer IS float32, so we pass it directly.
        $bridge->check(
            $ffi->XGDMatrixSetFloatInfo($dmatBox[0], 'label', $y->buffer, $n),
            'XGDMatrixSetFloatInfo(label)'
        );

        // ── Step 5: Create Booster ─────────────────────────────────────
        //
        // XGBoosterCreate takes const void** dmats (array of DMatrixHandle),
        // so we pass the address of our dmat handle (a void* in a void*[1] box).
        //   \FFI::addr($dmatBox[0]) → void**  (pointer to the single DMatrix)
        $this->boosterBox = $ffi->new('void*[1]');
        $bridge->check(
            $ffi->XGBoosterCreate(
                \FFI::addr($dmatBox[0]),   // const void**  (array of 1 DMatrix)
                1,                          // bst_ulong len
                \FFI::addr($this->boosterBox[0])  // void** out → BoosterHandle
            ),
            'XGBoosterCreate'
        );

        $bst = $this->boosterBox[0];

        // ── Step 6: Set hyperparameters ───────────────────────────────
        //
        // All XGBoost parameters are passed as string pairs.
        // num_class is mandatory for multi:softmax / multi:softprob.
        $params = [
            'max_depth'          => (string)$this->max_depth,
            'eta'                => (string)$this->learning_rate,
            'subsample'          => (string)$this->subsample,
            'colsample_bytree'   => (string)$this->colsample_bytree,
            'lambda'             => (string)$this->reg_lambda,
            'alpha'              => (string)$this->reg_alpha,
            'min_child_weight'   => (string)$this->min_child_weight,
            'objective'          => $obj,
            'seed'               => (string)$this->random_state,
            'nthread'            => (string)$this->n_jobs,
            'verbosity'          => '0',    // suppress XGBoost console output
        ];

        // num_class required when the objective operates on multiple classes
        if (str_starts_with($obj, 'multi:')) {
            $params['num_class'] = (string)$K;
        }

        foreach ($params as $name => $value) {
            $bridge->check(
                $ffi->XGBoosterSetParam($bst, $name, $value),
                "XGBoosterSetParam({$name})"
            );
        }

        // ── Step 7: Boosting loop ─────────────────────────────────────
        //
        // Each call to XGBoosterUpdateOneIter adds one tree (or one set of
        // trees for multi-class) to the ensemble.  The iter parameter is
        // passed to XGBoost for internal scheduling (learning rate decay, etc.)
        for ($iter = 0; $iter < $this->n_estimators; $iter++) {
            $bridge->check(
                $ffi->XGBoosterUpdateOneIter($bst, $iter, $dmatBox[0]),
                "XGBoosterUpdateOneIter(iter={$iter})"
            );
        }

        // ── Step 8: Free training DMatrix ─────────────────────────────
        //
        // The booster has finished training and keeps its own copy of the
        // learned trees.  We no longer need the DMatrix.
        $ffi->XGDMatrixFree($dmatBox[0]);

        $this->fitted = true;
        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict class labels for $X.
     *
     * For binary classification: thresholds predict_proba at 0.5.
     * For multi-class: takes argmax over the K class probabilities.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples] — integer class labels (as float32)
     */
    public function predict(Tensor $X): Tensor
    {
        $proba = $this->predict_proba($X);
        $m     = $X->shape[0];
        $K     = $this->n_classes_;
        $out   = new Tensor([$m]);

        if ($K === 2) {
            // binary: proba[i] = P(class=1); threshold at 0.5
            for ($i = 0; $i < $m; $i++) {
                $out->buffer[$i] = (float)$this->classes_[
                    (float)$proba->buffer[$i] >= 0.5 ? 1 : 0
                ];
            }
        } else {
            // multi-class: argmax over K probabilities per sample
            for ($i = 0; $i < $m; $i++) {
                $bestK = 0;
                $bestP = (float)$proba->buffer[$i * $K];
                for ($k = 1; $k < $K; $k++) {
                    $p = (float)$proba->buffer[$i * $K + $k];
                    if ($p > $bestP) { $bestP = $p; $bestK = $k; }
                }
                $out->buffer[$i] = (float)$this->classes_[$bestK];
            }
        }

        return $out;
    }

    /**
     * Predict class probabilities for $X.
     *
     * Returns a Tensor of shape:
     *   [n_samples]         for binary classification (P(class=1))
     *   [n_samples, K]      for K-class classification (softmax probs)
     *
     * Workflow:
     *   1. Create a temporary DMatrix from $X (zero-copy).
     *   2. Call XGBoosterPredict → XGBoost writes a const float* into out_result.
     *   3. \FFI::memcpy the output into a new Pml Tensor immediately.
     *   4. Free the temporary DMatrix.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples] or [n_samples, n_classes]
     */
    public function predict_proba(Tensor $X): Tensor
    {
        if (!$this->fitted) {
            throw new \RuntimeException('XGBClassifier is not fitted. Call fit() first.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('XGBClassifier::predict_proba() requires a 2-D tensor.');
        }

        [$m, $d] = $X->shape;

        if ($d !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "XGBClassifier: expected {$this->n_features_in_} features, got {$d}."
            );
        }

        $bridge = XGBoostBridge::get();
        $ffi    = $bridge->ffi;

        // ── Step 1: Create test DMatrix ────────────────────────────────
        //
        // Same zero-copy approach as fit(): pass $X->buffer directly.
        // This DMatrix is temporary — freed after predict.
        $dmatBox = $ffi->new('void*[1]');
        $bridge->check(
            $ffi->XGDMatrixCreateFromMat($X->buffer, $m, $d, NAN, \FFI::addr($dmatBox[0])),
            'XGDMatrixCreateFromMat (predict)'
        );

        // ── Step 2: Predict ───────────────────────────────────────────
        //
        // XGBoosterPredict signature:
        //   handle      → $this->boosterBox[0]          (the trained booster)
        //   dmat        → $dmatBox[0]                   (test DMatrix)
        //   option_mask → 0                             (normal prediction)
        //   ntree_limit → 0                             (use all trees)
        //   training    → 0                             (inference mode)
        //   out_len     → &$outLen                      (number of output floats)
        //   out_result  → &$outPtrs[0]                  (pointer to XGBoost buffer)
        //
        // out_result is a float** — XGBoost fills in the pointer to its internal
        // buffer.  We use a float*[1] array so the element is addressable.
        $outLen     = $ffi->new('bst_ulong');
        $outPtrBox  = $ffi->new('float*[1]');  // float*[1]: one slot for a float*

        $bridge->check(
            $ffi->XGBoosterPredict(
                $this->boosterBox[0],
                $dmatBox[0],
                0,     // option_mask: 0 = output probabilities (not margins)
                0,     // ntree_limit: 0 = all trees
                0,     // training:    0 = inference mode
                \FFI::addr($outLen),        // bst_ulong* out_len
                \FFI::addr($outPtrBox[0])   // const float** out_result → float**
            ),
            'XGBoosterPredict'
        );

        // ── Step 3: Copy output into a new Pml Tensor ─────────────────
        //
        // $outPtrBox[0] is now a float* pointing to XGBoost's internal buffer.
        // We MUST copy immediately — the buffer is invalidated on the next call.
        //
        // Output shape:
        //   binary:logistic → out_len = m,    shape [m]
        //   multi:softprob  → out_len = m * K, shape [m, K]  (row-major)
        //   multi:softmax   → out_len = m,    shape [m]
        $totalFloats = (int)(float)$outLen->cdata;  // bst_ulong → PHP int

        if ($this->n_classes_ === 2) {
            $out = new Tensor([$m]);
        } else {
            // For multi:softprob, shape is [m, K]; for multi:softmax it's [m]
            $K = $this->n_classes_;
            $out = ($totalFloats === $m * $K)
                ? new Tensor([$m, $K])
                : new Tensor([$m]);
        }

        // Zero-overhead copy: $outPtrBox[0] is the raw float* source
        \FFI::memcpy($out->buffer, $outPtrBox[0], $totalFloats * 4);

        // ── Step 4: Free temporary test DMatrix ───────────────────────
        $ffi->XGDMatrixFree($dmatBox[0]);

        return $out;
    }
}
