<?php

declare(strict_types=1);

namespace Pml\Classic\Ensemble;

use Pml\Tensor;
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  XGBRegressor — XGBoost gradient-boosted tree regressor
//
//  Wraps the XGBoost C API (via XGBoostBridge FFI) with a scikit-learn
//  compatible interface for continuous-output regression tasks.
//
//  ── Objective Function ───────────────────────────────────────────────────
//
//  Fixed to 'reg:squarederror': minimises the mean squared error between
//  the ensemble output F_T(x) and the continuous target y.
//
//    L(y, ŷ) = (y − ŷ)²
//
//  Gradient:  g_i = ŷ_i − y_i
//  Hessian:   h_i = 1.0    (constant for MSE)
//
//  This is the XGBoost default for regression and is numerically stable
//  for most continuous targets.  For targets with heavy tails or large
//  outliers, prefer 'reg:pseudohubererror' (not implemented here).
//
//  ── Zero-Copy DMatrix Creation ───────────────────────────────────────────
//
//  Identical to XGBClassifier: XGDMatrixCreateFromMat reads directly from
//  the Tensor's FFI buffer.  XGBoost copies the data into its own internal
//  format immediately, so the buffer is safe to release after the call.
//
//  ── Prediction Output Layout ─────────────────────────────────────────────
//
//  XGBoosterPredict writes float[n_samples] — one raw regression output per
//  sample.  No sigmoid, argmax, or softmax is applied.  We \FFI::memcpy the
//  output directly into a new [n_samples] Pml Tensor.
//
//  ── Handle Lifecycle ─────────────────────────────────────────────────────
//
//  After fit():
//    $this->booster   — the trained BoosterHandle (void*)
//  Temporary DMatrix handles are created per predict() call and freed inline.
//  __destruct() calls XGBoosterFree to release all tree structures.
// ═══════════════════════════════════════════════════════════════════════════

final class XGBRegressor implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    public readonly int   $n_features_in_;

    /** Always 'reg:squarederror' — exposed for introspection / serialisation. */
    public readonly string $objective_;

    // ── Internal XGBoost handle ───────────────────────────────────────────

    /**
     * void*[1] box holding the BoosterHandle.
     * Array-of-one makes the pointer addressable for XGBoosterCreate's void**.
     */
    private \FFI\CData $boosterBox;
    private bool $fitted = false;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int        $n_estimators     Number of boosting rounds (trees).
     * @param int        $max_depth        Maximum tree depth.  Shallower trees
     *                                     (3–6) generalise better for regression.
     * @param float      $learning_rate    Shrinkage applied to each tree (η).
     *                                     Lower values reduce overfitting but need
     *                                     more trees to reach the same loss level.
     * @param float      $subsample        Row subsampling fraction per tree (0, 1].
     *                                     < 1.0 adds stochasticity (Friedman-style).
     * @param float      $colsample_bytree Column subsampling fraction per tree (0, 1].
     * @param float      $reg_lambda       L2 leaf-weight regularisation (λ).
     * @param float      $reg_alpha        L1 leaf-weight regularisation (α).
     * @param float      $min_child_weight Minimum sum of instance weights per leaf.
     *                                     Higher values prevent overfitting to noisy
     *                                     samples.
     * @param int        $n_jobs           Parallel threads (0 = all CPUs).
     * @param int|null   $random_state     RNG seed (null → XGBoost default 0).
     */
    public function __construct(
        private readonly int   $n_estimators     = 100,
        private readonly int   $max_depth        = 3,
        private readonly float $learning_rate    = 0.1,
        private readonly float $subsample        = 1.0,
        private readonly float $colsample_bytree = 1.0,
        private readonly float $reg_lambda       = 1.0,
        private readonly float $reg_alpha        = 0.0,
        private readonly float $min_child_weight = 1.0,
        private readonly int   $n_jobs           = 0,
        private readonly ?int  $random_state     = null,
    ) {}

    // ── Destructor ────────────────────────────────────────────────────────

    public function __destruct()
    {
        if ($this->fitted) {
            XGBoostBridge::get()->ffi->XGBoosterFree($this->boosterBox[0]);
            $this->fitted = false;
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Train the XGBoost regressor.
     *
     * Workflow:
     *   1. Create DMatrix from $X->buffer (zero-copy read via FFI).
     *   2. Bind $y labels to the DMatrix via XGDMatrixSetFloatInfo.
     *   3. Allocate a Booster and bind it to the training DMatrix.
     *   4. Set hyperparameters — objective is always 'reg:squarederror'.
     *   5. Run n_estimators boosting rounds via XGBoosterUpdateOneIter.
     *   6. Free the training DMatrix (booster retains its own tree data).
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Continuous target values [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('XGBRegressor: $y must be provided.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('XGBRegressor: X must be 2-D [n_samples, n_features].');
        }

        [$n, $d]              = $X->shape;
        $this->n_features_in_ = $d;
        $this->objective_     = 'reg:squarederror';

        $bridge = XGBoostBridge::get();
        $ffi    = $bridge->ffi;

        // ── Step 1: Create DMatrix from Tensor buffer (zero-copy read) ─────
        //
        // XGDMatrixCreateFromMat reads from the flat float* directly.
        // nrow = n samples, ncol = d features, missing = NAN (no sentinel).
        $dmatBox = $ffi->new('void*[1]');
        $bridge->check(
            $ffi->XGDMatrixCreateFromMat($X->buffer, $n, $d, NAN, \FFI::addr($dmatBox[0])),
            'XGDMatrixCreateFromMat'
        );

        // ── Step 2: Bind continuous target labels ─────────────────────────
        //
        // The target Tensor is already float32 — pass its buffer directly.
        $bridge->check(
            $ffi->XGDMatrixSetFloatInfo($dmatBox[0], 'label', $y->buffer, $n),
            'XGDMatrixSetFloatInfo(label)'
        );

        // ── Step 3: Create Booster bound to the training DMatrix ──────────
        $this->boosterBox = $ffi->new('void*[1]');
        $bridge->check(
            $ffi->XGBoosterCreate(
                \FFI::addr($dmatBox[0]),
                1,
                \FFI::addr($this->boosterBox[0])
            ),
            'XGBoosterCreate'
        );

        $bst = $this->boosterBox[0];

        // ── Step 4: Set hyperparameters ───────────────────────────────────
        $params = [
            'max_depth'          => (string) $this->max_depth,
            'eta'                => (string) $this->learning_rate,
            'subsample'          => (string) $this->subsample,
            'colsample_bytree'   => (string) $this->colsample_bytree,
            'lambda'             => (string) $this->reg_lambda,
            'alpha'              => (string) $this->reg_alpha,
            'min_child_weight'   => (string) $this->min_child_weight,
            'objective'          => 'reg:squarederror',
            'seed'               => (string) ($this->random_state ?? 0),
            'nthread'            => (string) $this->n_jobs,
            'verbosity'          => '0',
        ];

        foreach ($params as $name => $value) {
            $bridge->check(
                $ffi->XGBoosterSetParam($bst, $name, $value),
                "XGBoosterSetParam({$name})"
            );
        }

        // ── Step 5: Boosting loop ─────────────────────────────────────────
        for ($iter = 0; $iter < $this->n_estimators; $iter++) {
            $bridge->check(
                $ffi->XGBoosterUpdateOneIter($bst, $iter, $dmatBox[0]),
                "XGBoosterUpdateOneIter(iter={$iter})"
            );
        }

        // ── Step 6: Free training DMatrix ─────────────────────────────────
        $ffi->XGDMatrixFree($dmatBox[0]);

        $this->fitted = true;
        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict continuous target values for $X.
     *
     * Workflow:
     *   1. Create a temporary DMatrix from $X (zero-copy).
     *   2. Call XGBoosterPredict → XGBoost writes float[n_samples] directly.
     *   3. \FFI::memcpy the output into a new [n_samples] Pml Tensor.
     *   4. Free the temporary DMatrix.
     *
     * @param Tensor $X  Feature matrix [n_samples, n_features]
     * @return Tensor    Predicted continuous values [n_samples]
     */
    public function predict(Tensor $X): Tensor
    {
        if (!$this->fitted) {
            throw new \RuntimeException('XGBRegressor is not fitted. Call fit() first.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('XGBRegressor::predict() requires a 2-D tensor.');
        }

        [$m, $d] = $X->shape;

        if ($d !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "XGBRegressor: expected {$this->n_features_in_} features, got {$d}."
            );
        }

        $bridge = XGBoostBridge::get();
        $ffi    = $bridge->ffi;

        // ── Step 1: Temporary test DMatrix ────────────────────────────────
        $dmatBox = $ffi->new('void*[1]');
        $bridge->check(
            $ffi->XGDMatrixCreateFromMat($X->buffer, $m, $d, NAN, \FFI::addr($dmatBox[0])),
            'XGDMatrixCreateFromMat (predict)'
        );

        // ── Step 2: Predict ───────────────────────────────────────────────
        //
        // reg:squarederror outputs exactly m floats — one raw prediction per sample.
        // option_mask=0, ntree_limit=0 (all trees), training=0 (inference mode).
        $outLen    = $ffi->new('bst_ulong');
        $outPtrBox = $ffi->new('float*[1]');

        $bridge->check(
            $ffi->XGBoosterPredict(
                $this->boosterBox[0],
                $dmatBox[0],
                0,     // option_mask: 0 = raw output scores
                0,     // ntree_limit: 0 = use all trees
                0,     // training:    0 = inference mode
                \FFI::addr($outLen),
                \FFI::addr($outPtrBox[0])
            ),
            'XGBoosterPredict'
        );

        // ── Step 3: Copy output into a new Pml Tensor ─────────────────────
        //
        // $outPtrBox[0] points to XGBoost's internal buffer — must copy
        // immediately before the next API call invalidates it.
        $totalFloats = (int) (float) $outLen->cdata;
        $out         = new Tensor([$m]);
        \FFI::memcpy($out->buffer, $outPtrBox[0], $totalFloats * 4);

        // ── Step 4: Free temporary DMatrix ────────────────────────────────
        $ffi->XGDMatrixFree($dmatBox[0]);

        return $out;
    }

    /**
     * R² score on test data.
     * Mirrors sklearn's RegressorMixin.score().
     */
    public function score(Tensor $X, Tensor $y): float
    {
        $pred  = $this->predict($X);
        $n     = $y->size;

        $yMean = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $yMean += (float) $y->buffer[$i];
        }
        $yMean /= $n;

        $ssTot = 0.0;
        $ssRes = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $ssTot += ((float) $y->buffer[$i] - $yMean) ** 2;
            $ssRes += ((float) $y->buffer[$i] - (float) $pred->buffer[$i]) ** 2;
        }

        return ($ssTot === 0.0) ? 1.0 : 1.0 - $ssRes / $ssTot;
    }
}
