<?php

declare(strict_types=1);

namespace Pml\Classic\ModelSelection;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\Metrics\Metrics;

// ═══════════════════════════════════════════════════════════════════════════
//  Validation — cross_val_score and related utilities
//
//  Mirrors sklearn.model_selection.cross_val_score.
//
//  ── cross_val_score ──────────────────────────────────────────────────────
//
//  For each KFold split:
//    1. Clone the estimator (fresh, unfitted copy with same hyperparameters).
//    2. Slice X_train, X_test, y_train, y_test from the fold indices using
//       cblas_scopy — identical gather pattern to DataSplit::train_test_split.
//    3. Fit the clone on (X_train, y_train).
//    4. Predict on X_test.
//    5. Evaluate the scoring function → append to scores[].
//
//  Returns float[] — one score per fold.
//
//  ── Estimator Cloning ────────────────────────────────────────────────────
//
//  PHP lacks Python's sklearn.base.clone() which rebuilds an estimator from
//  its get_params() dict.  We approximate this with Reflection:
//
//    1. Enumerate the constructor parameters of the estimator's class.
//    2. Read each parameter's value from the (unfitted) estimator using
//       ReflectionProperty::getValue() — constructor-promoted properties have
//       the same name as the constructor parameter.
//    3. Create a fresh instance via ReflectionClass::newInstance(...$args).
//
//  This gives a new, unfitted instance with identical hyperparameters.
//  IMPORTANT: pass an UNFITTED estimator to cross_val_score.  If constructor
//  parameters are not yet stored as properties (e.g. a manually-built object),
//  the parameter's declared default value is used as a fallback.
//
//  ── Supported Scoring Strings ────────────────────────────────────────────
//
//  'accuracy'               → Metrics::accuracy_score      (classification)
//  'r2'                     → Metrics::r2_score             (regression)
//  'neg_mean_squared_error' → −Metrics::mean_squared_error  (regression, negated)
//  callable($y_true, $y_pred): float → user-supplied scorer
// ═══════════════════════════════════════════════════════════════════════════

final class Validation
{
    /**
     * Evaluate an estimator by cross-validation and return per-fold scores.
     *
     * @param object            $estimator  An UNFITTED estimator (Estimator + Predictor).
     * @param Tensor            $X          Feature matrix [n_samples, n_features]
     * @param Tensor            $y          Target vector  [n_samples]
     * @param int|KFold         $cv         Number of folds (int) or a pre-configured KFold.
     * @param string|callable   $scoring    Metric name or callable($y_true, $y_pred): float.
     *                                      Built-in: 'accuracy', 'r2', 'neg_mean_squared_error'.
     * @return float[]                      Array of per-fold scores, length = n_splits.
     */
    public static function cross_val_score(
        object           $estimator,
        Tensor           $X,
        Tensor           $y,
        int|KFold        $cv      = 5,
        string|callable  $scoring = 'accuracy',
    ): array {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException(
                'cross_val_score: X must be 2-D [n_samples, n_features].'
            );
        }

        // Resolve cross-validator
        $kfold = is_int($cv) ? new KFold($cv) : $cv;

        $scores = [];

        foreach ($kfold->split($X) as [$trainIdx, $testIdx]) {
            // ── Step 1: fresh, unfitted clone of the estimator ─────────────
            $est = self::cloneEstimator($estimator);

            // ── Step 2: gather fold tensors via cblas_scopy ────────────────
            [$Xtrain, $Xtest, $ytrain, $ytest] = self::gatherFold(
                $X, $y, $trainIdx, $testIdx
            );

            // ── Step 3: fit on training fold ───────────────────────────────
            $est->fit($Xtrain, $ytrain);

            // ── Step 4: predict on test fold ───────────────────────────────
            $yPred = $est->predict($Xtest);

            // ── Step 5: score ──────────────────────────────────────────────
            $scores[] = self::evaluate($scoring, $ytest, $yPred);
        }

        return $scores;
    }

    // ── Estimator cloning ─────────────────────────────────────────────────

    /**
     * Create a fresh unfitted copy of $estimator with identical hyperparameters.
     *
     * Algorithm (mirrors sklearn's clone()):
     *  1. Inspect the constructor's parameter list via Reflection.
     *  2. For each parameter name, read the corresponding property value from
     *     $estimator (constructor-promoted readonly props share the param name).
     *  3. Instantiate a new object with those args.
     *
     * If a property does not exist on the class (edge case), the parameter's
     * default value is used.  Uninitialized properties (common on fitted
     * attributes like coef_) are never read — only constructor params matter.
     *
     * @param object $estimator  The estimator to clone (should be unfitted).
     * @return object            A new, unfitted instance with same hyperparameters.
     */
    public static function cloneEstimator(object $estimator): object
    {
        $rc   = new \ReflectionClass($estimator);
        $ctor = $rc->getConstructor();

        // No constructor → parameterless instantiation
        if ($ctor === null) {
            return $rc->newInstance();
        }

        $args = [];
        foreach ($ctor->getParameters() as $param) {
            $pName = $param->getName();

            try {
                // Read the constructor parameter value from the matching property.
                // For constructor-promoted properties, the property name equals the param name.
                $prop = $rc->getProperty($pName);
                $prop->setAccessible(true);

                if ($prop->isInitialized($estimator)) {
                    $args[] = $prop->getValue($estimator);
                } elseif ($param->isDefaultValueAvailable()) {
                    // Property not yet set (e.g. fitted-only attribute — shouldn't be a ctor param
                    // but guard gracefully)
                    $args[] = $param->getDefaultValue();
                } else {
                    $args[] = null;
                }
            } catch (\ReflectionException) {
                // Property with this name does not exist: use the param's default or null
                $args[] = $param->isDefaultValueAvailable()
                    ? $param->getDefaultValue()
                    : null;
            }
        }

        return $rc->newInstance(...$args);
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /**
     * Gather training and test tensors from index lists using cblas_scopy.
     *
     * Identical gather pattern to DataSplit::train_test_split():
     *   cblas_scopy(d, srcRowPtr, 1, dstRowPtr, 1)  — one BLAS call per row.
     *
     * @param Tensor $X        Full feature matrix [n, d]
     * @param Tensor $y        Full label vector   [n]
     * @param int[]  $trainIdx Training row indices
     * @param int[]  $testIdx  Test row indices
     * @return array{0:Tensor, 1:Tensor, 2:Tensor, 3:Tensor}  [Xtrain, Xtest, ytrain, ytest]
     */
    private static function gatherFold(
        Tensor $X,
        Tensor $y,
        array  $trainIdx,
        array  $testIdx,
    ): array {
        $d      = $X->shape[1];
        $nTrain = count($trainIdx);
        $nTest  = count($testIdx);
        $blas   = BlasEngine::get()->ffi;

        $Xtrain = new Tensor([$nTrain, $d]);
        $Xtest  = new Tensor([$nTest,  $d]);
        $ytrain = new Tensor([$nTrain]);
        $ytest  = new Tensor([$nTest]);

        foreach ($trainIdx as $out => $src) {
            $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$src * $d]));
            $dstPtr = \FFI::cast('float*', \FFI::addr($Xtrain->buffer[$out * $d]));
            $blas->cblas_scopy($d, $srcPtr, 1, $dstPtr, 1);
            $ytrain->buffer[$out] = $y->buffer[$src];
        }

        foreach ($testIdx as $out => $src) {
            $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$src * $d]));
            $dstPtr = \FFI::cast('float*', \FFI::addr($Xtest->buffer[$out * $d]));
            $blas->cblas_scopy($d, $srcPtr, 1, $dstPtr, 1);
            $ytest->buffer[$out] = $y->buffer[$src];
        }

        return [$Xtrain, $Xtest, $ytrain, $ytest];
    }

    /**
     * Apply the scoring function to a single (y_true, y_pred) pair.
     *
     * @param string|callable $scoring  Metric name or scorer callable.
     * @param Tensor          $yTrue    Ground-truth labels [n_test]
     * @param Tensor          $yPred    Predicted labels    [n_test]
     * @return float                    Scalar score for this fold.
     */
    private static function evaluate(string|callable $scoring, Tensor $yTrue, Tensor $yPred): float
    {
        if (is_callable($scoring)) {
            return (float) $scoring($yTrue, $yPred);
        }

        return match ($scoring) {
            // Classification
            'accuracy'               => Metrics::accuracy_score($yTrue, $yPred),
            // Regression
            'r2'                     => Metrics::r2_score($yTrue, $yPred),
            'neg_mean_squared_error' => -Metrics::mean_squared_error($yTrue, $yPred),
            default                  => throw new \InvalidArgumentException(
                "cross_val_score: unknown scoring '{$scoring}'. "
                . "Use 'accuracy', 'r2', 'neg_mean_squared_error', or a callable."
            ),
        };
    }
}
