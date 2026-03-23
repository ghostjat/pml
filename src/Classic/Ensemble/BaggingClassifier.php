<?php

declare(strict_types=1);

namespace Pml\Classic\Ensemble;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};
use Pml\Classic\Tree\DecisionTreeClassifier;
use Pml\Classic\ModelSelection\Validation;

// ═══════════════════════════════════════════════════════════════════════════
//  BaggingClassifier — sklearn.ensemble.BaggingClassifier
//
//  Bootstrap Aggregation (bagging) of any Estimator + Predictor.
//  Each base estimator is cloned (via Validation::cloneEstimator()) and
//  trained on a bootstrap sample drawn with replacement from the training
//  set.  Predictions are aggregated by majority vote.
//
//  ── Bootstrap Sampling ───────────────────────────────────────────────────
//
//  For each of the n_estimators trees:
//    1. Draw n_samples row indices uniformly at random WITH replacement
//       using mt_rand(0, n-1).
//    2. Build bootstrap Tensors X_boot [n, d] and y_boot [n] by copying
//       rows from X using cblas_scopy(d, srcPtr, 1, dstPtr, 1) — identical
//       to the RandomForestClassifier bootstrap pattern.
//
//  ── Majority Vote ────────────────────────────────────────────────────────
//
//  For each test sample, each estimator casts one vote for its predicted
//  class label.  The label with the most votes wins.  Ties are broken by
//  the smallest label (PHP's arsort behavior on equal counts).
//
//  ── Estimator Cloning ────────────────────────────────────────────────────
//
//  Uses Validation::cloneEstimator() (Reflection-based, mirroring
//  sklearn.base.clone()) to produce independent unfitted copies of the
//  base estimator with identical hyperparameters.
// ═══════════════════════════════════════════════════════════════════════════

final class BaggingClassifier implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** @var (Estimator&Predictor)[]  Fitted ensemble members. */
    public readonly array $estimators_;

    /** @var int[]  Distinct class labels seen in training, sorted. */
    public readonly array $classes_;

    public readonly int $n_classes_;
    public readonly int $n_features_in_;
    public readonly int $n_estimators_fitted_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param (Estimator&Predictor)|null $estimator    Base estimator to clone.
     *                                                 Default: DecisionTreeClassifier.
     * @param int                        $n_estimators Number of bootstrap estimators.
     * @param int                        $random_state RNG seed.
     */
    public function __construct(
        private readonly ?object $estimator    = null,
        private readonly int     $n_estimators = 10,
        private readonly int     $random_state = 0,
    ) {}

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Bootstrap and fit n_estimators independent clones of the base estimator.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  [n_samples] — class labels
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('BaggingClassifier: y must be provided.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('BaggingClassifier: X must be 2-D.');
        }

        [$n, $d]              = $X->shape;
        $this->n_features_in_ = $d;

        // ── Discover all classes from the full training set ────────────
        //
        // Done BEFORE bootstrapping so that the classes_ array reflects all
        // possible labels even if a particular bootstrap sample omits rare classes.
        $labelSet = [];
        for ($i = 0; $i < $n; $i++) {
            $labelSet[(int)(float)$y->buffer[$i]] = true;
        }
        $classArr = array_keys($labelSet);
        sort($classArr);
        $this->classes_   = $classArr;
        $this->n_classes_ = count($classArr);

        // Resolve base estimator (default: DecisionTreeClassifier)
        $baseEstimator = $this->estimator ?? new DecisionTreeClassifier();

        mt_srand($this->random_state);
        $blas      = BlasEngine::get()->ffi;
        $fitted    = [];

        for ($t = 0; $t < $this->n_estimators; $t++) {
            // ── Clone base estimator ───────────────────────────────────
            $est = Validation::cloneEstimator($baseEstimator);

            // ── Bootstrap sample (with replacement) ───────────────────
            //
            // Draw n random indices in [0, n-1]; cblas_scopy copies each
            // row in one BLAS call (no PHP element loops over features).
            $Xboot = new Tensor([$n, $d]);
            $yboot = new Tensor([$n]);

            for ($i = 0; $i < $n; $i++) {
                $src    = mt_rand(0, $n - 1);
                $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$src * $d]));
                $dstPtr = \FFI::cast('float*', \FFI::addr($Xboot->buffer[$i * $d]));
                $blas->cblas_scopy($d, $srcPtr, 1, $dstPtr, 1);
                $yboot->buffer[$i] = $y->buffer[$src];
            }

            // ── Fit clone on bootstrap data ────────────────────────────
            $est->fit($Xboot, $yboot);
            $fitted[] = $est;
        }

        $this->estimators_        = $fitted;
        $this->n_estimators_fitted_ = count($fitted);

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict class labels by majority vote over all base estimators.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples]  predicted class labels (float32)
     */
    public function predict(Tensor $X): Tensor
    {
        if (!isset($this->estimators_)) {
            throw new \RuntimeException('BaggingClassifier is not fitted. Call fit() first.');
        }

        $m   = $X->shape[0];
        $out = new Tensor([$m]);

        // ── Accumulate vote counts: votes[sample][label] ───────────────
        //
        // PHP array of arrays is used for vote counting — this is metadata
        // (not high-dimensional float data), so it lives in PHP-land.
        $votes = array_fill(0, $m, []);

        foreach ($this->estimators_ as $est) {
            $yPred = $est->predict($X);
            for ($i = 0; $i < $m; $i++) {
                $lbl = (int)(float) $yPred->buffer[$i];
                $votes[$i][$lbl] = ($votes[$i][$lbl] ?? 0) + 1;
            }
        }

        // ── Pick the label with the highest vote count ─────────────────
        for ($i = 0; $i < $m; $i++) {
            arsort($votes[$i]);       // sort descending by vote count
            reset($votes[$i]);
            $out->buffer[$i] = (float) key($votes[$i]);
        }

        return $out;
    }
}
