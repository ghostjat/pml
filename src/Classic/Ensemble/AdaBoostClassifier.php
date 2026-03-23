<?php

declare(strict_types=1);

namespace Pml\Classic\Ensemble;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};
use Pml\Classic\Tree\DecisionTreeClassifier;
use Pml\Classic\ModelSelection\Validation;

// ═══════════════════════════════════════════════════════════════════════════
//  AdaBoostClassifier — sklearn.ensemble.AdaBoostClassifier (SAMME)
//
//  Implements the SAMME (Stagewise Additive Modeling using a Multi-class
//  Exponential loss) algorithm, the multi-class generalisation of the
//  original Freund & Schapire AdaBoost.M1 (Zhu et al. 2009).
//
//  ── SAMME Algorithm ──────────────────────────────────────────────────────
//
//  Notation:
//    N          = n_samples
//    K          = n_classes
//    w[i]       = sample weight (initial 1/N, updated each round)
//    h_t(x)     = prediction of the t-th weak learner
//    α_t        = weight (confidence) of the t-th weak learner
//    err_t      = weighted misclassification rate of h_t
//
//  For t = 1, …, T  (T = n_estimators):
//
//    1. Weighted bootstrap: draw N samples with probability ∝ w[i].
//
//    2. Fit a DecisionTreeClassifier (max_depth=1 "stump" by default) on
//       the bootstrap sample.
//
//    3. Predict on the FULL training set: y_pred = h_t(X).
//
//    4. Compute weighted error:
//         err_t = Σ_i w[i] · I(y_pred[i] ≠ y[i]) / Σ_i w[i]
//       Because w is normalised (Σ w = 1) the denominator is always 1.
//
//    5. SAMME estimator weight (note the +log(K−1) vs binary AdaBoost):
//         α_t = log((1 − err_t) / err_t) + log(K − 1)
//
//       Intuition: log((1-e)/e) is the standard binary term; log(K-1) is a
//       correction for multi-class — it ensures α_t > 0 iff the weak
//       learner beats random chance (err_t < 1 − 1/K), and the final
//       classifier is consistent as T → ∞.
//
//    6. Update sample weights:
//         w[i] ← w[i] · exp(α_t · I(y_pred[i] ≠ y[i]))
//
//       Correctly classified samples (y_pred[i] = y[i]) keep their weight
//       (multiply by exp(0)=1); misclassified samples are boosted by exp(α_t).
//
//    7. Renormalise: w /= Σ w  (so w remains a probability distribution).
//
//  ── Early Stopping ───────────────────────────────────────────────────────
//
//  Two early-stop conditions:
//    a) err_t = 0.0  — perfect weak learner, one round is sufficient.
//    b) err_t >= 1 − 1/K  — weak learner is no better than random chance
//       in a K-class problem; SAMME weight α_t would be ≤ 0 which would
//       shrink rather than boost the signal.  Stop and discard this round.
//
//  ── Weighted Bootstrap ───────────────────────────────────────────────────
//
//  Since DecisionTreeClassifier does not accept sample_weight, we
//  approximate weighted training with weighted resampling: the cumulative
//  distribution function of w[] is built (O(N)) and each bootstrap draw is
//  a binary search into it (O(log N)), giving O(N log N) per round.
//
//  ── Prediction (Hard SAMME) ──────────────────────────────────────────────
//
//  For each sample x and class k:
//    H_k(x) = Σ_t α_t · I(h_t(x) = k)
//  Final prediction = argmax_k H_k(x).
// ═══════════════════════════════════════════════════════════════════════════

final class AdaBoostClassifier implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** @var (Estimator&Predictor)[]  Fitted weak learners. */
    public readonly array $estimators_;

    /** @var float[]  Estimator weights α_t, one per round. */
    public readonly array $estimator_weights_;

    /** @var float[]  Weighted error per round (diagnostic). */
    public readonly array $estimator_errors_;

    /** @var int[]    Sorted distinct class labels seen during fit(). */
    public readonly array $classes_;

    public readonly int $n_classes_;
    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param (Estimator&Predictor)|null $estimator    Base estimator.
     *                                                 Default: DecisionTreeClassifier(max_depth=1).
     * @param int                        $n_estimators Maximum number of boosting rounds.
     * @param float                      $learning_rate Shrinks each estimator's weight by this factor.
     *                                                  (sklearn convention; 1.0 = no shrinkage)
     * @param int                        $random_state  RNG seed.
     */
    public function __construct(
        private readonly ?object $estimator    = null,
        private readonly int     $n_estimators = 50,
        private readonly float   $learning_rate = 1.0,
        private readonly int     $random_state  = 0,
    ) {}

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Run the SAMME boosting loop for up to n_estimators rounds.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  [n_samples] — class labels
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('AdaBoostClassifier: y must be provided.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('AdaBoostClassifier: X must be 2-D.');
        }

        [$n, $d]              = $X->shape;
        $this->n_features_in_ = $d;

        // ── Discover classes ──────────────────────────────────────────
        $labelSet = [];
        for ($i = 0; $i < $n; $i++) {
            $labelSet[(int)(float)$y->buffer[$i]] = true;
        }
        $classArr = array_keys($labelSet);
        sort($classArr);
        $this->classes_   = $classArr;
        $K                = count($classArr);
        $this->n_classes_ = $K;
        $classToIdx       = array_flip($classArr);

        // ── Extract integer class labels into a PHP array ─────────────
        $yInt = [];
        for ($i = 0; $i < $n; $i++) {
            $yInt[$i] = (int)(float)$y->buffer[$i];
        }

        // ── Random threshold for early stop (1 − 1/K) ────────────────
        //
        // If err_t ≥ 1 − 1/K the weak learner is no better than random
        // guessing in a K-class problem — SAMME weights would go negative.
        $randThreshold = 1.0 - 1.0 / $K;

        // ── Initialise uniform sample weights w[i] = 1/N ─────────────
        $w = array_fill(0, $n, 1.0 / $n);

        mt_srand($this->random_state);
        $blas = BlasEngine::get()->ffi;

        $baseEstimator  = $this->estimator ?? new DecisionTreeClassifier(max_depth: 1);
        $estimators     = [];
        $weights        = [];
        $errors         = [];

        for ($t = 0; $t < $this->n_estimators; $t++) {
            // ── Step 1: Weighted bootstrap resample ───────────────────
            //
            // Build the CDF of sample weights.  A uniform random draw
            // u ~ U[0,1] locates a bootstrap index via binary search.
            $cdf = [];
            $cumW = 0.0;
            for ($i = 0; $i < $n; $i++) {
                $cumW  += $w[$i];
                $cdf[$i] = $cumW;
            }
            // Ensure the last bucket reaches exactly 1.0 (floating-point safety)
            $cdf[$n - 1] = 1.0;

            $Xboot = new Tensor([$n, $d]);
            $yboot = new Tensor([$n]);

            for ($i = 0; $i < $n; $i++) {
                // Draw a sample proportional to w using binary search in CDF
                $u   = mt_rand() / mt_getrandmax();
                $src = $this->searchCdf($cdf, $u);

                $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$src * $d]));
                $dstPtr = \FFI::cast('float*', \FFI::addr($Xboot->buffer[$i * $d]));
                $blas->cblas_scopy($d, $srcPtr, 1, $dstPtr, 1);
                $yboot->buffer[$i] = $y->buffer[$src];
            }

            // ── Step 2: Fit weak learner on the bootstrap sample ──────
            $est = Validation::cloneEstimator($baseEstimator);
            $est->fit($Xboot, $yboot);

            // ── Step 3: Predict on the FULL training set ──────────────
            $yPred = $est->predict($X);

            // ── Step 4: Compute weighted misclassification error ──────
            //
            //   err_t = Σ_i w[i] · I(h_t(x_i) ≠ y_i)
            //
            // Since w is normalised (Σ w = 1) this is already in [0,1].
            $errT = 0.0;
            for ($i = 0; $i < $n; $i++) {
                if ((int)(float)$yPred->buffer[$i] !== $yInt[$i]) {
                    $errT += $w[$i];
                }
            }

            // ── Early stop: perfect learner ────────────────────────────
            if ($errT === 0.0) {
                // Perfect fit on training data — α = ∞ → use large finite value
                $estimators[] = $est;
                $weights[]    = $this->learning_rate * 10.0;  // practical cap
                $errors[]     = 0.0;
                break;
            }

            // ── Early stop: weak learner below random chance ───────────
            if ($errT >= $randThreshold) {
                // α_t would be ≤ 0, which would hurt the ensemble.  Stop here.
                break;
            }

            // ── Step 5: SAMME estimator weight ───────────────────────
            //
            //   α_t = η · [log((1 − err_t) / err_t) + log(K − 1)]
            //
            // The extra log(K-1) term vs binary AdaBoost compensates for
            // the fact that a K-class random guesser has error (K-1)/K.
            // For K=2, log(1) = 0 and this reduces to standard AdaBoost.
            $alphaT = $this->learning_rate
                * (log((1.0 - $errT) / $errT) + log($K - 1));

            $estimators[] = $est;
            $weights[]    = $alphaT;
            $errors[]     = $errT;

            // ── Step 6 & 7: Update and renormalise sample weights ─────
            //
            //   w[i] ← w[i] · exp(α_t · I(h_t(x_i) ≠ y_i))
            //
            // Correctly classified samples are multiplied by exp(0) = 1.
            // Misclassified samples are boosted by exp(α_t) > 1, so the
            // next weak learner focuses more on the hard examples.
            $wSum = 0.0;
            for ($i = 0; $i < $n; $i++) {
                if ((int)(float)$yPred->buffer[$i] !== $yInt[$i]) {
                    // Misclassified: increase weight
                    $w[$i] *= exp($alphaT);
                }
                // Correctly classified: weight unchanged (multiply by exp(0)=1)
                $wSum += $w[$i];
            }

            // Renormalise so Σ w = 1 (required for next round's CDF to work)
            for ($i = 0; $i < $n; $i++) {
                $w[$i] /= $wSum;
            }
        }

        $this->estimators_        = $estimators;
        $this->estimator_weights_ = $weights;
        $this->estimator_errors_  = $errors;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Hard SAMME prediction: argmax_k Σ_t α_t · I(h_t(x) = k).
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples]  predicted class labels
     */
    public function predict(Tensor $X): Tensor
    {
        if (!isset($this->estimators_)) {
            throw new \RuntimeException('AdaBoostClassifier is not fitted. Call fit() first.');
        }

        $m          = $X->shape[0];
        $K          = $this->n_classes_;
        $classToIdx = array_flip($this->classes_);

        // H[i][c] = Σ_t α_t · I(h_t(x_i) = class_c)
        // Stored as flat PHP array [m × K] to avoid Tensor allocation overhead
        // for a metadata structure (not a feature-dimension operation).
        $H = array_fill(0, $m * $K, 0.0);

        foreach ($this->estimators_ as $idx => $est) {
            $alphaT = $this->estimator_weights_[$idx];
            $yPred  = $est->predict($X);

            for ($i = 0; $i < $m; $i++) {
                // Locate the predicted class in the forest-wide class index
                $predLabel = (int)(float) $yPred->buffer[$i];
                $c         = $classToIdx[$predLabel] ?? -1;
                if ($c >= 0) {
                    // Accumulate: H[sample i, class c] += α_t
                    $H[$i * $K + $c] += $alphaT;
                }
            }
        }

        // Final prediction: argmax_k H[i][k]
        $out = new Tensor([$m]);
        for ($i = 0; $i < $m; $i++) {
            $bestC = 0;
            $bestV = $H[$i * $K];
            for ($c = 1; $c < $K; $c++) {
                $v = $H[$i * $K + $c];
                if ($v > $bestV) { $bestV = $v; $bestC = $c; }
            }
            $out->buffer[$i] = (float) $this->classes_[$bestC];
        }

        return $out;
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /**
     * Binary search in the cumulative distribution array.
     *
     * Returns the smallest index $k such that $cdf[$k] >= $u.
     * O(log N) per call.
     *
     * @param float[] $cdf  Cumulative distribution array (non-decreasing, last element = 1.0)
     * @param float   $u    Uniform draw in [0, 1]
     * @return int          Sample index
     */
    private function searchCdf(array $cdf, float $u): int
    {
        $lo = 0;
        $hi = count($cdf) - 1;
        while ($lo < $hi) {
            $mid = ($lo + $hi) >> 1;
            if ($cdf[$mid] < $u) {
                $lo = $mid + 1;
            } else {
                $hi = $mid;
            }
        }
        return $lo;
    }
}
