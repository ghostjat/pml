<?php

declare(strict_types=1);

namespace Pml\Classic\NaiveBayes;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  MultinomialNB — sklearn.naive_bayes.MultinomialNB
//
//  Naïve Bayes for multinomially-distributed data (e.g., word counts).
//  Assumes each feature is a non-negative count; the likelihood of a sample
//  x given class c is:
//
//    P(x | c) ∝ Π_j  θ_{c,j}^{x_j}
//
//  where θ_{c,j} = P(feature j | class c) is smoothed to avoid zero probs.
//
//  ── Additive (Laplace/Lidstone) Smoothing ─────────────────────────────────
//
//  With smoothing parameter α:
//
//    θ_{c,j} = (N_{c,j} + α) / (N_c + α·d)
//
//  where N_{c,j} = Σ_{i: y_i=c} X_{i,j}  (total count of feature j in c)
//        N_c     = Σ_j N_{c,j}            (total count of all features in c)
//        d       = number of features
//
//  In log-space (for numerical stability):
//
//    log θ_{c,j} = log(N_{c,j} + α) − log(N_c + α·d)
//
//  ── Fit via BLAS ──────────────────────────────────────────────────────────
//
//  Build a one-hot indicator matrix Y [n, K]:
//    Y[i, c] = 1.0 if y_i == c, else 0.0
//
//  Class feature sums [K, d]:
//    feature_counts_ = Y^T @ X = sgemm(Trans, NoTrans, K, d, n)
//
//  Class total counts [K]:
//    N_c = feature_counts_ @ ones_d  = sgemv(NoTrans, K, d, ...)
//
//  Then compute log θ element-wise in a K×d PHP loop (O(K·d), negligible).
//
//  ── Predict via BLAS ──────────────────────────────────────────────────────
//
//  For a batch of m test samples X [m, d]:
//
//    logJoints [m, K] = X @ feature_log_prob_^T
//                     = sgemm(NoTrans, Trans, m, K, d, 1.0, X, d, logLik, d)
//
//  Broadcast log-priors:
//    sger(m, K, 1.0, ones_m, 1, log_class_prior_, 1, logJoints, K)
//
//  Normalise via log-sum-exp per row to obtain probabilities:
//    maxLog      = max_c logJoint[i,c]
//    log_denom   = maxLog + log(Σ_c exp(logJoint[i,c] − maxLog))
//    P(c | x_i) = exp(logJoint[i,c] − log_denom)
//
//  ── Complexity ────────────────────────────────────────────────────────────
//
//  fit:     O(n·K + K·d)  dominated by Y^T @ X  sgemm
//  predict: O(m·K·d)      dominated by X @ logLik^T  sgemm
// ═══════════════════════════════════════════════════════════════════════════

final class MultinomialNB implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Log of smoothed per-class feature probabilities log θ_{c,j}.
     * Shape: [n_classes, n_features]  (flat 1D row-major).
     */
    public readonly Tensor $feature_log_prob_;

    /**
     * Log of class prior probabilities log P(c).
     * Shape: [n_classes].
     */
    public readonly Tensor $class_log_prior_;

    /**
     * Raw (un-normalised, un-smoothed) feature count sums per class.
     * Shape: [n_classes, n_features].
     */
    public readonly Tensor $feature_count_;

    /**
     * Number of samples per class.
     * Shape: [n_classes].
     */
    public readonly Tensor $class_count_;

    /** Unique class labels sorted ascending. @var int[] */
    public readonly array $classes_;

    public readonly int $n_classes_;
    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param float $alpha  Additive (Laplace/Lidstone) smoothing parameter.
     *                      α = 1.0 → Laplace smoothing (sklearn default).
     *                      α = 0.0 → no smoothing (may cause log(0) for unseen features).
     *                      α ∈ (0,1) → Lidstone smoothing.
     */
    public function __construct(
        private readonly float $alpha = 1.0,
    ) {
        if ($alpha < 0.0) {
            throw new \InvalidArgumentException('MultinomialNB: alpha must be ≥ 0.');
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Fit the model by computing smoothed log-probabilities from training data.
     *
     * @param Tensor      $X  Non-negative feature counts [n_samples, n_features].
     * @param Tensor|null $y  Integer class labels [n_samples].
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('MultinomialNB::fit() requires target $y.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('MultinomialNB::fit() requires a 2-D feature matrix X.');
        }

        [$n, $d] = $X->shape;
        $blas    = BlasEngine::get()->ffi;
        $alpha   = $this->alpha;

        // ── Discover unique class labels ───────────────────────────────────
        $seen = [];
        for ($i = 0; $i < $n; $i++) {
            $seen[(int) round((float) $y->buffer[$i])] = true;
        }
        ksort($seen);
        $classes    = array_keys($seen);
        $nC         = count($classes);
        $classToPos = array_flip($classes);

        // ── Build one-hot indicator matrix Y [n, K] ────────────────────────
        //
        // Y[i, c] = 1.0 if y_i == c, else 0.0.
        // Used as the right-hand factor in Y^T @ X.
        $Y = new Tensor([$n, $nC]);   // zero-initialised

        $classCountArr = array_fill(0, $nC, 0);

        for ($i = 0; $i < $n; $i++) {
            $lbl = (int) round((float) $y->buffer[$i]);
            $c   = $classToPos[$lbl];
            $Y->buffer[$i * $nC + $c] = 1.0;
            $classCountArr[$c]++;
        }

        // ── Class feature sums: featureCounts [K, d] = Y^T @ X ────────────
        //
        //   sgemm(Order=RowMajor, TransA=Trans, TransB=NoTrans,
        //         M=K, N=d, K=n,
        //         alpha=1.0, A=Y, lda=K, B=X, ldb=d,
        //         beta=0.0,  C=featureCounts, ldc=d)
        //
        // Result: featureCounts[c, j] = Σ_{i} Y[i,c] · X[i,j]
        //                             = Σ_{i: y_i=c} X[i,j]
        $featureCounts = new Tensor([$nC, $d]);
        $blas->cblas_sgemm(
            101,      // CblasRowMajor
            112,      // CblasTrans   (A = Y is transposed)
            111,      // CblasNoTrans (B = X)
            $nC, $d, $n,
            1.0, $Y->buffer, $nC, $X->buffer, $d,
            0.0, $featureCounts->buffer, $d
        );

        unset($Y);

        // ── Compute smoothed log θ_{c,j} and log-priors ───────────────────
        //
        // For each class c:
        //   N_c = Σ_j featureCounts[c,j]   (total feature count in c)
        //   smoothed_count[c,j] = featureCounts[c,j] + α
        //   log θ_{c,j}         = log(smoothed_count[c,j]) − log(N_c + α·d)
        $featureLogProb = new Tensor([$nC, $d]);
        $alphaDtimes    = $alpha * $d;

        for ($c = 0; $c < $nC; $c++) {
            $base    = $c * $d;
            $totalNc = 0.0;
            for ($j = 0; $j < $d; $j++) {
                $totalNc += (float) $featureCounts->buffer[$base + $j];
            }
            $logDenom = log($totalNc + $alphaDtimes);

            for ($j = 0; $j < $d; $j++) {
                $smoothedCount = (float) $featureCounts->buffer[$base + $j] + $alpha;
                $featureLogProb->buffer[$base + $j] =
                    (float) log($smoothedCount) - $logDenom;
            }
        }

        // ── Log-priors log P(c) = log(n_c / n) ────────────────────────────
        $classCount    = new Tensor([$nC]);
        $classLogPrior = new Tensor([$nC]);
        $logN          = log((float) $n);

        for ($c = 0; $c < $nC; $c++) {
            $nc = (float) $classCountArr[$c];
            $classCount->buffer[$c]    = $nc;
            $classLogPrior->buffer[$c] = (float) log($nc) - $logN;
        }

        // ── Store fitted attributes ────────────────────────────────────────
        $this->feature_log_prob_ = $featureLogProb;
        $this->class_log_prior_  = $classLogPrior;
        $this->feature_count_    = $featureCounts;
        $this->class_count_      = $classCount;
        $this->classes_          = $classes;
        $this->n_classes_        = $nC;
        $this->n_features_in_    = $d;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict class labels.
     *
     * @param Tensor $X  Feature counts [n_samples, n_features].
     * @return Tensor    Predicted labels [n_samples].
     */
    public function predict(Tensor $X): Tensor
    {
        $this->checkFitted();

        $logJoints = $this->computeLogJoints($X);   // [m, K] flat Tensor
        $m         = $X->shape[0];
        $nC        = $this->n_classes_;
        $out       = new Tensor([$m]);

        for ($i = 0; $i < $m; $i++) {
            $base    = $i * $nC;
            $bestPos = 0;
            $bestVal = (float) $logJoints->buffer[$base];
            for ($c = 1; $c < $nC; $c++) {
                $v = (float) $logJoints->buffer[$base + $c];
                if ($v > $bestVal) { $bestVal = $v; $bestPos = $c; }
            }
            $out->buffer[$i] = (float) $this->classes_[$bestPos];
        }

        return $out;
    }

    /**
     * Predict class probabilities via log-sum-exp normalisation.
     *
     * @param Tensor $X  Feature counts [n_samples, n_features].
     * @return Tensor    Probability matrix [n_samples, n_classes].
     */
    public function predict_proba(Tensor $X): Tensor
    {
        $this->checkFitted();

        $logJoints = $this->computeLogJoints($X);   // [m, K] flat Tensor
        $m         = $X->shape[0];
        $nC        = $this->n_classes_;
        $out       = new Tensor([$m, $nC]);

        // Log-sum-exp per row to obtain normalised probabilities
        for ($i = 0; $i < $m; $i++) {
            $base = $i * $nC;

            // Find row maximum for numerical stability
            $maxLog = (float) $logJoints->buffer[$base];
            for ($c = 1; $c < $nC; $c++) {
                $v = (float) $logJoints->buffer[$base + $c];
                if ($v > $maxLog) { $maxLog = $v; }
            }

            // Σ exp(lj_c − maxLog)
            $sumExp = 0.0;
            for ($c = 0; $c < $nC; $c++) {
                $sumExp += exp((float) $logJoints->buffer[$base + $c] - $maxLog);
            }
            $logDenom = $maxLog + log($sumExp);

            for ($c = 0; $c < $nC; $c++) {
                $out->buffer[$base + $c] =
                    (float) exp((float) $logJoints->buffer[$base + $c] - $logDenom);
            }
        }

        return $out;
    }

    /**
     * Accuracy score on test data.
     */
    public function score(Tensor $X, Tensor $y): float
    {
        $pred = $this->predict($X);
        $n    = $y->size;
        $ok   = 0;
        for ($i = 0; $i < $n; $i++) {
            if ((int) round((float) $y->buffer[$i]) === (int) round((float) $pred->buffer[$i])) {
                $ok++;
            }
        }
        return $ok / $n;
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * Compute log-joint scores for all samples: X @ feature_log_prob^T + log_prior.
     *
     * ── BLAS ──────────────────────────────────────────────────────────────
     *
     *   sgemm(NoTrans, Trans, m, K, d, 1.0,
     *         X [m×d], d,  feature_log_prob [K×d], d,
     *         0.0, logJoints [m×K], K)
     *
     *   → logJoints[i,c] = Σ_j X[i,j] · log θ_{c,j}
     *
     *   Then broadcast log-priors across rows:
     *   sger(m, K, 1.0, ones_m, 1, class_log_prior_, 1, logJoints, K)
     *   → logJoints[i,c] += log P(c)
     *
     * @return Tensor  [m, K] flat row-major
     */
    private function computeLogJoints(Tensor $X): Tensor
    {
        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "MultinomialNB::predict() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        $m    = $X->shape[0];
        $d    = $this->n_features_in_;
        $nC   = $this->n_classes_;
        $blas = BlasEngine::get()->ffi;

        // logJoints [m, K] = X @ feature_log_prob^T
        $logJoints = new Tensor([$m, $nC]);
        $blas->cblas_sgemm(
            101,   // CblasRowMajor
            111,   // CblasNoTrans (X)
            112,   // CblasTrans   (feature_log_prob transposed)
            $m, $nC, $d,
            1.0, $X->buffer, $d, $this->feature_log_prob_->buffer, $d,
            0.0, $logJoints->buffer, $nC
        );

        // Broadcast log-priors: logJoints[i,c] += log P(c)
        $onesM = Tensor::ones([$m]);
        $blas->cblas_sger(
            101, $m, $nC,
            1.0, $onesM->buffer, 1, $this->class_log_prior_->buffer, 1,
            $logJoints->buffer, $nC
        );

        return $logJoints;
    }

    private function checkFitted(): void
    {
        if (!isset($this->feature_log_prob_)) {
            throw new \RuntimeException('MultinomialNB is not fitted. Call fit() first.');
        }
    }
}
