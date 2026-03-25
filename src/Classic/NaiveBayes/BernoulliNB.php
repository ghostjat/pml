<?php

declare(strict_types=1);

namespace Pml\Classic\NaiveBayes;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  BernoulliNB — sklearn.naive_bayes.BernoulliNB
//
//  Naïve Bayes for multivariate Bernoulli models (binary / boolean features).
//  Assumes each feature is a binary indicator; the likelihood of a sample x
//  given class c is:
//
//    P(x | c) = Π_j  p_{c,j}^{x_j} · (1 − p_{c,j})^{1−x_j}
//
//  where p_{c,j} = P(x_j = 1 | c).
//
//  Note: unlike MultinomialNB, non-zero features are NOT considered independently
//  of zero features — a 0 in feature j explicitly contributes (1 − p_{c,j}).
//
//  ── Additive (Laplace/Lidstone) Smoothing ─────────────────────────────────
//
//    p_{c,j} = (N_{c,j} + α) / (N_c + 2α)
//
//  where N_{c,j} = Σ_{i: y_i=c} binarize(X_{i,j})
//        N_c     = number of training samples in class c
//        α       = smoothing parameter (1.0 → Laplace; default)
//
//  ── Log-ratio Prediction Trick ─────────────────────────────────────────────
//
//  The log-likelihood log P(x | c) can be rewritten as:
//
//    log P(x | c) = Σ_j x_j · log(p_{c,j} / (1 − p_{c,j}))
//                 + Σ_j log(1 − p_{c,j})
//
//  Let:
//    log_ratio_[c,j]   = log(p_{c,j} / (1 − p_{c,j}))   (log-odds)
//    neg_log_sum_[c]   = Σ_j log(1 − p_{c,j})            (constant per class)
//
//  Then:
//    logLik[i,c] = x_i · log_ratio_[c,:]  +  neg_log_sum_[c]
//
//  Expanded for a batch:
//    logLiks [m, K] = X @ log_ratio_^T
//                   = sgemm(NoTrans, Trans, m, K, d)
//
//  Then broadcast (log_class_prior_ + neg_log_sum_) per row:
//    sger(m, K, 1.0, ones_m, bias_per_class, logLiks, K)
//
//  This formulation is identical to MultinomialNB's BLAS pattern but uses the
//  log-ratio matrix instead of feature_log_prob, and includes an additional
//  negative-log-complement constant per class.
//
//  ── Binarization ──────────────────────────────────────────────────────────
//
//  If $binarize is a float threshold t:
//    X_bin[i,j] = 1.0 if X[i,j] > t, else 0.0
//
//  If $binarize is null: X is assumed to already be binary (0.0 / 1.0).
//  Binarization is applied at both fit and predict time.
//
//  ── Complexity ────────────────────────────────────────────────────────────
//
//  fit:     O(n·K + K·d)  dominated by Y^T @ X_bin  sgemm
//  predict: O(m·K·d)      dominated by X_bin @ log_ratio^T  sgemm
// ═══════════════════════════════════════════════════════════════════════════

final class BernoulliNB implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Log-odds per class and feature: log(p_{c,j} / (1 − p_{c,j})).
     * Shape: [n_classes, n_features]  (flat 1D row-major).
     */
    public readonly Tensor $log_ratio_;

    /**
     * Per-class bias term: log P(c) + Σ_j log(1 − p_{c,j}).
     * Shape: [n_classes].
     */
    public readonly Tensor $class_log_prior_;

    /**
     * Raw binarized feature count sums per class N_{c,j}.
     * Shape: [n_classes, n_features].
     */
    public readonly Tensor $feature_count_;

    /**
     * Number of training samples per class.
     * Shape: [n_classes].
     */
    public readonly Tensor $class_count_;

    /** Unique class labels sorted ascending. @var int[] */
    public readonly array $classes_;

    public readonly int $n_classes_;
    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param float      $alpha     Additive (Laplace/Lidstone) smoothing parameter (≥ 0).
     * @param float|null $binarize  Threshold for binarizing features.  If null, X must
     *                              already be binary.  Binarization is applied at both
     *                              fit() and predict() time.
     */
    public function __construct(
        private readonly float      $alpha     = 1.0,
        private readonly float|null $binarize  = 0.0,
    ) {
        if ($alpha < 0.0) {
            throw new \InvalidArgumentException('BernoulliNB: alpha must be ≥ 0.');
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Fit the model by computing smoothed log-ratios from training data.
     *
     * @param Tensor      $X  Binary (or binarizable) feature matrix [n_samples, n_features].
     * @param Tensor|null $y  Integer class labels [n_samples].
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('BernoulliNB::fit() requires target $y.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('BernoulliNB::fit() requires a 2-D feature matrix X.');
        }

        [$n, $d] = $X->shape;
        $blas    = BlasEngine::get()->ffi;
        $alpha   = $this->alpha;

        // Binarize X if requested
        $Xbin = $this->maybeBinarize($X, $n, $d);

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
        $Y             = new Tensor([$n, $nC]);
        $classCountArr = array_fill(0, $nC, 0);

        for ($i = 0; $i < $n; $i++) {
            $lbl = (int) round((float) $y->buffer[$i]);
            $c   = $classToPos[$lbl];
            $Y->buffer[$i * $nC + $c] = 1.0;
            $classCountArr[$c]++;
        }

        // ── Binary feature count sums: featureCounts [K, d] = Y^T @ X_bin ─
        //
        //   sgemm(RowMajor, Trans, NoTrans, K, d, n,
        //         1.0, Y [n×K], lda=K, Xbin [n×d], ldb=d,
        //         0.0, featureCounts [K×d], ldc=d)
        $featureCounts = new Tensor([$nC, $d]);
        $blas->cblas_sgemm(
            101,         // CblasRowMajor
            112,         // CblasTrans   (Y is transposed → gives Y^T @ X)
            111,         // CblasNoTrans (X_bin)
            $nC, $d, $n,
            1.0, $Y->buffer, $nC, $Xbin->buffer, $d,
            0.0, $featureCounts->buffer, $d
        );

        unset($Y, $Xbin);

        // ── Compute smoothed p_{c,j}, log-ratio, and neg_log_sum_ ─────────
        //
        // p_{c,j}    = (N_{c,j} + α) / (N_c + 2α)
        // log_ratio  = log(p) − log(1 − p)   [= log(p/(1-p))]
        // neg_log_sum_[c] = Σ_j log(1 − p_{c,j})
        $logRatio   = new Tensor([$nC, $d]);
        $negLogSums = [];

        for ($c = 0; $c < $nC; $c++) {
            $base       = $c * $d;
            $nc         = (float) $classCountArr[$c];
            $denom      = $nc + 2.0 * $alpha;
            $negLogSum  = 0.0;

            for ($j = 0; $j < $d; $j++) {
                $p = ((float) $featureCounts->buffer[$base + $j] + $alpha) / $denom;

                // Clamp to (0, 1) to avoid log(0)
                if ($p <= 0.0) { $p = 1e-10; }
                if ($p >= 1.0) { $p = 1.0 - 1e-10; }

                $logP                            = log($p);
                $logOneMinusP                    = log(1.0 - $p);
                $logRatio->buffer[$base + $j]    = (float) ($logP - $logOneMinusP);
                $negLogSum                       += $logOneMinusP;
            }
            $negLogSums[$c] = $negLogSum;
        }

        // ── Class counts and log-priors ────────────────────────────────────
        $classCount    = new Tensor([$nC]);
        $classLogPrior = new Tensor([$nC]);    // stores log P(c) + neg_log_sum_[c]
        $logN          = log((float) $n);

        for ($c = 0; $c < $nC; $c++) {
            $nc = (float) $classCountArr[$c];
            $classCount->buffer[$c]    = $nc;
            // Fold neg_log_sum into the per-class bias for a single sger broadcast
            $classLogPrior->buffer[$c] = (float) (log($nc) - $logN + $negLogSums[$c]);
        }

        // ── Store fitted attributes ────────────────────────────────────────
        $this->log_ratio_        = $logRatio;
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
     * @param Tensor $X  Binary features [n_samples, n_features].
     * @return Tensor    Predicted labels [n_samples].
     */
    public function predict(Tensor $X): Tensor
    {
        $this->checkFitted();

        $logJoints = $this->computeLogJoints($X);
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
     * @param Tensor $X  Binary features [n_samples, n_features].
     * @return Tensor    Probability matrix [n_samples, n_classes].
     */
    public function predict_proba(Tensor $X): Tensor
    {
        $this->checkFitted();

        $logJoints = $this->computeLogJoints($X);
        $m         = $X->shape[0];
        $nC        = $this->n_classes_;
        $out       = new Tensor([$m, $nC]);

        for ($i = 0; $i < $m; $i++) {
            $base = $i * $nC;

            $maxLog = (float) $logJoints->buffer[$base];
            for ($c = 1; $c < $nC; $c++) {
                $v = (float) $logJoints->buffer[$base + $c];
                if ($v > $maxLog) { $maxLog = $v; }
            }

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
     * Compute log-joint scores for all samples.
     *
     * ── BLAS ──────────────────────────────────────────────────────────────
     *
     *   sgemm(NoTrans, Trans, m, K, d, 1.0,
     *         X_bin [m×d], d,  log_ratio [K×d], d,
     *         0.0, logJoints [m×K], K)
     *
     *   → logJoints[i,c] = Σ_j X_bin[i,j] · log_ratio[c,j]
     *
     *   Then broadcast per-class bias (log_prior + neg_log_sum):
     *   sger(m, K, 1.0, ones_m, 1, class_log_prior_, 1, logJoints, K)
     *   → logJoints[i,c] += log P(c) + Σ_j log(1 − p_{c,j})
     *
     * @return Tensor  [m, K] flat row-major
     */
    private function computeLogJoints(Tensor $X): Tensor
    {
        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "BernoulliNB::predict() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        $m    = $X->shape[0];
        $d    = $this->n_features_in_;
        $nC   = $this->n_classes_;
        $blas = BlasEngine::get()->ffi;

        // Binarize if needed
        $Xbin = $this->maybeBinarize($X, $m, $d);

        // logJoints [m, K] = X_bin @ log_ratio^T
        $logJoints = new Tensor([$m, $nC]);
        $blas->cblas_sgemm(
            101,   // CblasRowMajor
            111,   // CblasNoTrans (X_bin)
            112,   // CblasTrans   (log_ratio transposed)
            $m, $nC, $d,
            1.0, $Xbin->buffer, $d, $this->log_ratio_->buffer, $d,
            0.0, $logJoints->buffer, $nC
        );

        // Broadcast per-class bias (log_prior + neg_log_sum)
        $onesM = Tensor::ones([$m]);
        $blas->cblas_sger(
            101, $m, $nC,
            1.0, $onesM->buffer, 1, $this->class_log_prior_->buffer, 1,
            $logJoints->buffer, $nC
        );

        return $logJoints;
    }

    /**
     * Apply threshold binarization to X if $this->binarize is non-null.
     * Returns the same Tensor unchanged if binarize is null.
     * Always returns a fresh Tensor when binarizing (does not modify $X in place).
     */
    private function maybeBinarize(Tensor $X, int $rows, int $cols): Tensor
    {
        if ($this->binarize === null) {
            return $X;
        }

        $t    = $this->binarize;
        $size = $rows * $cols;
        $out  = new Tensor([$rows, $cols]);

        for ($k = 0; $k < $size; $k++) {
            $out->buffer[$k] = ((float) $X->buffer[$k] > $t) ? 1.0 : 0.0;
        }

        return $out;
    }

    private function checkFitted(): void
    {
        if (!isset($this->log_ratio_)) {
            throw new \RuntimeException('BernoulliNB is not fitted. Call fit() first.');
        }
    }
}
