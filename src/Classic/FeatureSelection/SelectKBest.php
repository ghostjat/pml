<?php

declare(strict_types=1);

namespace Pml\Classic\FeatureSelection;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Transformer};

// ═══════════════════════════════════════════════════════════════════════════
//  SelectKBest — sklearn.feature_selection.SelectKBest
//
//  Retains the top K features ranked by a univariate statistical score.
//  The built-in scoring function is f_classif (ANOVA F-value), which tests
//  whether each feature's mean differs significantly across class groups.
//
//  ── ANOVA F-value (f_classif) ─────────────────────────────────────────────
//
//  For feature j, with K classes of sizes n_1, …, n_K and grand total n:
//
//    grand mean:   μ_j  = (1/n) Σ_i x_{ij}
//    class mean:   μ_{kj} = (1/n_k) Σ_{i:y_i=k} x_{ij}
//
//    SS_between  = Σ_k n_k (μ_{kj} − μ_j)²
//    SS_within   = Σ_k Σ_{i:y_i=k} (x_{ij} − μ_{kj})²
//               = Σ_k [ Σ_{i:y_i=k} x_{ij}² − n_k μ_{kj}² ]
//
//    F(j)  = (SS_between / (K−1)) / (SS_within / (n−K))
//
//  Features with higher F have a larger between-group mean difference
//  relative to within-group spread — i.e. more discriminative for y.
//
//  ── BLAS strategy ─────────────────────────────────────────────────────────
//
//  The inner sums are computed via three BLAS calls before any PHP loop:
//
//  1. Grand means — single sgemv (Trans):
//       grand_means = (1/n) X^T ones_n
//
//  2. Class sums — single sgemm (X^T @ Y):
//       class_sums[j, k] = Σ_{i:y_i=k} X[i, j]
//       Y[n, K]: one-hot class indicator matrix
//
//  3. Class sum-of-squares — single sgemm (X_sq^T @ Y):
//       class_sumSq[j, k] = Σ_{i:y_i=k} X[i, j]²
//       X_sq: element-wise square of X (one O(n·d) PHP pass to build)
//
//  After these three C-level calls, the F-statistic for all d features is
//  computed in a compact O(d·K) PHP loop — typically very fast.
//
//  Transform — strided column copy (identical to VarianceThreshold):
//    cblas_scopy(n, &X[0, src_col], stride=d_in, &out[0, dst_col], stride=d_out)
//
//  ── Selection rule ────────────────────────────────────────────────────────
//
//  After computing scores_, features are ranked by score descending.
//  The top min(k, n_features) are kept.  Ties are broken by lower feature
//  index (stable sort in ascending index order within the top-k set).
//
//  ── Custom scoring functions ──────────────────────────────────────────────
//
//  Pass any callable(Tensor $X, Tensor $y): float[] as $score_func to
//  replace f_classif.  The callable must return a float[] of length d
//  where higher = more important.
// ═══════════════════════════════════════════════════════════════════════════

final class SelectKBest implements Estimator, Transformer
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Per-feature scores (ANOVA F-values, or custom scoring function output).
     * @var float[]
     */
    public readonly array $scores_;

    /**
     * Per-feature p-values.
     * Computing exact p-values requires the regularised incomplete Beta
     * function (F-distribution CDF), which has no PHP built-in.
     * Values are set to NAN.  Pass a custom score_func that returns
     * p-values separately if needed.
     * @var float[]
     */
    public readonly array $pvalues_;

    /**
     * Boolean support mask: true = kept, false = dropped.
     * @var bool[]
     */
    public readonly array $get_support_;

    public readonly int $n_features_in_;
    public readonly int $n_features_out_;

    /** Indices of kept features (top-k by score). */
    private readonly array $keptIndices_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int           $k           Number of top features to keep.
     *                                   Use PHP_INT_MAX to keep all features
     *                                   (useful when only scores_ matter).
     * @param callable|null $score_func  Scoring function: (Tensor $X, Tensor $y): float[].
     *                                   Default null → use built-in f_classif.
     */
    public function __construct(
        private readonly int   $k          = 10,
        private readonly mixed $score_func = null,
    ) {
        if ($k < 1) {
            throw new \InvalidArgumentException('SelectKBest: k must be ≥ 1.');
        }
        if ($score_func !== null && !is_callable($score_func)) {
            throw new \InvalidArgumentException('SelectKBest: score_func must be callable or null.');
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Compute feature scores and build the top-k selection mask.
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Class labels [n_samples] — required for f_classif.
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('SelectKBest: X must be 2-D [n_samples, n_features].');
        }
        if ($y === null && $this->score_func === null) {
            throw new \InvalidArgumentException(
                'SelectKBest: y must be provided when using the default f_classif scorer.'
            );
        }

        [$n, $d] = $X->shape;

        // ── Compute scores ─────────────────────────────────────────────────
        $scores = ($this->score_func !== null)
            ? ($this->score_func)($X, $y)
            : $this->fClassif($X, $y, $n, $d);

        if (count($scores) !== $d) {
            throw new \RuntimeException(
                "SelectKBest: score_func returned " . count($scores) . " scores but expected {$d}."
            );
        }

        // ── Rank by score (descending), stable by feature index ────────────
        //
        // arsort preserves keys (feature indices) and sorts values descending.
        // This gives us the feature indices ordered from highest to lowest score.
        $indexed = $scores;
        arsort($indexed);

        $kActual = min($this->k, $d);
        $topK    = array_slice(array_keys($indexed), 0, $kActual, preserve_keys: false);

        // Re-sort kept indices in ascending order for clean transform output
        // (so column 0 of the output always corresponds to the original lower index)
        sort($topK);

        // ── Build support mask ─────────────────────────────────────────────
        $keptSet = array_fill_keys($topK, true);
        $support = [];
        for ($j = 0; $j < $d; $j++) {
            $support[$j] = isset($keptSet[$j]);
        }

        $this->scores_         = array_values($scores);
        $this->pvalues_        = array_fill(0, $d, NAN);  // see docblock
        $this->get_support_    = $support;
        $this->keptIndices_    = $topK;
        $this->n_features_in_  = $d;
        $this->n_features_out_ = count($topK);

        return $this;
    }

    // ── Transformer ───────────────────────────────────────────────────────

    /**
     * Reduce X to the selected top-k feature columns.
     *
     * Uses cblas_scopy with non-unit source stride to extract columns
     * without allocating intermediate buffers — same technique as
     * VarianceThreshold::transform().
     *
     * @param Tensor $X  [n_samples, n_features_in]
     * @return Tensor    [n_samples, k]
     */
    public function transform(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "SelectKBest::transform() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        [$n, $dIn] = $X->shape;
        $dOut = $this->n_features_out_;
        $blas = BlasEngine::get()->ffi;
        $out  = new Tensor([$n, $dOut]);

        // Copy each kept column using strided cblas_scopy:
        //   src: X->buffer[$srcCol], stride $dIn  (walks down the source column)
        //   dst: out->buffer[$dstCol], stride $dOut (walks down the output column)
        foreach ($this->keptIndices_ as $dstCol => $srcCol) {
            $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$srcCol]));
            $dstPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$dstCol]));
            $blas->cblas_scopy($n, $srcPtr, $dIn, $dstPtr, $dOut);
        }

        return $out;
    }

    public function fit_transform(Tensor $X, ?Tensor $y = null): Tensor
    {
        return $this->fit($X, $y)->transform($X);
    }

    /**
     * Return the indices of the kept features.
     * Mirrors sklearn's get_support(indices=True).
     *
     * @return int[]
     */
    public function getSupportIndices(): array
    {
        $this->checkFitted();
        return $this->keptIndices_;
    }

    // ── f_classif ─────────────────────────────────────────────────────────

    /**
     * Compute the ANOVA F-value for each feature vs. the class target.
     *
     * ── BLAS-accelerated computation ──────────────────────────────────────
     *
     * Step 1 — Grand means [1 × sgemv call]:
     *   grand_means = (1/n) X^T ones_n
     *   cblas_sgemv(Trans, n, d, 1/n, X, d, ones, 1, 0, grand_means, 1)
     *
     * Step 2 — Class indicator matrix Y [n, K] (PHP loop, O(n)):
     *   Y[i, k] = 1.0 if y[i] == class_k else 0.0
     *
     * Step 3 — Class sums [1 × sgemm call]:
     *   class_sums[d, K] = X^T @ Y
     *   cblas_sgemm(Trans, NoTrans, d, K, n, 1.0, X, d, Y, K, 0.0, class_sums, K)
     *   class_sums[j, k] = Σ_{i:y_i=k} X[i,j]
     *
     * Step 4 — Element-wise X² (O(n·d) PHP write pass):
     *   X_sq[i,j] = X[i,j]²
     *
     * Step 5 — Class sum-of-squares [1 × sgemm call]:
     *   class_sumSq[d, K] = X_sq^T @ Y
     *   class_sumSq[j, k] = Σ_{i:y_i=k} X[i,j]²
     *
     * Step 6 — F-statistics [O(d·K) PHP loop]:
     *   class_mean[j,k] = class_sums[j,k] / n_k
     *   SS_B[j] = Σ_k n_k (class_mean[j,k] − grand_means[j])²
     *   SS_W[j] = Σ_k (class_sumSq[j,k] − n_k · class_mean[j,k]²)
     *   F[j] = (SS_B[j] / (K−1)) / (SS_W[j] / (n−K))
     *
     * @return float[]  F-statistics, length d.  Features with no variance get F=0.
     */
    private function fClassif(Tensor $X, Tensor $y, int $n, int $d): array
    {
        $blas = BlasEngine::get()->ffi;

        // ── Discover classes ───────────────────────────────────────────────
        $seen = [];
        for ($i = 0; $i < $n; $i++) {
            $seen[(int) round((float) $y->buffer[$i])] = true;
        }
        ksort($seen);
        $classes  = array_keys($seen);
        $K        = count($classes);
        $classPos = array_flip($classes);

        if ($K < 2) {
            throw new \RuntimeException(
                'SelectKBest / f_classif: y must contain at least 2 distinct class labels.'
            );
        }

        // ── Class sizes ────────────────────────────────────────────────────
        $nk = array_fill(0, $K, 0);
        for ($i = 0; $i < $n; $i++) {
            $nk[$classPos[(int) round((float) $y->buffer[$i])]]++;
        }

        // ── Step 1: Grand means via sgemv ─────────────────────────────────
        //
        // grand_means[j] = (1/n) Σ_i X[i,j]
        // sgemv(Trans, n, d, 1/n, X, d, ones_n, 1, 0, grand_means, 1)
        $grandMeans = new Tensor([$d]);
        $onesN      = Tensor::ones([$n]);
        $blas->cblas_sgemv(
            101,      // CblasRowMajor
            112,      // CblasTrans
            $n, $d,
            1.0 / $n,
            $X->buffer, $d,
            $onesN->buffer, 1,
            0.0,
            $grandMeans->buffer, 1
        );

        // ── Step 2: Class indicator matrix Y [n, K] ────────────────────────
        //
        // Y[i, k] = 1.0 if y[i] == classes[k] else 0.0
        $Y = Tensor::zeros([$n, $K]);
        for ($i = 0; $i < $n; $i++) {
            $Y->buffer[$i * $K + $classPos[(int) round((float) $y->buffer[$i])]] = 1.0;
        }

        // ── Step 3: Class sums [d, K] = X^T @ Y ────────────────────────────
        //
        // class_sums[j, k] = Σ_{i: y_i=k} X[i,j]
        // sgemm(Trans, NoTrans, d, K, n, 1.0, X, d, Y, K, 0.0, class_sums, K)
        $classSums = new Tensor([$d, $K]);
        $blas->cblas_sgemm(
            101,      // CblasRowMajor
            112,      // CblasTrans   (X → X^T)
            111,      // CblasNoTrans (Y as-is)
            $d, $K, $n,
            1.0,
            $X->buffer, $d,
            $Y->buffer, $K,
            0.0,
            $classSums->buffer, $K
        );

        // ── Step 4: Build X_sq (element-wise square of X) ─────────────────
        //
        // This is an O(n·d) PHP write pass — unavoidable for masked sum-of-
        // squares, but kept to a single flat-buffer loop with no PHP overhead
        // beyond the arithmetic.
        $Xsq   = $X->clone();
        $total = $n * $d;
        for ($idx = 0; $idx < $total; $idx++) {
            $v              = (float) $Xsq->buffer[$idx];
            $Xsq->buffer[$idx] = $v * $v;
        }

        // ── Step 5: Class sum-of-squares [d, K] = X_sq^T @ Y ──────────────
        //
        // class_sumSq[j, k] = Σ_{i: y_i=k} X[i,j]²
        $classSumSq = new Tensor([$d, $K]);
        $blas->cblas_sgemm(
            101, 112, 111,
            $d, $K, $n,
            1.0,
            $Xsq->buffer, $d,
            $Y->buffer, $K,
            0.0,
            $classSumSq->buffer, $K
        );

        unset($Xsq, $Y, $onesN);  // free temporaries

        // ── Step 6: F-statistics ───────────────────────────────────────────
        //
        // For each feature j, across K classes:
        //   class_mean[j,k] = class_sums[j,k] / n_k
        //   SS_B[j]         = Σ_k n_k * (class_mean[j,k] − grand_means[j])²
        //   SS_W[j]         = Σ_k (class_sumSq[j,k] − n_k * class_mean[j,k]²)
        //   F[j]            = (SS_B / (K−1)) / (SS_W / (n−K))
        $scores = array_fill(0, $d, 0.0);
        $dof1   = $K - 1;
        $dof2   = $n - $K;

        for ($j = 0; $j < $d; $j++) {
            $gm  = (float) $grandMeans->buffer[$j];
            $ssB = 0.0;
            $ssW = 0.0;

            for ($k = 0; $k < $K; $k++) {
                $nkk = $nk[$k];
                if ($nkk === 0) {
                    continue;
                }
                $base = $j * $K + $k;

                $mu_kj = (float) $classSums->buffer[$base] / $nkk;
                $diff  = $mu_kj - $gm;
                $ssB  += $nkk * $diff * $diff;

                // SS_within_k = Σ x² − n_k μ²
                $ssW += (float) $classSumSq->buffer[$base] - $nkk * $mu_kj * $mu_kj;
            }

            // Guard against degenerate cases (constant feature or one class)
            if ($ssW > 1e-14 && $dof1 > 0 && $dof2 > 0) {
                $scores[$j] = ($ssB / $dof1) / ($ssW / $dof2);
            }
        }

        return $scores;
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private function checkFitted(): void
    {
        if (!isset($this->scores_)) {
            throw new \RuntimeException(
                'SelectKBest is not fitted. Call fit() first.'
            );
        }
    }
}
