<?php

declare(strict_types=1);

namespace Pml\Classic\FeatureSelection;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Transformer};

// ═══════════════════════════════════════════════════════════════════════════
//  VarianceThreshold — sklearn.feature_selection.VarianceThreshold
//
//  Removes all features whose training-set variance falls below a given
//  threshold.  A threshold of 0.0 (the default) removes constant features
//  (variance exactly zero), mirroring sklearn's default behaviour.
//
//  ── Variance Computation ─────────────────────────────────────────────────
//
//  For each feature column j, the biased population variance is computed:
//
//    Var(j) = E[X_j²] − E[X_j]²
//           = (1/n) Σ x_ij² − ((1/n) Σ x_ij)²
//
//  This two-term formula is numerically stable for single-pass computation:
//
//    Pass 1 (single): accumulate sum_j = Σ x_ij and sumSq_j = Σ x_ij²
//    Var(j) = sumSq_j / n  −  (sum_j / n)²
//
//  BLAS acceleration: each column is accessed with stride $d in cblas_sdot
//  (dot-product of column with itself = Σ x_ij²) and cblas_sasum would give
//  absolute sums.  Since we need signed sums, we use a PHP accumulation loop.
//  The column-stride dot trick is used for sumSq only.
//
//  ── Feature Mask ─────────────────────────────────────────────────────────
//
//  After fit(), $this->get_support_ is a bool[] of length n_features.
//  get_support_[j] = true  iff  variances_[j] >= threshold.
//
//  transform() uses the support mask to extract only the kept columns,
//  copying each kept column via cblas_scopy with stride-d source indexing.
// ═══════════════════════════════════════════════════════════════════════════

final class VarianceThreshold implements Estimator, Transformer
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Per-column variances computed from training data.
     * Length = n_features_in_.
     * @var float[]
     */
    public readonly array $variances_;

    /**
     * Boolean support mask: true = keep, false = drop.
     * Mirrors sklearn's get_support() method as a public property.
     * Length = n_features_in_.
     * @var bool[]
     */
    public readonly array $get_support_;

    /** Number of features seen during fit(). */
    public readonly int $n_features_in_;

    /** Indices of kept features (derived from get_support_). */
    private readonly array $keptIndices_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param float $threshold Features with variance strictly below this
     *                         value are removed.  Default 0.0 removes only
     *                         constant (zero-variance) features.
     */
    public function __construct(
        private readonly float $threshold = 0.0,
    ) {}

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Compute per-column variances and build the feature support mask.
     *
     * Single-pass: sum and sum-of-squares are accumulated together so the
     * data is read only once — O(n * d) with minimal PHP overhead.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  Ignored.
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('VarianceThreshold: X must be 2-D [n_samples, n_features].');
        }

        [$n, $d] = $X->shape;
        $this->n_features_in_ = $d;

        $blas = BlasEngine::get()->ffi;

        // ── Accumulate column sums and column sum-of-squares ──────────
        //
        // For sumSq[j] we leverage cblas_sdot(n, col_j, stride_d, col_j, stride_d)
        // which computes Σ_i x_{i,j}² using BLAS level-1 with stride $d —
        // no physical column copy needed.
        //
        // For sum[j] we must use a PHP loop (no signed BLAS sum primitive).
        $sum   = array_fill(0, $d, 0.0);
        $sumSq = array_fill(0, $d, 0.0);

        for ($j = 0; $j < $d; $j++) {
            // Column j pointer: buffer[j], stride d
            $colPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$j]));

            // Σ x_{i,j}²  — BLAS sdot of column with itself
            $sumSq[$j] = (float) $blas->cblas_sdot($n, $colPtr, $d, $colPtr, $d);
        }

        // Signed sum: PHP loop (BLAS sasum returns |sum|, not sum)
        for ($i = 0; $i < $n; $i++) {
            $rowOff = $i * $d;
            for ($j = 0; $j < $d; $j++) {
                $sum[$j] += (float) $X->buffer[$rowOff + $j];
            }
        }

        // ── Variance: Var(j) = E[x²] − E[x]² ────────────────────────
        $variances = [];
        for ($j = 0; $j < $d; $j++) {
            $mean       = $sum[$j] / $n;
            $variances[$j] = $sumSq[$j] / $n - $mean * $mean;
        }
        $this->variances_ = $variances;

        // ── Build support mask ────────────────────────────────────────
        $support = [];
        $kept    = [];
        for ($j = 0; $j < $d; $j++) {
            $keep       = $variances[$j] >= $this->threshold;
            $support[$j] = $keep;
            if ($keep) {
                $kept[] = $j;
            }
        }
        $this->get_support_ = $support;
        $this->keptIndices_ = $kept;

        if (count($kept) === 0) {
            throw new \RuntimeException(
                'VarianceThreshold: no features passed the variance threshold '
                . $this->threshold . '. All features were removed.'
            );
        }

        return $this;
    }

    // ── Transformer ───────────────────────────────────────────────────────

    /**
     * Remove low-variance features according to the fitted support mask.
     *
     * Each surviving column is copied from X into the output tensor using
     * cblas_scopy with incX = $d (column stride in row-major layout) and
     * incY = 1 (contiguous in the output column).  This avoids materialising
     * a physical column buffer — the data is read directly from X's buffer
     * with a non-unit stride.
     *
     * @param Tensor $X  [n_samples, n_features_in]
     * @return Tensor    [n_samples, n_features_out]  (n_features_out ≤ n_features_in)
     */
    public function transform(Tensor $X): Tensor
    {
        if (!isset($this->get_support_)) {
            throw new \RuntimeException('VarianceThreshold is not fitted. Call fit() first.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('VarianceThreshold: X must be 2-D [n_samples, n_features].');
        }
        [$n, $dIn] = $X->shape;
        if ($dIn !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "VarianceThreshold: expected {$this->n_features_in_} features, got {$dIn}."
            );
        }

        $blas   = BlasEngine::get()->ffi;
        $kept   = $this->keptIndices_;
        $dOut   = count($kept);
        $out    = new Tensor([$n, $dOut]);

        // ── Copy each kept column: source stride = dIn, dest stride = dOut ─
        //
        // For column j_src at output position j_dst:
        //   src: X->buffer[j_src], X->buffer[j_src + dIn], ..., X->buffer[j_src + (n-1)*dIn]
        //   dst: out->buffer[j_dst], out->buffer[j_dst + dOut], ..., out->buffer[j_dst + (n-1)*dOut]
        //
        // cblas_scopy(n, src_ptr, incSrc=dIn, dst_ptr, incDst=dOut) performs
        // the strided copy in a single BLAS call — no PHP-level element loop.
        foreach ($kept as $outCol => $srcCol) {
            $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$srcCol]));
            $dstPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$outCol]));
            $blas->cblas_scopy($n, $srcPtr, $dIn, $dstPtr, $dOut);
        }

        return $out;
    }

    /**
     * Convenience: fit(X, y) then transform(X).
     */
    public function fit_transform(Tensor $X, ?Tensor $y = null): Tensor
    {
        return $this->fit($X, $y)->transform($X);
    }

    /**
     * Return the indices of features that pass the threshold.
     * Mirrors sklearn's get_support(indices=True).
     *
     * @return int[]
     */
    public function getSupportIndices(): array
    {
        return $this->keptIndices_;
    }
}
