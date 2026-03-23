<?php

declare(strict_types=1);

namespace Pml\Classic\Preprocess;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Transformer};

// ═══════════════════════════════════════════════════════════════════════════
//  MinMaxScaler — sklearn.preprocessing.MinMaxScaler
//
//  Scales each feature column to a given range (default [0, 1]).
//
//  Formula:
//    scale_[j]   = (r_max − r_min) / max(data_max_[j] − data_min_[j], ε)
//    X_scaled[i,j] = X[i,j] · scale_[j] + (r_min − data_min_[j] · scale_[j])
//
//  The combined shift term is stored as min_[j]:
//    min_[j] = r_min − data_min_[j] · scale_[j]
//
//  so transform reduces to:
//    X_scaled = X · scale_ + min_  (per-column, broadcast over rows)
//
//  BLAS column-wise operations on row-major data [m, n]:
//    Column j occupies elements {j, j+n, j+2n, …} — stride = n_features.
//    cblas_sscal(m, alpha, ptr_to_col_j, stride=n) scales the column.
//    cblas_saxpy(m, alpha, ones[m], 1, ptr_to_col_j, stride=n) shifts it.
//
//  fit() min/max reduction:
//    BLAS has no per-column min/max primitive (cblas_isamax finds the index
//    of the max *absolute* value, not signed min/max).  The PHP loop in
//    fit() is O(m · n) — unavoidable — but it involves only comparisons and
//    no allocation, so it is typically fast enough.
// ═══════════════════════════════════════════════════════════════════════════

final class MinMaxScaler implements Estimator, Transformer
{
    // ── Fitted attributes (sklearn convention: trailing underscore) ────────

    /** @var Tensor  Per-column minimum of training data [n_features] */
    public readonly Tensor $data_min_;

    /** @var Tensor  Per-column maximum of training data [n_features] */
    public readonly Tensor $data_max_;

    /** @var Tensor  Per-column range = data_max_ − data_min_ [n_features] */
    public readonly Tensor $data_range_;

    /**
     * Per-column scale factor:
     *   scale_[j] = (feature_range[1] − feature_range[0]) / max(range[j], ε)
     * @var Tensor  [n_features]
     */
    public readonly Tensor $scale_;

    /**
     * Per-column additive shift after scaling:
     *   min_[j] = feature_range[0] − data_min_[j] · scale_[j]
     * @var Tensor  [n_features]
     */
    public readonly Tensor $min_;

    public readonly int $n_features_in_;
    public readonly int $n_samples_seen_;

    /**
     * @param float[] $feature_range  Target range [min, max].  Default [0, 1].
     * @param float   $eps            Stability term for zero-range features.
     */
    public function __construct(
        private readonly array $feature_range = [0.0, 1.0],
        private readonly float $eps           = 1e-8,
    ) {
        if (count($feature_range) !== 2 || $feature_range[0] >= $feature_range[1]) {
            throw new \InvalidArgumentException(
                'MinMaxScaler: feature_range must be [min, max] with min < max.'
            );
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Compute per-column min and max from training data.
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Ignored (present for API parity).
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('MinMaxScaler::fit() requires a 2-D tensor X.');
        }

        [$m, $n] = $X->shape;

        $dataMin   = new Tensor([$n]);
        $dataMax   = new Tensor([$n]);

        // ── Initialise with the first row ─────────────────────────────────
        for ($j = 0; $j < $n; $j++) {
            $v               = (float) $X->buffer[$j]; // row 0, col j
            $dataMin->buffer[$j] = $v;
            $dataMax->buffer[$j] = $v;
        }

        // ── Scan remaining rows ───────────────────────────────────────────
        //
        // BLAS cblas_isamax finds |max|, not signed min/max — unusable here.
        // A PHP comparison loop is the only option; it is purely arithmetic
        // (no allocation inside) and O(m·n) total.
        for ($i = 1; $i < $m; $i++) {
            $off = $i * $n;
            for ($j = 0; $j < $n; $j++) {
                $v = (float) $X->buffer[$off + $j];
                if ($v < (float) $dataMin->buffer[$j]) { $dataMin->buffer[$j] = $v; }
                if ($v > (float) $dataMax->buffer[$j]) { $dataMax->buffer[$j] = $v; }
            }
        }

        // ── Derived quantities ────────────────────────────────────────────
        [$rMin, $rMax] = $this->feature_range;
        $rWidth = $rMax - $rMin;

        $dataRange = new Tensor([$n]);
        $scale     = new Tensor([$n]);
        $min       = new Tensor([$n]);

        for ($j = 0; $j < $n; $j++) {
            $dMin = (float) $dataMin->buffer[$j];
            $dMax = (float) $dataMax->buffer[$j];
            $r    = $dMax - $dMin;

            $dataRange->buffer[$j] = $r;

            $s = $rWidth / max($r, $this->eps);   // scale_[j]
            $scale->buffer[$j] = $s;

            // min_[j] is the additive offset after scaling:
            //   X_scaled = X * scale_ + (r_min − data_min_ * scale_)
            $min->buffer[$j] = $rMin - $dMin * $s;
        }

        $this->data_min_      = $dataMin;
        $this->data_max_      = $dataMax;
        $this->data_range_    = $dataRange;
        $this->scale_         = $scale;
        $this->min_           = $min;
        $this->n_features_in_ = $n;
        $this->n_samples_seen_ = $m;

        return $this;
    }

    // ── Transformer ────────────────────────────────────────────────────────

    /**
     * Scale features of X according to feature_range.
     *
     * Per-column transform (BLAS):
     *   1. Copy X → out                                    (cblas_scopy)
     *   2. col_j *= scale_[j]   with stride = n_features   (cblas_sscal)
     *   3. col_j += min_[j]     with stride = n_features   (cblas_saxpy on ones)
     *
     * @param Tensor $X  Feature matrix [n_samples, n_features]
     * @return Tensor    Scaled matrix  [n_samples, n_features]
     */
    public function transform(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "MinMaxScaler::transform() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        [$m, $n] = $X->shape;
        $blas    = BlasEngine::get()->ffi;

        // Full copy of X into out — preserves original
        $out = new Tensor([$m, $n]);
        $blas->cblas_scopy($m * $n, $X->buffer, 1, $out->buffer, 1);

        // Ones column of length m — shared across all column operations
        $ones = Tensor::ones([$m]);

        for ($j = 0; $j < $n; $j++) {
            // Pointer to element [0, j] — stride = n walks down column j
            $colPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$j]));

            // Step 2: col_j *= scale_[j]
            $blas->cblas_sscal($m, (float) $this->scale_->buffer[$j], $colPtr, $n);

            // Step 3: col_j += min_[j] · ones[m]
            $shift = (float) $this->min_->buffer[$j];
            if ($shift !== 0.0) {
                $blas->cblas_saxpy($m, $shift, $ones->buffer, 1, $colPtr, $n);
            }
        }

        return $out;
    }

    /**
     * Undo the MinMax scaling: X_orig = (X_scaled − min_[j]) / scale_[j]
     *
     * @param Tensor $X  Scaled matrix [n_samples, n_features]
     * @return Tensor    Original-scale matrix
     */
    public function inverse_transform(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "MinMaxScaler::inverse_transform() expected [*, {$this->n_features_in_}]."
            );
        }

        [$m, $n] = $X->shape;
        $blas    = BlasEngine::get()->ffi;

        $out  = new Tensor([$m, $n]);
        $blas->cblas_scopy($m * $n, $X->buffer, 1, $out->buffer, 1);
        $ones = Tensor::ones([$m]);

        for ($j = 0; $j < $n; $j++) {
            $colPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$j]));

            // Step 1: col_j -= min_[j]
            $shift = (float) $this->min_->buffer[$j];
            if ($shift !== 0.0) {
                $blas->cblas_saxpy($m, -$shift, $ones->buffer, 1, $colPtr, $n);
            }

            // Step 2: col_j /= scale_[j]
            $s = (float) $this->scale_->buffer[$j];
            if ($s !== 0.0) {
                $blas->cblas_sscal($m, 1.0 / $s, $colPtr, $n);
            }
        }

        return $out;
    }

    /** Fit then immediately transform — avoids a redundant copy. */
    public function fit_transform(Tensor $X, ?Tensor $y = null): Tensor
    {
        return $this->fit($X, $y)->transform($X);
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    private function checkFitted(): void
    {
        if (!isset($this->scale_)) {
            throw new \RuntimeException('MinMaxScaler is not fitted. Call fit() first.');
        }
    }
}
