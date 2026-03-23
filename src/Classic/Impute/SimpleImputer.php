<?php

declare(strict_types=1);

namespace Pml\Classic\Impute;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Transformer};

// ═══════════════════════════════════════════════════════════════════════════
//  SimpleImputer — sklearn.impute.SimpleImputer
//
//  Replaces missing values (IEEE 754 NaN) in a feature matrix with
//  per-column statistics learned from the training set.
//
//  ── Supported Strategies ─────────────────────────────────────────────────
//
//  'mean'     Replace NaN with the per-column mean computed over non-NaN
//             training samples.  Uses two PHP passes:
//               Pass 1: accumulate column sums and non-NaN counts.
//               Pass 2 (transform): replace NaN elements in-place.
//
//  'median'   Replace NaN with the per-column median computed over non-NaN
//             training samples.  Collects non-NaN values per column, sorts
//             them in PHP, then takes the middle value (or average of the
//             two middle values for even-count columns).
//             Robust to outliers — preferred for skewed distributions.
//
//  'constant' Replace NaN with a fixed scalar $fill_value (default 0.0).
//             fit() stores fill_value for every column; transform() applies it.
//
//  ── Statistics Storage ───────────────────────────────────────────────────
//
//  After fit(), $this->statistics_ is a PHP float[] of length n_features.
//  For 'mean':    statistics_[j] = mean of column j (NaN-ignored).
//  For 'constant': statistics_[j] = fill_value  for all j.
//
//  ── NaN Detection ────────────────────────────────────────────────────────
//
//  PHP's is_nan() is used on each cast-to-float element.  FFI buffer values
//  that are IEEE 754 NaN satisfy is_nan((float)$buf[$i]).
//
//  ── BLAS Usage ───────────────────────────────────────────────────────────
//
//  cblas_scopy copies entire rows when building the output tensor in
//  transform(), avoiding per-element PHP loops over non-NaN values.
//  The NaN replacement is then a targeted PHP loop over only those
//  positions that actually contain NaN (typically rare).
// ═══════════════════════════════════════════════════════════════════════════

final class SimpleImputer implements Estimator, Transformer
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Per-column fill values learned in fit().
     * Length = n_features.  Mirrors sklearn's statistics_ attribute.
     *
     * @var float[]
     */
    public readonly array $statistics_;

    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param string $strategy   Imputation strategy: 'mean', 'median', or 'constant'.
     * @param float  $fill_value Fill value for strategy='constant'.  Ignored
     *                           for 'mean' and 'median'.  Default = 0.0.
     */
    public function __construct(
        private readonly string $strategy   = 'mean',
        private readonly float  $fill_value = 0.0,
    ) {
        if (!in_array($strategy, ['mean', 'median', 'constant'], true)) {
            throw new \InvalidArgumentException(
                "SimpleImputer: unknown strategy '{$strategy}'. Use 'mean', 'median', or 'constant'."
            );
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Compute per-column fill values from the training data.
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Ignored (unsupervised transformer).
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('SimpleImputer: X must be 2-D [n_samples, n_features].');
        }

        [$n, $d] = $X->shape;
        $this->n_features_in_ = $d;

        if ($this->strategy === 'constant') {
            // ── Constant strategy: every column gets fill_value ─────────
            $this->statistics_ = array_fill(0, $d, $this->fill_value);
            return $this;
        }

        if ($this->strategy === 'median') {
            // ── Median strategy: sort non-NaN column values ────────────
            //
            // For each column j, collect all non-NaN floats, sort them,
            // then take the middle value (or average of the two middle
            // values for even-length arrays).  The median is robust to
            // outliers and well-suited for skewed numeric features like
            // Age or Fare in tabular datasets.
            $stats = [];
            for ($j = 0; $j < $d; $j++) {
                $colVals = [];
                for ($i = 0; $i < $n; $i++) {
                    $v = (float) $X->buffer[$i * $d + $j];
                    if (!is_nan($v)) {
                        $colVals[] = $v;
                    }
                }
                sort($colVals);
                $cnt = count($colVals);
                if ($cnt === 0) {
                    $stats[$j] = 0.0;
                } elseif ($cnt % 2 === 1) {
                    $stats[$j] = $colVals[intdiv($cnt, 2)];
                } else {
                    $stats[$j] = ($colVals[$cnt / 2 - 1] + $colVals[$cnt / 2]) / 2.0;
                }
            }
            $this->statistics_ = $stats;
            return $this;
        }

        // ── Mean strategy: two-pass column statistics ──────────────────
        //
        // Pass 1: For each column j, accumulate the sum and count of
        //         non-NaN values.  PHP's is_nan() is the correct guard
        //         because float equality comparisons with NaN always fail.
        $sums   = array_fill(0, $d, 0.0);
        $counts = array_fill(0, $d, 0);

        for ($i = 0; $i < $n; $i++) {
            $rowOff = $i * $d;
            for ($j = 0; $j < $d; $j++) {
                $v = (float) $X->buffer[$rowOff + $j];
                if (!is_nan($v)) {
                    $sums[$j]   += $v;
                    $counts[$j] += 1;
                }
            }
        }

        // Compute per-column means; fallback to 0.0 if column is all-NaN
        $stats = [];
        for ($j = 0; $j < $d; $j++) {
            $stats[$j] = ($counts[$j] > 0)
                ? $sums[$j] / $counts[$j]
                : 0.0;
        }
        $this->statistics_ = $stats;

        return $this;
    }

    // ── Transformer ───────────────────────────────────────────────────────

    /**
     * Replace NaN values in X with the learned per-column statistics.
     *
     * Strategy: Copy the full input buffer row by row (via cblas_scopy),
     * then scan for NaN positions and replace them with statistics_[j].
     * This amortises the PHP loop overhead — only NaN elements pay the
     * assignment cost; non-NaN values are copied at BLAS throughput.
     *
     * @param Tensor $X  Feature matrix [n_samples, n_features]
     * @return Tensor    Imputed matrix [n_samples, n_features] (new allocation)
     */
    public function transform(Tensor $X): Tensor
    {
        if (!isset($this->statistics_)) {
            throw new \RuntimeException('SimpleImputer is not fitted. Call fit() first.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('SimpleImputer: X must be 2-D [n_samples, n_features].');
        }
        [$n, $d] = $X->shape;
        if ($d !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "SimpleImputer: expected {$this->n_features_in_} features, got {$d}."
            );
        }

        $blas = BlasEngine::get()->ffi;
        $out  = new Tensor([$n, $d]);

        // ── Step 1: bulk-copy X into output via cblas_scopy ───────────
        $blas->cblas_scopy($n * $d, $X->buffer, 1, $out->buffer, 1);

        // ── Step 2: scan for NaN and replace with per-column statistic ─
        //
        // We iterate in row-major order; for each NaN element at column j
        // we write statistics_[j] into the output buffer.  Non-NaN elements
        // were already copied correctly in Step 1.
        for ($i = 0; $i < $n; $i++) {
            $rowOff = $i * $d;
            for ($j = 0; $j < $d; $j++) {
                $v = (float) $out->buffer[$rowOff + $j];
                if (is_nan($v)) {
                    $out->buffer[$rowOff + $j] = $this->statistics_[$j];
                }
            }
        }

        return $out;
    }

    /**
     * Convenience: fit(X) then transform(X).
     */
    public function fit_transform(Tensor $X, ?Tensor $y = null): Tensor
    {
        return $this->fit($X, $y)->transform($X);
    }
}
