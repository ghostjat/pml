<?php

declare(strict_types=1);

namespace Pml\Classic\Preprocess;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Transformer};

// ═══════════════════════════════════════════════════════════════════════════
//  StandardScaler — sklearn.preprocessing.StandardScaler
//
//  Standardises each feature to zero mean and unit variance:
//
//    z = (x − μ) / σ
//
//  where μ[j] and σ[j] are the per-column mean and standard deviation
//  computed over the training set.
//
//  BLAS strategy:
//    fit()      — column means via one cblas_sgemv call (X^T * ones / m);
//                 column variances via per-column cblas_sdot with stride n
//                 (treats column-major slice of a row-major matrix).
//    transform() — row-wise mean subtraction via per-row cblas_saxpy;
//                  column-wise standard-deviation division via cblas_sscal
//                  with stride n (scales every m-th element = one column).
//
//  No PHP loops touch the heavy float math — all O(m·n) work is in C.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @property-read Tensor $mean_   Per-feature means     [n_features]
 * @property-read Tensor $scale_  Per-feature std-devs  [n_features]
 * @property-read int    $n_features_in_
 * @property-read int    $n_samples_seen_
 */
final class StandardScaler implements Estimator, Transformer
{
    // ── Fitted attributes (sklearn naming convention: trailing underscore) ──
    public readonly Tensor $mean_;
    public readonly Tensor $scale_;
    public readonly int    $n_features_in_;
    public readonly int    $n_samples_seen_;

    /**
     * @param bool  $with_mean  Subtract the mean (default true, like sklearn)
     * @param bool  $with_std   Divide by std-dev  (default true, like sklearn)
     * @param float $eps        Floor for std-dev to avoid division by zero
     */
    public function __construct(
        private readonly bool  $with_mean = true,
        private readonly bool  $with_std  = true,
        private readonly float $eps       = 1e-8,
    ) {}

    // ── Estimator ─────────────────────────────────────────────────────────

    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('StandardScaler::fit() requires a 2D tensor [n_samples, n_features].');
        }

        [$m, $n] = $X->shape;
        $blas    = BlasEngine::get()->ffi;

        // ── Column means via sgemv ─────────────────────────────────────────
        //
        //  We want:  mean[j] = (1/m) * Σ_i X[i,j]  for each feature j.
        //
        //  BLAS Level-2 sgemv with Trans flag computes:
        //    y = alpha * A^T * x + beta * y
        //  Setting A = X [m,n], x = ones[m], alpha = 1/m, beta = 0:
        //    mean = (1/m) * X^T * ones  →  mean[j] = column j sum / m  ✓
        //
        //  This is a single C call for all n column means — no PHP loop.
        $mean  = new Tensor([$n]);
        $ones  = Tensor::ones([$m]);
        $blas->cblas_sgemv(
            101,          // CblasRowMajor
            112,          // CblasTrans — X^T acts on the ones vector
            $m, $n,
            1.0 / $m,     // alpha = 1/m → divides while summing
            $X->buffer, $n,
            $ones->buffer, 1,
            0.0,           // beta = 0 → overwrite mean
            $mean->buffer, 1
        );

        // ── Column variances via sdot with stride n ────────────────────────
        //
        //  After centering X_c = X − mean, the variance of column j is:
        //    var[j] = (1/m) * Σ_i X_c[i,j]²
        //           = (1/m) * sdot( m, X_c[0,j], stride=n,
        //                              X_c[0,j], stride=n )
        //
        //  Passing stride = n makes BLAS step through the row-major buffer
        //  by n elements at a time, landing on the same column j in each row.
        //  This is the canonical BLAS trick for column-wise operations on
        //  row-major matrices — no transposition or data copy needed.
        $scale = new Tensor([$n]);

        if ($this->with_std) {
            // Build centered copy in C memory (O(m·n) via saxpy, one row at a time)
            $Xc = $X->clone();
            for ($i = 0; $i < $m; $i++) {
                // saxpy: Xc[i,:] += -1.0 * mean   →  Xc[i,:] = X[i,:] − mean
                $rowPtr = \FFI::cast('float*', \FFI::addr($Xc->buffer[$i * $n]));
                $blas->cblas_saxpy($n, -1.0, $mean->buffer, 1, $rowPtr, 1);
            }

            for ($j = 0; $j < $n; $j++) {
                // Pointer to X_c[0, j] — first element of column j
                $colPtr = \FFI::cast('float*', \FFI::addr($Xc->buffer[$j]));

                // sdot(m, col_j, stride=n, col_j, stride=n) = Σ X_c[i,j]²
                $dotVal  = (float) $blas->cblas_sdot($m, $colPtr, $n, $colPtr, $n);
                $varJ    = $dotVal / $m;
                $scale->buffer[$j] = max(sqrt($varJ), $this->eps);
            }
        } else {
            // with_std=false: scale_ = 1.0 (no scaling, sklearn behaviour)
            $bytes = pack('f*', ...array_fill(0, $n, 1.0));
            \FFI::memcpy($scale->buffer, $bytes, $n * 4);
        }

        $this->mean_           = $mean;
        $this->scale_          = $scale;
        $this->n_features_in_  = $n;
        $this->n_samples_seen_ = $m;

        return $this;
    }

    // ── Transformer ───────────────────────────────────────────────────────

    public function transform(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "StandardScaler::transform() expected [*, {$this->n_features_in_}], "
                . "got [" . implode(', ', $X->shape) . "]."
            );
        }

        [$m, $n] = $X->shape;
        $blas    = BlasEngine::get()->ffi;
        $out     = $X->clone();

        if ($this->with_mean) {
            // ── Subtract column mean, row by row ──────────────────────────
            //
            //  saxpy per row: out[i,:] += -1.0 * mean_
            //  O(m) FFI calls, each doing O(n) BLAS work.
            for ($i = 0; $i < $m; $i++) {
                $rowPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$i * $n]));
                $blas->cblas_saxpy($n, -1.0, $this->mean_->buffer, 1, $rowPtr, 1);
            }
        }

        if ($this->with_std) {
            // ── Divide each column by its std-dev ─────────────────────────
            //
            //  sscal with stride n: scales every n-th element starting at
            //  out[0, j], landing on out[1,j], out[2,j], ... — i.e., the
            //  entire j-th column.  One C call per feature.
            for ($j = 0; $j < $n; $j++) {
                $colPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$j]));
                $invStd = 1.0 / (float)$this->scale_->buffer[$j];
                $blas->cblas_sscal($m, $invStd, $colPtr, $n);
            }
        }

        return $out;
    }

    public function fit_transform(Tensor $X, ?Tensor $y = null): Tensor
    {
        return $this->fit($X, $y)->transform($X);
    }

    // ── Inverse transform ─────────────────────────────────────────────────

    /**
     * Reverse the standardisation: x = z * σ + μ
     * Mirrors sklearn's StandardScaler.inverse_transform().
     */
    public function inverse_transform(Tensor $X): Tensor
    {
        $this->checkFitted();

        [$m, $n] = $X->shape;
        $blas    = BlasEngine::get()->ffi;
        $out     = $X->clone();

        // Multiply each column by its std-dev (undo division)
        if ($this->with_std) {
            for ($j = 0; $j < $n; $j++) {
                $colPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$j]));
                $blas->cblas_sscal($m, (float)$this->scale_->buffer[$j], $colPtr, $n);
            }
        }

        // Add back the mean (undo subtraction)
        if ($this->with_mean) {
            for ($i = 0; $i < $m; $i++) {
                $rowPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$i * $n]));
                $blas->cblas_saxpy($n, 1.0, $this->mean_->buffer, 1, $rowPtr, 1);
            }
        }

        return $out;
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private function checkFitted(): void
    {
        if (!isset($this->mean_)) {
            throw new \RuntimeException(
                'StandardScaler is not fitted yet. Call fit() before transform().'
            );
        }
    }
}
