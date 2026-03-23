<?php

declare(strict_types=1);

namespace Pml\Classic\LinearModel;

use Pml\{Tensor, BlasEngine, Ops};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  LinearRegression — sklearn.linear_model.LinearRegression
//
//  Ordinary Least Squares (OLS) solved exactly via LAPACKE_sgels.
//
//  Algorithm:
//    Augment X with a column of 1s:  X_aug [m, n+1]
//    Solve:  min ||X_aug · θ − y||₂  →  θ = [coef_, intercept_]
//
//  LAPACKE_sgels uses QR factorisation (or LQ for under-determined systems)
//  to find the minimum-norm least-squares solution in a single LAPACK call.
//  This is exact (not iterative) and avoids forming X^T X (which would
//  square the condition number).
//
//  After fitting:
//    coef_      [n_features]   regression coefficients
//    intercept_ scalar         bias term
//
//  predict():
//    ŷ = X @ coef_ + intercept_   via cblas_sgemv + scalar broadcast.
// ═══════════════════════════════════════════════════════════════════════════

final class LinearRegression implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────
    /** @var Tensor  Regression coefficients [n_features] */
    public readonly Tensor $coef_;

    /** @var float   Intercept (bias) term */
    public readonly float $intercept_;

    public readonly int $n_features_in_;

    public function __construct(
        private readonly bool $fit_intercept = true,
    ) {}

    // ── Estimator ─────────────────────────────────────────────────────────

    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('LinearRegression::fit() requires target tensor $y.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('LinearRegression::fit() requires a 2D feature matrix X.');
        }

        [$m, $n] = $X->shape;
        $blas    = BlasEngine::get()->ffi;

        if ($this->fit_intercept) {
            // ── Build augmented matrix X_aug [m, n+1] with ones in last col ─
            //
            //  We set the last column to 1.0 so the least-squares solution
            //  automatically yields coef_[0..n-1] and coef_[n] = intercept_.
            //
            //  Memory layout: X_aug is contiguous row-major [m, n+1].
            //  We copy each row of X then set element [i, n] = 1.0.
            $nAug = $n + 1;
            $Xaug = new Tensor([$m, $nAug]);

            for ($i = 0; $i < $m; $i++) {
                // Copy n features from X row i → X_aug row i (first n cols)
                $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$i * $n]));
                $dstPtr = \FFI::cast('float*', \FFI::addr($Xaug->buffer[$i * $nAug]));
                $blas->cblas_scopy($n, $srcPtr, 1, $dstPtr, 1);

                // Set augmented bias column to 1.0
                $Xaug->buffer[$i * $nAug + $n] = 1.0;
            }
        } else {
            $nAug = $n;
            $Xaug = $X;
        }

        // ── Solve min ||X_aug · θ − y||₂ via Ops::lstsq ──────────────────
        //
        //  Ops::lstsq clones both inputs (LAPACK overwrites them) and calls
        //  LAPACKE_sgels(RowMajor, 'N', m, nAug, 1, A, nAug, b, 1).
        //
        //  For an overdetermined system (m > nAug, typical for ML):
        //    The solution θ[nAug] is written into b[0..nAug-1].
        //    Residuals occupy b[nAug..m-1] (ignored).
        //
        //  y must be at least [m] elements; the solution lands in [0..nAug).
        $theta = Ops::lstsq($Xaug, $y);

        // ── Extract coef_ and intercept_ from the solution buffer ─────────
        $coef = new Tensor([$n]);
        $blas->cblas_scopy($n, $theta->buffer, 1, $coef->buffer, 1);

        $this->coef_          = $coef;
        $this->intercept_     = $this->fit_intercept ? (float) $theta->buffer[$n] : 0.0;
        $this->n_features_in_ = $n;

        return $this;
    }

    // ── Predictor ─────────────────────────────────────────────────────────

    public function predict(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "LinearRegression::predict() expected [*, {$this->n_features_in_}], "
                . "got [" . implode(', ', $X->shape) . "]."
            );
        }

        [$m, $n] = $X->shape;
        $blas    = BlasEngine::get()->ffi;
        $out     = new Tensor([$m]);

        // ── ŷ = X @ coef_ + intercept_ via sgemv ─────────────────────────
        //
        //  sgemv(RowMajor, NoTrans, m, n, 1.0, X, n, coef_, 1, 0.0, out, 1)
        //  computes: out[i] = Σ_j X[i,j] · coef_[j]
        //
        //  Then we add the scalar intercept via a PHP loop (O(m), BLAS has
        //  no broadcast-scalar-add primitive).
        $blas->cblas_sgemv(
            101,   // CblasRowMajor
            111,   // CblasNoTrans
            $m, $n,
            1.0,
            $X->buffer, $n,
            $this->coef_->buffer, 1,
            0.0,
            $out->buffer, 1
        );

        // Add intercept scalar to all predictions
        if ($this->intercept_ !== 0.0) {
            for ($i = 0; $i < $m; $i++) {
                $out->buffer[$i] += $this->intercept_;
            }
        }

        return $out;
    }

    // ── Metrics helpers ───────────────────────────────────────────────────

    /**
     * Coefficient of determination R² = 1 − SS_res / SS_tot
     * Mirrors sklearn's LinearRegression.score().
     */
    public function score(Tensor $X, Tensor $y): float
    {
        $yPred = $this->predict($X);
        $m     = $y->size;
        $blas  = BlasEngine::get()->ffi;

        // Mean of y
        $yMean = 0.0;
        for ($i = 0; $i < $m; $i++) {
            $yMean += (float) $y->buffer[$i];
        }
        $yMean /= $m;

        $ssTot = 0.0;
        $ssRes = 0.0;
        for ($i = 0; $i < $m; $i++) {
            $yi    = (float) $y->buffer[$i];
            $ypi   = (float) $yPred->buffer[$i];
            $ssTot += ($yi - $yMean) ** 2;
            $ssRes += ($yi - $ypi) ** 2;
        }

        return $ssTot > 0.0 ? 1.0 - $ssRes / $ssTot : 1.0;
    }

    private function checkFitted(): void
    {
        if (!isset($this->coef_)) {
            throw new \RuntimeException('LinearRegression is not fitted. Call fit() first.');
        }
    }
}
