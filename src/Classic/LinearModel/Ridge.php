<?php

declare(strict_types=1);

namespace Pml\Classic\LinearModel;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  Ridge — sklearn.linear_model.Ridge
//
//  Ridge regression (L2 regularization) solved via the normal equations
//  with Cholesky factorisation.
//
//  ── Normal Equations ─────────────────────────────────────────────────────
//
//  OLS solves:  min_θ ||X θ − y||²
//  Ridge adds an L2 penalty: min_θ ||X θ − y||² + α ||θ||²
//
//  Setting the gradient to zero gives the Ridge normal equations:
//
//    (X^T X + α I) θ = X^T y
//    └──────────────┘         (= A, symmetric positive-definite for α > 0)
//
//  Compared to OLS (via LAPACKE_sgels / QR):
//    - OLS avoids forming X^T X (which squares the condition number).
//    - Ridge MUST add α to the diagonal, so forming X^T X is required.
//    - For moderate n_features, the condition number is bounded by
//      (σ_max² + α) / (σ_min² + α), which α controls.
//    - For large α, the system is very well-conditioned and Cholesky is
//      extremely stable.
//
//  ── BLAS / LAPACKE Steps ──────────────────────────────────────────────────
//
//  With X_aug [m, n+1] (bias column of 1s appended):
//
//    1. A  = X_aug^T @ X_aug   [n+1, n+1]  — one cblas_sgemm call
//    2. A[j,j] += α  for j < n             — PHP loop (O(n)), bias not penalised
//    3. b  = X_aug^T @ y       [n+1]       — one cblas_sgemv (Trans) call
//    4. Solve  A θ = b         [n+1]       — LAPACKE_sposv (Cholesky)
//
//  LAPACKE_sposv (Cholesky for SPD matrices):
//    Guaranteed to succeed because A = X^T X + αI is always SPD for α > 0.
//    It is faster and more numerically stable than LAPACKE_sgesv (LU) for
//    symmetric problems.
//    Already declared in BlasEngine::LAPACKE_HEADER and available via
//    BlasEngine::get()->lapacke — no new FFI binding is required.
//
//  ── Intercept ────────────────────────────────────────────────────────────
//
//  Like LinearRegression, the intercept is absorbed into the last column of
//  the augmented matrix.  The diagonal penalty is NOT applied to the bias
//  term (column n+1), matching sklearn's fit_intercept=True behaviour.
//
//  predict():
//    ŷ = X @ coef_ + intercept_   via cblas_sgemv + scalar PHP loop.
//
//  score():
//    R² = 1 − SS_res / SS_tot     (RegressorMixin.score() equivalent).
// ═══════════════════════════════════════════════════════════════════════════

final class Ridge implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** @var Tensor  Regression coefficients [n_features] */
    public readonly Tensor $coef_;

    /** @var float   Intercept (bias) term */
    public readonly float $intercept_;

    public readonly int $n_features_in_;

    /**
     * @param float $alpha         Regularisation strength λ.  Must be > 0.
     *                             Larger α → heavier shrinkage toward zero.
     * @param bool  $fit_intercept Whether to fit a bias term (recommended).
     */
    public function __construct(
        private readonly float $alpha         = 1.0,
        private readonly bool  $fit_intercept = true,
    ) {
        if ($alpha <= 0.0) {
            throw new \InvalidArgumentException("Ridge: alpha must be > 0, got {$alpha}.");
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Fit Ridge via the normal equations: (X^T X + α I) θ = X^T y
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Target vector  [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('Ridge::fit() requires target tensor $y.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('Ridge::fit() requires a 2-D feature matrix X.');
        }

        [$m, $n] = $X->shape;
        $blas    = BlasEngine::get()->ffi;
        $lapacke = BlasEngine::get()->lapacke;

        // ── Step 0: Build augmented matrix X_aug [m, n+1] ─────────────────
        //
        // The last column is all 1.0 — its coefficient will become intercept_.
        // Bias column is never penalised (alpha is added only to the first n
        // diagonal entries of X^T X + αI), matching sklearn's convention.
        if ($this->fit_intercept) {
            $nAug = $n + 1;
            $Xaug = new Tensor([$m, $nAug]);

            for ($i = 0; $i < $m; $i++) {
                $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$i * $n]));
                $dstPtr = \FFI::cast('float*', \FFI::addr($Xaug->buffer[$i * $nAug]));
                $blas->cblas_scopy($n, $srcPtr, 1, $dstPtr, 1);
                $Xaug->buffer[$i * $nAug + $n] = 1.0; // bias column
            }
        } else {
            $nAug = $n;
            $Xaug = $X;
        }

        // ── Step 1: A = X_aug^T @ X_aug  [nAug, nAug] ────────────────────
        //
        //  sgemm(RowMajor, Trans, NoTrans, nAug, nAug, m,
        //        1.0, X_aug[m, nAug], nAug,
        //             X_aug[m, nAug], nAug,
        //        0.0, A[nAug, nAug],  nAug)
        //
        //  The result is a symmetric PSD matrix; with α > 0 added to its
        //  diagonal in step 2, it becomes strictly SPD.
        $A = new Tensor([$nAug, $nAug]);
        $blas->cblas_sgemm(
            101,            // CblasRowMajor
            112,            // CblasTrans   — X_aug^T is [nAug, m]
            111,            // CblasNoTrans — X_aug   is [m, nAug]
            $nAug, $nAug, $m,
            1.0,
            $Xaug->buffer, $nAug,   // A = X_aug,   lda = nAug
            $Xaug->buffer, $nAug,   // B = X_aug,   ldb = nAug
            0.0,
            $A->buffer, $nAug        // C = A,       ldc = nAug
        );

        // ── Step 2: A[j,j] += α  for j = 0 … n−1  (Ridge penalty) ────────
        //
        // The bias column (index n) is excluded — we do NOT penalise the
        // intercept, matching sklearn's fit_intercept=True semantics.
        // A PHP loop is used because BLAS has no "add scalar to stride-k
        // elements" primitive (the closest, cblas_sscal, multiplies rather
        // than adds).
        for ($j = 0; $j < $n; $j++) {
            // Diagonal element A[j, j] in row-major layout is at offset j*nAug+j.
            $A->buffer[$j * $nAug + $j] = (float) $A->buffer[$j * $nAug + $j] + $this->alpha;
        }

        // ── Step 3: b = X_aug^T @ y  [nAug] ──────────────────────────────
        //
        //  sgemv(RowMajor, Trans, M=m, N=nAug, 1.0,
        //        X_aug[m, nAug], nAug,
        //        y[m], 1,
        //        0.0, b[nAug], 1)
        //
        //  When Trans: op(A) is [nAug, m], so x must be length m → b is length nAug.
        $b = new Tensor([$nAug]);
        $blas->cblas_sgemv(
            101,            // CblasRowMajor
            112,            // CblasTrans — X_aug^T acts as [nAug, m]
            $m, $nAug,
            1.0,
            $Xaug->buffer, $nAug,
            $y->buffer, 1,
            0.0,
            $b->buffer, 1
        );

        // ── Step 4: Solve A θ = b via LAPACKE_sposv (Cholesky) ────────────
        //
        //  LAPACKE_sposv(RowMajor, 'U', n, nrhs, a, lda, b, ldb)
        //    — 'U': use the upper triangle of A (both triangles are populated
        //           by sgemm, so this is correct).
        //    — lda = nAug (physical column count of A).
        //    — ldb = 1    (b is an [nAug × 1] column vector; row-major ldb = nrhs).
        //    — A and b are OVERWRITTEN: A → Cholesky factor, b → solution θ.
        //
        //  A clone of A is NOT needed: we have no further use for the original.
        $info = $lapacke->LAPACKE_sposv(
            BlasEngine::LAPACK_ROW_MAJOR,
            'U',
            $nAug,          // n: order of the square SPD matrix
            1,              // nrhs: one right-hand side
            $A->buffer, $nAug,  // a, lda
            $b->buffer, 1       // b, ldb (b is length nAug; ldb = nrhs = 1)
        );

        if ($info !== 0) {
            throw new \RuntimeException(
                "Ridge: LAPACKE_sposv failed with info={$info}. "
                . 'The system (X^T X + αI) may be numerically singular '
                . '— try increasing alpha.'
            );
        }

        // ── Step 5: Extract coef_ and intercept_ from solution buffer ─────
        $coef = new Tensor([$n]);
        $blas->cblas_scopy($n, $b->buffer, 1, $coef->buffer, 1);

        $this->coef_          = $coef;
        $this->intercept_     = $this->fit_intercept ? (float) $b->buffer[$n] : 0.0;
        $this->n_features_in_ = $n;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict targets: ŷ = X @ coef_ + intercept_
     *
     * @param Tensor $X  Feature matrix [n_samples, n_features]
     * @return Tensor    Predictions    [n_samples]
     */
    public function predict(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "Ridge::predict() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        [$m, $n] = $X->shape;
        $blas    = BlasEngine::get()->ffi;
        $out     = new Tensor([$m]);

        // ŷ = X @ coef_  via sgemv(NoTrans)
        $blas->cblas_sgemv(
            101, 111,       // CblasRowMajor, CblasNoTrans
            $m, $n,
            1.0,
            $X->buffer, $n,
            $this->coef_->buffer, 1,
            0.0,
            $out->buffer, 1
        );

        // ŷ += intercept_  (scalar broadcast — no BLAS primitive)
        if ($this->intercept_ !== 0.0) {
            for ($i = 0; $i < $m; $i++) {
                $out->buffer[$i] = (float) $out->buffer[$i] + $this->intercept_;
            }
        }

        return $out;
    }

    // ── Metrics helper ─────────────────────────────────────────────────────

    /**
     * R² = 1 − SS_res / SS_tot  — mirrors sklearn's RegressorMixin.score().
     */
    public function score(Tensor $X, Tensor $y): float
    {
        $yPred = $this->predict($X);
        $m     = $y->size;
        $blas  = BlasEngine::get()->ffi;

        // ȳ = mean(y)  — PHP loop (unavoidable, BLAS has no signed-sum primitive)
        $yMean = 0.0;
        for ($i = 0; $i < $m; $i++) {
            $yMean += (float) $y->buffer[$i];
        }
        $yMean /= $m;

        // SS_res = ||y − ŷ||²
        $res = new Tensor([$m]);
        $blas->cblas_scopy($m, $y->buffer, 1, $res->buffer, 1);
        $blas->cblas_saxpy($m, -1.0, $yPred->buffer, 1, $res->buffer, 1);
        $ssRes = (float) $blas->cblas_sdot($m, $res->buffer, 1, $res->buffer, 1);

        // SS_tot = ||y − ȳ||²
        $tot  = new Tensor([$m]);
        $ones = Tensor::ones([$m]);
        $blas->cblas_scopy($m, $y->buffer, 1, $tot->buffer, 1);
        $blas->cblas_saxpy($m, -$yMean, $ones->buffer, 1, $tot->buffer, 1);
        $ssTot = (float) $blas->cblas_sdot($m, $tot->buffer, 1, $tot->buffer, 1);

        return $ssTot > 0.0 ? 1.0 - $ssRes / $ssTot : 1.0;
    }

    private function checkFitted(): void
    {
        if (!isset($this->coef_)) {
            throw new \RuntimeException('Ridge is not fitted. Call fit() first.');
        }
    }
}
