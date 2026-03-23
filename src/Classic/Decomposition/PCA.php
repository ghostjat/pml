<?php

declare(strict_types=1);

namespace Pml\Classic\Decomposition;

use Pml\{Tensor, BlasEngine, Ops};
use Pml\Classic\{Estimator, Transformer};

// ═══════════════════════════════════════════════════════════════════════════
//  PCA — sklearn.decomposition.PCA
//
//  Principal Component Analysis via the compact (thin) Singular Value
//  Decomposition of the mean-centred data matrix.
//
//  Algorithm:
//    1. Centre X:     X_c = X − mean(X, axis=0)
//    2. Thin SVD:     X_c = U · diag(S) · Vt      (LAPACKE_sgesvd, jobu='S')
//    3. Components:   components_ = Vt[:n_components, :]  [n_comp, n_feat]
//    4. Project:      X_new = X_c @ components_^T         [n_samp, n_comp]
//
//  Why SVD and not eigendecomposition of X^T X?
//    SVD is numerically superior for wide matrices (n_features >> n_samples)
//    and avoids squaring the condition number.  This is exactly what sklearn
//    does internally (TruncatedSVD on centered data).
//
//  BLAS/LAPACK calls:
//    fit():       sgemv (column means), saxpy (centering), LAPACKE_sgesvd
//    transform(): saxpy (centering), cblas_sgemm (projection)
// ═══════════════════════════════════════════════════════════════════════════

final class PCA implements Estimator, Transformer
{
    // ── Fitted attributes (sklearn naming convention) ──────────────────────
    /** @var Tensor  Row vectors of principal axes [n_components, n_features] */
    public readonly Tensor $components_;

    /** @var Tensor  Per-component variance = S[i]² / (n_samples − 1) */
    public readonly Tensor $explained_variance_;

    /** @var Tensor  Fraction of total variance explained by each component */
    public readonly Tensor $explained_variance_ratio_;

    /** @var Tensor  Singular values corresponding to each component */
    public readonly Tensor $singular_values_;

    /** @var Tensor  Per-feature mean of the training set */
    public readonly Tensor $mean_;

    public readonly int $n_components_;
    public readonly int $n_features_in_;
    public readonly int $n_samples_seen_;

    /**
     * @param int $n_components  Number of principal components to retain.
     */
    public function __construct(
        private readonly int $n_components,
    ) {
        if ($n_components < 1) {
            throw new \InvalidArgumentException('PCA: n_components must be >= 1.');
        }
    }

    // ── Estimator ─────────────────────────────────────────────────────────

    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('PCA::fit() requires a 2D tensor [n_samples, n_features].');
        }

        [$m, $n] = $X->shape;
        $k       = min($m, $n);

        if ($this->n_components > $k) {
            throw new \InvalidArgumentException(
                "PCA: n_components={$this->n_components} > min(n_samples, n_features)={$k}."
            );
        }

        $blas = BlasEngine::get()->ffi;

        // ── 1. Column means via sgemv ──────────────────────────────────────
        //  mean[j] = (1/m) * Σ_i X[i,j]  — same trick as StandardScaler.
        $mean = new Tensor([$n]);
        $ones = Tensor::ones([$m]);
        $blas->cblas_sgemv(101, 112, $m, $n, 1.0 / $m, $X->buffer, $n,
                           $ones->buffer, 1, 0.0, $mean->buffer, 1);

        // ── 2. Centre X_c = X − mean ──────────────────────────────────────
        $Xc = $X->clone();
        for ($i = 0; $i < $m; $i++) {
            $rowPtr = \FFI::cast('float*', \FFI::addr($Xc->buffer[$i * $n]));
            $blas->cblas_saxpy($n, -1.0, $mean->buffer, 1, $rowPtr, 1);
        }

        // ── 3. Thin SVD: X_c = U · diag(S) · Vt ──────────────────────────
        //
        //  Ops::svd() calls LAPACKE_sgesvd with jobu='S', jobvt='S' giving:
        //    U  [m, k]  — left singular vectors  (k = min(m,n))
        //    S  [k]     — singular values in descending order
        //    Vt [k, n]  — right singular vectors (rows = principal axes)
        //
        //  We discard U; only Vt[:n_components] and S[:n_components] are kept.
        [, $S, $Vt] = Ops::svd($Xc);

        // ── 4. Retain top n_components rows of Vt ─────────────────────────
        $components = new Tensor([$this->n_components, $n]);
        $blas->cblas_scopy(
            $this->n_components * $n,
            $Vt->buffer, 1,
            $components->buffer, 1
        );

        // ── 5. Explained variance = S[i]² / (m − 1) ──────────────────────
        $ev    = new Tensor([$this->n_components]);
        $evr   = new Tensor([$this->n_components]);
        $sv    = new Tensor([$this->n_components]);

        // Total variance across ALL k singular values (for ratio denominator)
        $totalVar = 0.0;
        for ($i = 0; $i < $k; $i++) {
            $si        = (float) $S->buffer[$i];
            $totalVar += ($si * $si) / max(1, $m - 1);
        }

        for ($i = 0; $i < $this->n_components; $i++) {
            $si               = (float) $S->buffer[$i];
            $evi              = ($si * $si) / max(1, $m - 1);
            $ev->buffer[$i]   = $evi;
            $evr->buffer[$i]  = $totalVar > 0.0 ? $evi / $totalVar : 0.0;
            $sv->buffer[$i]   = $si;
        }

        $this->mean_                    = $mean;
        $this->components_              = $components;
        $this->explained_variance_      = $ev;
        $this->explained_variance_ratio_ = $evr;
        $this->singular_values_         = $sv;
        $this->n_components_            = $this->n_components;
        $this->n_features_in_           = $n;
        $this->n_samples_seen_          = $m;

        return $this;
    }

    // ── Transformer ───────────────────────────────────────────────────────

    public function transform(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "PCA::transform() expected [*, {$this->n_features_in_}], "
                . "got [" . implode(', ', $X->shape) . "]."
            );
        }

        [$m, $n] = $X->shape;
        $blas    = BlasEngine::get()->ffi;

        // ── Centre X ──────────────────────────────────────────────────────
        $Xc = $X->clone();
        for ($i = 0; $i < $m; $i++) {
            $rowPtr = \FFI::cast('float*', \FFI::addr($Xc->buffer[$i * $n]));
            $blas->cblas_saxpy($n, -1.0, $this->mean_->buffer, 1, $rowPtr, 1);
        }

        // ── Project: X_new = X_c @ components_^T ─────────────────────────
        //
        //  components_ is [n_components, n_features].
        //  X_c is         [n_samples,    n_features].
        //  X_c @ components_^T = [n_samples, n_components] via sgemm.
        //
        //  sgemm(RowMajor, NoTrans, Trans, m, n_comp, n_feat,
        //        1.0, Xc, n_feat, components_, n_feat, 0.0, out, n_comp)
        $nc  = $this->n_components;
        $out = new Tensor([$m, $nc]);
        $blas->cblas_sgemm(
            101,   // CblasRowMajor
            111,   // CblasNoTrans  — X_c is [m, n]
            112,   // CblasTrans    — components_ [n_comp, n] treated as [n, n_comp]
            $m, $nc, $n,
            1.0,
            $Xc->buffer, $n,
            $this->components_->buffer, $n,
            0.0,
            $out->buffer, $nc
        );

        return $out;
    }

    public function fit_transform(Tensor $X, ?Tensor $y = null): Tensor
    {
        return $this->fit($X, $y)->transform($X);
    }

    // ── Inverse transform ─────────────────────────────────────────────────

    /**
     * Map components back to original feature space.
     * X_orig ≈ X_new @ components_ + mean_
     */
    public function inverse_transform(Tensor $X): Tensor
    {
        $this->checkFitted();

        [$m, $nc] = $X->shape;
        $n        = $this->n_features_in_;
        $blas     = BlasEngine::get()->ffi;
        $out      = new Tensor([$m, $n]);

        // out = X_new @ components_   [m, nc] @ [nc, n] → [m, n]
        $blas->cblas_sgemm(
            101, 111, 111,
            $m, $n, $nc,
            1.0,
            $X->buffer, $nc,
            $this->components_->buffer, $n,
            0.0,
            $out->buffer, $n
        );

        // Add back mean (broadcast per row)
        for ($i = 0; $i < $m; $i++) {
            $rowPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$i * $n]));
            $blas->cblas_saxpy($n, 1.0, $this->mean_->buffer, 1, $rowPtr, 1);
        }

        return $out;
    }

    private function checkFitted(): void
    {
        if (!isset($this->components_)) {
            throw new \RuntimeException('PCA is not fitted. Call fit() first.');
        }
    }
}
