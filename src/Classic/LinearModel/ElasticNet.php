<?php

declare(strict_types=1);

namespace Pml\Classic\LinearModel;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  ElasticNet — sklearn.linear_model.ElasticNet
//
//  Linear regression with combined L1 and L2 regularisation (Elastic Net),
//  solved by Coordinate Descent.
//
//  ── Loss Function ────────────────────────────────────────────────────────
//
//    L(w, b) = (1/2n) · ||Xw + b − y||²₂
//              + α · l1_ratio · ||w||₁
//              + (α/2) · (1 − l1_ratio) · ||w||²₂
//
//  where:
//    α         = total regularisation strength (>0)
//    l1_ratio  = mix between L1 and L2 (0 = pure Ridge, 1 = pure Lasso)
//
//  sklearn uses the above parameterisation so that:
//    l1_ratio = 1  →  ElasticNet reduces to Lasso (α = lasso alpha)
//    l1_ratio = 0  →  ElasticNet reduces to Ridge (α/2 = ridge alpha)
//
//  ── Coordinate Descent Update ────────────────────────────────────────────
//
//  For a fixed feature j the per-coordinate subproblem is:
//
//    min_{w_j}  (1/2n)(r_i - x_{ij} w_j)² + λ₁|w_j| + (λ₂/2) w_j²
//
//  where:
//    λ₁ = α · l1_ratio          (L1 penalty)
//    λ₂ = α · (1 − l1_ratio)    (L2 penalty)
//    r  = y − X_{-j}w_{-j} − b  (partial residual w.r.t. j)
//
//  The unconstrained minimiser (L2 only) is ρ_j = (1/n) x_j^T r_j.
//  Adding L1: apply soft thresholding (same as Lasso).
//  Adding L2: changes the effective curvature from σ_j to σ_j + λ₂.
//
//    ┌──────────────────────────────────────────────────────────────────┐
//    │  ElasticNet Coordinate Update:                                   │
//    │                                                                  │
//    │    ρ_j = (1/n) x_j^T r_j                                        │
//    │    w_j* = S(ρ_j, λ₁) / (σ_j + λ₂)                             │
//    │                                                                  │
//    │  where S(ρ, λ) = sign(ρ) · max(0, |ρ| − λ)  [soft threshold]  │
//    │        σ_j    = (1/n) x_j^T x_j              [feature norm²]   │
//    │                                                                  │
//    │  Key insight: the L2 penalty ADDS to the denominator rather      │
//    │  than the numerator.  This makes the update "ridge-like" in the  │
//    │  denominator while "lasso-like" in the numerator.               │
//    │                                                                  │
//    │  Derivation: the per-feature subproblem gradient is             │
//    │    -(1/n) x_j^T r_j + (σ_j + λ₂) w_j + λ₁ sign(w_j) = 0     │
//    │  Solving for w_j with the sub-gradient of |w_j|:               │
//    │    (σ_j + λ₂) w_j = S((1/n)x_j^T r_j, λ₁)                    │
//    │    w_j* = S(ρ_j, λ₁) / (σ_j + λ₂)                             │
//    └──────────────────────────────────────────────────────────────────┘
//
//  ── Residual Maintenance & Column Stride (same as Lasso) ─────────────────
//
//  r is maintained incrementally after each coordinate update:
//    r ← r + x_j · (w_j_old − w_j_new)   via cblas_saxpy with stride d.
//
//  Column j of X (row-major [n,d]) is accessed with stride d in all BLAS
//  calls: cblas_sdot(n, &X[j], d, r, 1) computes x_j^T r without copying.
// ═══════════════════════════════════════════════════════════════════════════

final class ElasticNet implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** Coefficient vector [n_features]. */
    public readonly Tensor $coef_;

    /** Scalar intercept. */
    public readonly float $intercept_;

    public readonly int $n_features_in_;

    /** Number of CD iterations run. */
    public readonly int $n_iter_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param float $alpha          Total regularisation strength. α ≥ 0.
     * @param float $l1_ratio       Elastic Net mixing parameter.
     *                               0.0 = pure Ridge, 1.0 = pure Lasso.
     *                               Must be in [0, 1].
     * @param bool  $fit_intercept  Whether to fit an intercept term.
     * @param int   $max_iter       Maximum coordinate descent passes.
     * @param float $tol            Convergence tolerance on max coordinate change.
     */
    public function __construct(
        private readonly float $alpha         = 1.0,
        private readonly float $l1_ratio      = 0.5,
        private readonly bool  $fit_intercept = true,
        private readonly int   $max_iter      = 1000,
        private readonly float $tol           = 1e-4,
    ) {
        if ($alpha < 0.0) {
            throw new \InvalidArgumentException('ElasticNet: alpha must be >= 0.');
        }
        if ($l1_ratio < 0.0 || $l1_ratio > 1.0) {
            throw new \InvalidArgumentException('ElasticNet: l1_ratio must be in [0, 1].');
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Fit ElasticNet via Coordinate Descent.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('ElasticNet: y must be provided.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('ElasticNet: X must be 2-D [n_samples, n_features].');
        }

        [$n, $d]              = $X->shape;
        $this->n_features_in_ = $d;
        $blas                 = BlasEngine::get()->ffi;

        // ── Decompose α into L1 and L2 penalty coefficients ───────────
        //
        // sklearn parameterisation:
        //   λ₁ = α · l1_ratio        (L1 term, soft threshold threshold)
        //   λ₂ = α · (1 − l1_ratio)  (L2 term, added to denominator)
        //
        // For l1_ratio=1 (Lasso): λ₁=α, λ₂=0 → denominator = σ_j (same as Lasso).
        // For l1_ratio=0 (Ridge): λ₁=0, λ₂=α → S(ρ,0)=ρ, denom=σ_j+α (Ridge CD).
        $lambda1 = $this->alpha * $this->l1_ratio;
        $lambda2 = $this->alpha * (1.0 - $this->l1_ratio);

        // ── Precompute per-column norms: σ_j = (1/n) x_j^T x_j ───────
        //
        // The effective denominator for each coordinate update is (σ_j + λ₂).
        // Precomputing saves one sdot call per feature per iteration.
        $colNormSq    = [];
        $effectiveDen = [];
        for ($j = 0; $j < $d; $j++) {
            $colPtr           = \FFI::cast('float*', \FFI::addr($X->buffer[$j]));
            $sigmaJ           = (float)$blas->cblas_sdot($n, $colPtr, $d, $colPtr, $d) / $n;
            $colNormSq[$j]    = $sigmaJ;
            // ── Core ElasticNet denominator: σ_j + λ₂ ─────────────────
            //
            // The L2 penalty adds λ₂ to the curvature of the per-feature
            // subproblem: the update shrinks less aggressively than pure Lasso,
            // preventing coefficient blow-up when features are correlated.
            $effectiveDen[$j] = $sigmaJ + $lambda2;
        }

        // ── Initialise weights and intercept ───────────────────────────
        $w = array_fill(0, $d, 0.0);
        $b = 0.0;

        if ($this->fit_intercept) {
            $ySum = 0.0;
            for ($i = 0; $i < $n; $i++) { $ySum += (float)$y->buffer[$i]; }
            $b = $ySum / $n;
        }

        // ── Initialise residuals r = y − Xw − b = y − b (w=0) ────────
        $r = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $r->buffer[$i] = (float)$y->buffer[$i] - $b;
        }

        // ── Coordinate Descent main loop ───────────────────────────────
        $nIter = 0;

        for ($iter = 0; $iter < $this->max_iter; $iter++) {
            $nIter++;
            $maxDelta = 0.0;

            for ($j = 0; $j < $d; $j++) {
                $denJ = $effectiveDen[$j];
                if ($denJ < 1e-10) {
                    // Near-constant feature and no L2 pushback — skip safely
                    continue;
                }

                $colPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$j]));

                // ── Compute ρ_j (coordinate gradient direction) ────────
                //
                // ρ_j = (1/n) x_j^T r_j
                //     = (1/n) x_j^T (r + x_j w_j)         [add j back]
                //     = (1/n) x_j^T r  +  w_j · σ_j
                //
                // cblas_sdot with stride d reads every d-th element = column j
                $xjDotR = (float)$blas->cblas_sdot($n, $colPtr, $d, $r->buffer, 1);
                $rhoJ   = $xjDotR / $n + $w[$j] * $colNormSq[$j];

                // ── ElasticNet Coordinate Update ───────────────────────
                //
                //   w_j* = S(ρ_j, λ₁) / (σ_j + λ₂)
                //
                // Numerator: soft threshold shrinks |ρ_j| by λ₁ (L1 effect).
                //   If |ρ_j| ≤ λ₁, the feature is completely zeroed (sparsity).
                //   If |ρ_j| > λ₁, the coefficient moves toward sign(ρ_j).
                //
                // Denominator: σ_j + λ₂ inflates the effective curvature (L2 effect).
                //   Higher λ₂ → smaller coefficient magnitude → ridge-like shrinkage.
                //   λ₂ = 0 → reduces to Lasso coordinate update exactly.
                $wjNew = self::softThreshold($rhoJ, $lambda1) / $denJ;

                // ── Incremental residual update ────────────────────────
                //
                //   r ← r + x_j · (w_j_old − w_j_new)
                //
                // saxpy(n, Δ, col_j, stride_d, r, 1): O(n) BLAS-1
                $delta = $w[$j] - $wjNew;
                if (abs($delta) > 1e-14) {
                    $blas->cblas_saxpy($n, $delta, $colPtr, $d, $r->buffer, 1);
                }

                $w[$j] = $wjNew;
                if (abs($delta) > $maxDelta) { $maxDelta = abs($delta); }
            }

            // ── Intercept update (unpenalised) ─────────────────────────
            if ($this->fit_intercept) {
                $rSum = 0.0;
                for ($i = 0; $i < $n; $i++) { $rSum += (float)$r->buffer[$i]; }
                $bDelta = $rSum / $n;
                $b     += $bDelta;
                for ($i = 0; $i < $n; $i++) {
                    $r->buffer[$i] = (float)$r->buffer[$i] - $bDelta;
                }
            }

            if ($maxDelta < $this->tol) {
                break;
            }
        }

        // ── Store fitted attributes ────────────────────────────────────
        $this->n_iter_    = $nIter;
        $this->intercept_ = $b;

        $coefT = new Tensor([$d]);
        $bytes = pack('f*', ...$w);
        \FFI::memcpy($coefT->buffer, $bytes, $d * 4);
        $this->coef_ = $coefT;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict: ŷ = X @ coef_ + intercept_
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples]
     */
    public function predict(Tensor $X): Tensor
    {
        if (!isset($this->coef_)) {
            throw new \RuntimeException('ElasticNet is not fitted. Call fit() first.');
        }

        $blas = BlasEngine::get()->ffi;
        $n    = $X->shape[0];
        $d    = $X->shape[1];
        $out  = new Tensor([$n]);

        // cblas_sgemv: out = X @ coef_  ([n,d] @ [d] → [n])
        $blas->cblas_sgemv(101, 111, $n, $d, 1.0, $X->buffer, $d, $this->coef_->buffer, 1, 0.0, $out->buffer, 1);

        if ($this->intercept_ !== 0.0) {
            for ($i = 0; $i < $n; $i++) {
                $out->buffer[$i] = (float)$out->buffer[$i] + $this->intercept_;
            }
        }

        return $out;
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /**
     * Soft Thresholding operator: S(ρ, λ) = sign(ρ) · max(0, |ρ| − λ).
     *
     * ┌──────────────────────────────────────────────────────────────────┐
     * │  Applied to the L1 numerator of the ElasticNet update.          │
     * │  When λ₂ > 0, the L2 effect appears in the denominator (σ+λ₂)  │
     * │  rather than here — this factorisation is what makes ElasticNet  │
     * │  exactly decomposable into independent L1 + L2 contributions.   │
     * │                                                                  │
     * │  ρ > λ   →  ρ − λ   (positive, shrunk by L1)                   │
     * │  ρ < −λ  →  ρ + λ   (negative, shrunk by L1)                   │
     * │  |ρ|≤ λ  →  0       (zeroed: feature dropped when L1 wins)      │
     * └──────────────────────────────────────────────────────────────────┘
     */
    private static function softThreshold(float $rho, float $lambda): float
    {
        if ($rho > $lambda)  { return $rho - $lambda; }
        if ($rho < -$lambda) { return $rho + $lambda; }
        return 0.0;
    }
}
