<?php

declare(strict_types=1);

namespace Pml\Classic\LinearModel;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  Lasso — sklearn.linear_model.Lasso
//
//  Linear regression with L1 regularisation, solved by Coordinate Descent.
//
//  ── Loss Function ────────────────────────────────────────────────────────
//
//    L(w, b) = (1 / 2n) · ||Xw + b - y||²₂  +  α · ||w||₁
//
//  where n = n_samples, α = regularisation strength.
//  The intercept b is NOT regularised (sklearn convention).
//
//  ── Coordinate Descent ───────────────────────────────────────────────────
//
//  Coordinate descent updates one weight w_j at a time, keeping all other
//  weights fixed, then cycles through all j until convergence.
//
//  For a fixed j, the partial problem in w_j alone is:
//
//    min_{w_j}  (1/2n) Σ_i (r_i - x_{ij} w_j)²  +  α |w_j|
//
//  where r_i = y_i - Σ_{k≠j} x_{ik} w_k - b   (partial residual w.r.t. j)
//
//  The unconstrained minimiser of the squared term is:
//
//    ρ_j = (1/n) · x_j^T r_j   (where r_j is the partial residual)
//
//  Including the |w_j| penalty, the exact solution is the Soft Thresholding
//  operator S applied to ρ_j:
//
//    ┌──────────────────────────────────────────────────────────────────┐
//    │  S(ρ, λ)  =  sign(ρ) · max(0, |ρ| − λ)                        │
//    │                                                                  │
//    │  Geometrically: shift ρ toward zero by λ, clamp at zero.        │
//    │  This is the proximal operator of the L1 norm.                  │
//    └──────────────────────────────────────────────────────────────────┘
//
//  Then:  w_j* = S(ρ_j, α) / σ_j
//
//  where σ_j = (1/n) x_j^T x_j  (the per-feature scaling / curvature).
//
//  ── Efficient Residual Maintenance ───────────────────────────────────────
//
//  Rather than recomputing r = y - Xw from scratch after each coordinate
//  update (O(nd) per update), we maintain r incrementally:
//
//    After updating w_j by Δ_j = w_j_new − w_j_old:
//      r ← r + x_j · Δ_j      (via cblas_saxpy with stride d on column j)
//
//  This reduces each coordinate update to O(n) — one BLAS sdot + one saxpy.
//
//  ── Column Stride Trick ──────────────────────────────────────────────────
//
//  X is stored row-major [n, d].  Column j has elements at byte offsets
//  j, j+d, j+2d, … (stride = d elements = d*sizeof(float) bytes).
//
//  BLAS level-1 routines accept an increment parameter (incX), so
//  cblas_sdot(n, &X[j], d, &r[0], 1) computes x_j^T r without any
//  column-copy temporary — pure strided in-place reads.
//
//  ── Convergence ──────────────────────────────────────────────────────────
//
//  After each full pass over all features, the maximum absolute coordinate
//  change max_j |Δ_j| is compared to tol.  Convergence is declared if
//  max_j |Δ_j| < tol.  Maximum max_iter passes are allowed.
// ═══════════════════════════════════════════════════════════════════════════

final class Lasso implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** Coefficient vector [n_features]. */
    public readonly Tensor $coef_;

    /** Scalar intercept. */
    public readonly float $intercept_;

    public readonly int $n_features_in_;

    /** Number of coordinate descent iterations actually run. */
    public readonly int $n_iter_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param float $alpha          L1 regularisation strength. α ≥ 0.
     * @param bool  $fit_intercept  Whether to fit an intercept term.
     * @param int   $max_iter       Maximum number of coordinate descent passes.
     * @param float $tol            Convergence tolerance on max coordinate change.
     */
    public function __construct(
        private readonly float $alpha         = 1.0,
        private readonly bool  $fit_intercept = true,
        private readonly int   $max_iter      = 1000,
        private readonly float $tol           = 1e-4,
    ) {
        if ($alpha < 0.0) {
            throw new \InvalidArgumentException('Lasso: alpha must be >= 0.');
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Fit the Lasso model via Coordinate Descent.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('Lasso: y must be provided.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('Lasso: X must be 2-D [n_samples, n_features].');
        }

        [$n, $d]              = $X->shape;
        $this->n_features_in_ = $d;
        $blas                 = BlasEngine::get()->ffi;

        // ── Precompute per-column norms squared: σ_j = (1/n) x_j^T x_j ─
        //
        // These are the curvatures of the per-feature subproblems; they
        // only need to be computed once because the columns of X do not change.
        $colNormSq = [];
        for ($j = 0; $j < $d; $j++) {
            $colPtr        = \FFI::cast('float*', \FFI::addr($X->buffer[$j]));
            $colNormSq[$j] = (float)$blas->cblas_sdot($n, $colPtr, $d, $colPtr, $d) / $n;
        }

        // ── Initialise weights to zero, intercept to mean(y) ──────────
        //
        // Starting from zero is equivalent to sklearn's warm_start=False.
        // Setting b = mean(y) initially centres the residuals, which
        // speeds up convergence for the first few iterations.
        $w = array_fill(0, $d, 0.0);
        $b = 0.0;

        if ($this->fit_intercept) {
            $ySum = 0.0;
            for ($i = 0; $i < $n; $i++) { $ySum += (float)$y->buffer[$i]; }
            $b = $ySum / $n;
        }

        // ── Initialise residuals r = y - Xw - b = y - b (since w=0) ──
        $r = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $r->buffer[$i] = (float)$y->buffer[$i] - $b;
        }

        // ── Coordinate Descent main loop ───────────────────────────────
        $l1    = $this->alpha;
        $nIter = 0;

        for ($iter = 0; $iter < $this->max_iter; $iter++) {
            $nIter++;
            $maxDelta = 0.0;

            // ── Inner loop: cycle over all features j ─────────────────
            for ($j = 0; $j < $d; $j++) {
                $sigmaJ = $colNormSq[$j];
                if ($sigmaJ < 1e-10) {
                    // Near-constant feature — skip to avoid division by zero
                    continue;
                }

                $colPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$j]));

                // ── Step A: Compute ρ_j ────────────────────────────────
                //
                // Partial residual w.r.t. j: r_j = r + x_j * w_j
                // We avoid materialising r_j by using the identity:
                //
                //   (1/n) x_j^T r_j = (1/n) x_j^T r  +  w_j * σ_j
                //
                // cblas_sdot with stride d reads every d-th element = column j
                $xjDotR = (float)$blas->cblas_sdot($n, $colPtr, $d, $r->buffer, 1);
                $rhoJ   = $xjDotR / $n + $w[$j] * $sigmaJ;

                // ── Step B: Soft Thresholding — the core of Lasso CD ──
                //
                //   S(ρ, λ) = sign(ρ) · max(0, |ρ| - λ)
                //
                // Full derivation: the per-feature subproblem reduces to
                // minimising  ½σ_j(w_j - ρ_j/σ_j)² + λ|w_j|, whose solution
                // via the proximal operator of the L1 norm is exactly S(ρ_j, λ)/σ_j.
                //
                //   |ρ_j| > λ  →  w_j* = S(ρ_j, λ) / σ_j  (non-zero coefficient)
                //   |ρ_j| ≤ λ  →  w_j* = 0                 (feature pruned by L1)
                $wjNew = self::softThreshold($rhoJ, $l1) / $sigmaJ;

                // ── Step C: Update residuals incrementally ────────────
                //
                //   r ← r + x_j · (w_j_old - w_j_new)
                //
                // cblas_saxpy with stride d: y += alpha * x[0::stride]
                // This "un-applies" the old contribution and applies the new one
                // in a single BLAS call, keeping r = y - Xw - b exactly.
                $delta = $w[$j] - $wjNew;
                if (abs($delta) > 1e-14) {
                    $blas->cblas_saxpy($n, $delta, $colPtr, $d, $r->buffer, 1);
                }

                $w[$j] = $wjNew;
                if (abs($delta) > $maxDelta) { $maxDelta = abs($delta); }
            }

            // ── Intercept update ───────────────────────────────────────
            //
            // The intercept b* (unpenalised) satisfies ∂L/∂b = 0:
            //   b* = (1/n) Σ_i (y_i - Σ_j x_{ij} w_j) = b + mean(r)
            //
            // We update b by the residual mean and subtract that same value
            // from r to keep r = y - Xw - b correct.
            if ($this->fit_intercept) {
                $rSum = 0.0;
                for ($i = 0; $i < $n; $i++) { $rSum += (float)$r->buffer[$i]; }
                $bDelta = $rSum / $n;
                $b     += $bDelta;
                // r ← r − bDelta  (PHP loop: BLAS has no scalar-subtract primitive)
                for ($i = 0; $i < $n; $i++) {
                    $r->buffer[$i] = (float)$r->buffer[$i] - $bDelta;
                }
            }

            // ── Convergence check ──────────────────────────────────────
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
            throw new \RuntimeException('Lasso is not fitted. Call fit() first.');
        }

        $blas = BlasEngine::get()->ffi;
        $n    = $X->shape[0];
        $d    = $X->shape[1];
        $out  = new Tensor([$n]);

        // ŷ = X @ coef_  via cblas_sgemv (matrix-vector: [n,d] @ [d] → [n])
        $blas->cblas_sgemv(101, 111, $n, $d, 1.0, $X->buffer, $d, $this->coef_->buffer, 1, 0.0, $out->buffer, 1);

        // Add intercept (PHP loop — BLAS has no "add scalar to vector" primitive)
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
     * │  This is the PROXIMAL OPERATOR of the L1 norm:                  │
     * │    prox_{λ‖·‖₁}(ρ) = argmin_w  λ|w| + ½(w - ρ)²              │
     * │                                                                  │
     * │  Derivation (sub-gradient of the closed-form objective):        │
     * │    For w > 0:  ∂/∂w [½(w-ρ)² + λw] = (w-ρ) + λ = 0           │
     * │                → w* = ρ - λ  (valid only when ρ > λ)           │
     * │    For w < 0:  ∂/∂w [½(w-ρ)² - λw] = (w-ρ) - λ = 0           │
     * │                → w* = ρ + λ  (valid only when ρ < -λ)          │
     * │    For w = 0:  zero is optimal iff |ρ| ≤ λ                     │
     * │                (sub-gradient of |w| at 0 spans [-1, +1])       │
     * └──────────────────────────────────────────────────────────────────┘
     *
     * @param float $rho    Unconstrained minimiser (coordinate gradient).
     * @param float $lambda L1 penalty (= Lasso α).
     * @return float        Shrunk value.
     */
    private static function softThreshold(float $rho, float $lambda): float
    {
        if ($rho > $lambda) {
            return $rho - $lambda;   // positive side: shrink toward zero by λ
        }
        if ($rho < -$lambda) {
            return $rho + $lambda;   // negative side: shrink toward zero by λ
        }
        return 0.0;                  // dead-zone [−λ, λ]: L1 regularises to zero
    }
}
