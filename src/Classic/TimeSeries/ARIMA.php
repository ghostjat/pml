<?php

declare(strict_types=1);

namespace Pml\Classic\TimeSeries;

use Pml\{Tensor, BlasEngine, LinAlg};

// ═══════════════════════════════════════════════════════════════════════════
//  ARIMA — AutoRegressive Integrated Moving Average
//
//  Implements the Box-Jenkins ARIMA(p, d, q) model for univariate time series.
//
//  ── Model Definition ──────────────────────────────────────────────────────
//
//  Let w_t = ∇^d y_t  (the d-times differenced series).  Then:
//
//    w_t = c + φ_1 w_{t-1} + … + φ_p w_{t-p}          (AR part)
//             + ε_t + θ_1 ε_{t-1} + … + θ_q ε_{t-q}   (MA part, standard signs)
//
//  where ε_t ~ WN(0, σ²) are white-noise innovations.
//
//  ── Parameter Estimation: Hannan-Rissanen CLS ──────────────────────────
//
//  For ARMA(p,q) on the differenced series w:
//
//  Step 0 (q=0 shortcut): if q=0 → OLS directly on AR lags.
//
//  Step 1 (long AR): Fit AR(m) via OLS where m = min(⌊n/4⌋, max(10, 2(p+q))).
//    Provides preliminary residuals ε̂_t for the MA regressors.
//    Uses BLAS sgemm to form X^T X, sgemv for X^T y.
//
//  Step 2 (ARMA design matrix): Build X_t = [w_{t-1},…,w_{t-p}, ε̂_{t-1},…,ε̂_{t-q}, 1]
//    and solve the normal equations  (X^T X + λI) β = X^T w  via LU (LinAlg).
//    The ridge term λ = 1e-6 · trace(X^T X) / nparams prevents near-singularity.
//
//  ── Normal Equations via BLAS ─────────────────────────────────────────────
//
//  X is [n_eff × nparams] (filled via PHP loop — scattered lag data, unavoidable).
//  Then:
//    X^T X  — cblas_sgemm(Trans, NoTrans, nparams, nparams, n_eff)
//    X^T y  — cblas_sgemv(Trans, n_eff, nparams)
//  Both are small (nparams ≤ ~15) and solved with LinAlg::solve().
//
//  ── Forecasting ───────────────────────────────────────────────────────────
//
//  h-step-ahead forecasts on the differenced series:
//    ŵ_{n+k} = c + Σ_i φ_i ŵ_{n+k-i} + Σ_j θ_j ε̂_{n+k-j}
//
//  where ε̂_{n+k-j} = 0 for k > j  (unknown future shocks set to zero),
//  and ε̂_{n+k-j} = residual at position n+k-j for k ≤ j.
//
//  Forecasts are un-differenced by d cumulative integrations using the
//  stored boundary values of each intermediate difference series.
//
//  ── Diagnostics ───────────────────────────────────────────────────────────
//
//    σ̂² = RSS / (n_eff − nparams)            (corrected residual variance)
//    log L̂ = −n_eff/2 · (1 + ln(2π σ̂²))     (Gaussian log-likelihood approx)
//    AIC = 2k − 2 ln L̂                       (k = nparams + 1 for σ²)
//    BIC = k ln(n_eff) − 2 ln L̂
//
//  ── Complexity ────────────────────────────────────────────────────────────
//
//  fit:      O(n·m) PHP for long-AR X build + O(n·nparams) BLAS for X^T X
//  forecast: O(h·(p+q)) PHP — negligible
// ═══════════════════════════════════════════════════════════════════════════

final class ARIMA
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * AR coefficients φ_1 … φ_p.
     * @var float[]
     */
    public readonly array $ar_params_;

    /**
     * MA coefficients θ_1 … θ_q  (standard positive-sign convention:
     * w_t = … + θ_j ε_{t-j}).
     * @var float[]
     */
    public readonly array $ma_params_;

    /** Constant/intercept term c. */
    public readonly float $const_;

    /** Corrected residual variance σ̂² = RSS / (n_eff − nparams). */
    public readonly float $sigma2_;

    /**
     * In-sample innovations on the differenced series (length n_eff).
     * @var float[]
     */
    public readonly array $residuals_;

    /** Length of the original training series. */
    public readonly int $n_obs_;

    /** Akaike Information Criterion. */
    public readonly float $aic_;

    /** Bayesian Information Criterion. */
    public readonly float $bic_;

    // ── Private forecast state ─────────────────────────────────────────────

    /**
     * Last max(p, 1) values of the d-times-differenced series.
     * Needed to seed the recursive forecast.
     * @var float[]
     */
    private array $diffedTail_;

    /**
     * Last q residuals from the ARMA fit.
     * @var float[]
     */
    private array $residTail_;

    /**
     * Boundary values for undifferencing: $undiffTails_[i] = last observed
     * value of the i-th order difference series (i = 0 is raw y).
     * @var float[]
     */
    private array $undiffTails_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int  $p             AR order (≥ 0)
     * @param int  $d             Integration (differencing) order (≥ 0)
     * @param int  $q             MA order (≥ 0)
     * @param bool $includeConst  Whether to include a constant term.
     *                            Best practice: true when d = 0; optional when d ≥ 1.
     */
    public function __construct(
        private readonly int  $p            = 1,
        private readonly int  $d            = 1,
        private readonly int  $q            = 0,
        private readonly bool $includeConst = true,
    ) {
        if ($p < 0 || $d < 0 || $q < 0) {
            throw new \InvalidArgumentException('ARIMA: p, d, q must be non-negative integers.');
        }
        if ($p === 0 && $q === 0 && !$includeConst) {
            throw new \InvalidArgumentException('ARIMA(0,d,0) without a constant predicts zero — meaningless.');
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Fit the ARIMA model to a univariate time series.
     *
     * @param float[] $y  1-D PHP array of observations (at least p+d+q+10 elements).
     */
    public function fit(array $y): static
    {
        $y = array_values($y);
        $n = count($y);

        $minLen = $this->p + $this->d + $this->q + 10;
        if ($n < $minLen) {
            throw new \InvalidArgumentException(
                "ARIMA: series too short. Need ≥ {$minLen} observations, got {$n}."
            );
        }

        $this->n_obs_ = $n;

        // ── Step 1: Apply d-order differencing ────────────────────────────
        //
        // Also store boundary values of each intermediate difference series
        // for undifferencing during forecast.
        //
        // $undiffTails_[$i] = last observed value of the $i-th difference series.
        //   i=0 → raw y_n
        //   i=1 → (Δy)_n
        //   ...
        //   i=$d-1 → (Δ^{d-1} y)_n
        //
        $undiffTails = [];
        $series      = $y;

        for ($i = 0; $i < $this->d; $i++) {
            $undiffTails[] = $series[count($series) - 1];
            $series        = self::diff1($series);
        }

        // $series is now the d-times-differenced series w_1 … w_{n-d}
        $w = $series;

        // ── Step 2: Estimate ARMA(p, q) on $w ────────────────────────────
        [$arCoeffs, $maCoeffs, $constant, $residuals] =
            $this->fitARMA($w, $this->p, $this->q, $this->includeConst);

        // ── Step 3: Compute diagnostics ───────────────────────────────────
        $nEff    = count($residuals);
        $nparams = $this->p + $this->q + ($this->includeConst ? 1 : 0);
        $df      = max(1, $nEff - $nparams);

        $rss = 0.0;
        foreach ($residuals as $e) { $rss += $e * $e; }

        $sigma2  = $rss / $df;
        $logLik  = -0.5 * $nEff * (1.0 + log(2.0 * M_PI * max(1e-15, $sigma2)));
        $k       = $nparams + 1;  // +1 for sigma²

        // ── Step 4: Store fitted attributes ───────────────────────────────
        $this->ar_params_ = $arCoeffs;
        $this->ma_params_ = $maCoeffs;
        $this->const_     = $constant;
        $this->sigma2_    = $sigma2;
        $this->residuals_ = $residuals;
        $this->aic_       = 2.0 * $k - 2.0 * $logLik;
        $this->bic_       = $k * log($nEff) - 2.0 * $logLik;

        // ── Step 5: Store forecast seeds ──────────────────────────────────
        $tailLen           = max($this->p, $this->q, 1);
        $this->diffedTail_ = array_slice($w, -$tailLen);
        $this->residTail_  = $this->q > 0
            ? array_slice($residuals, -$this->q)
            : [];
        $this->undiffTails_ = $undiffTails;

        return $this;
    }

    /**
     * Generate h-step-ahead point forecasts.
     *
     * Returns an array of length $h with forecasts for time n+1, n+2, …, n+h
     * in the original (un-differenced) scale.
     *
     * @param  int     $h  Number of steps ahead (≥ 1).
     * @return float[]
     */
    public function forecast(int $h = 1): array
    {
        if ($h < 1) {
            throw new \InvalidArgumentException('ARIMA::forecast(): h must be ≥ 1.');
        }

        $this->checkFitted();

        $p         = $this->p;
        $q         = $this->q;
        $arCoeffs  = $this->ar_params_;
        $maCoeffs  = $this->ma_params_;
        $c         = $this->const_;

        // Seed the recursive buffer with the last $max(p,q) observed diffs
        // plus the last $q residuals.
        // $wBuf[k] = w at position k (k ≥ 0 are forecasts, k < 0 are observed).
        // We use a flat array indexed from 0; offset maps 0 → first forecast.
        $wObs  = $this->diffedTail_;   // observed w values (most recent last)
        $eObs  = $this->residTail_;    // observed residuals (most recent last)

        $wFcst = [];   // w forecasts w_{n+1}, ..., w_{n+h}

        for ($k = 0; $k < $h; $k++) {
            $val = $c;

            // AR terms: φ_i * w_{n+k-i}   for i=1..p
            for ($i = 1; $i <= $p; $i++) {
                $lag = $k - $i;   // index into forecast array (0-based) when ≥ 0
                if ($lag >= 0) {
                    // Already-forecasted w
                    $val += $arCoeffs[$i - 1] * $wFcst[$lag];
                } else {
                    // Observed w: $wObs is indexed from the end, so
                    // position n+k-i = n - ($i - $k) → index from tail: ($i - $k - 1)
                    $obsIdx = count($wObs) - ($i - $k);
                    $val += $arCoeffs[$i - 1] * ($wObs[$obsIdx] ?? 0.0);
                }
            }

            // MA terms: θ_j * ε_{n+k-j}   for j=1..q
            // Future shocks ε_{n+k-j} = 0 whenever k-j > 0 (k > j).
            for ($j = 1; $j <= $q; $j++) {
                if ($k < $j) {
                    // Still in the observed residual window
                    $obsIdx = count($eObs) - ($j - $k);
                    $val += $maCoeffs[$j - 1] * ($eObs[$obsIdx] ?? 0.0);
                }
                // else: future shock = 0, contributes nothing
            }

            $wFcst[] = $val;
        }

        // ── Undifference d times ──────────────────────────────────────────
        return $this->undiff($wFcst);
    }

    /**
     * Return a brief text summary of the fitted model.
     */
    public function summary(): string
    {
        $this->checkFitted();

        $lines = [
            str_repeat('═', 54),
            sprintf(' ARIMA(%d,%d,%d)%s', $this->p, $this->d, $this->q,
                $this->includeConst ? ' + const' : ''),
            str_repeat('─', 54),
            sprintf(' n_obs   : %d   n_eff : %d',
                $this->n_obs_, count($this->residuals_)),
            sprintf(' σ²      : %.6f   √σ² : %.6f',
                $this->sigma2_, sqrt($this->sigma2_)),
            sprintf(' AIC     : %.4f', $this->aic_),
            sprintf(' BIC     : %.4f', $this->bic_),
            str_repeat('─', 54),
        ];

        if ($this->includeConst) {
            $lines[] = sprintf(' const   : %+.6f', $this->const_);
        }
        foreach ($this->ar_params_ as $i => $v) {
            $lines[] = sprintf(' ar.%-5s: %+.6f', 'L' . ($i + 1), $v);
        }
        foreach ($this->ma_params_ as $i => $v) {
            $lines[] = sprintf(' ma.%-5s: %+.6f', 'L' . ($i + 1), $v);
        }
        $lines[] = str_repeat('═', 54);

        return implode("\n", $lines) . "\n";
    }

    // ── ARMA fitting engine (also used by SARIMA) ─────────────────────────

    /**
     * Hannan-Rissanen Conditional Least Squares for ARMA(p,q).
     *
     * @param  float[] $w            Stationary series (d-times differenced).
     * @param  int     $p            AR order (0 = pure MA).
     * @param  int     $q            MA order (0 = pure AR).
     * @param  bool    $includeConst Include an intercept term.
     * @param  int[]   $arLags       AR lag positions (1-indexed). Default: 1..p.
     * @param  int[]   $maLags       MA lag positions (1-indexed). Default: 1..q.
     * @return array   [ar_coeffs[], ma_coeffs[], const, residuals[]]
     *                 ar_coeffs is indexed to match $arLags;
     *                 ma_coeffs is indexed to match $maLags.
     */
    public static function fitARMA(
        array $w,
        int   $p,
        int   $q,
        bool  $includeConst = true,
        array $arLags       = [],
        array $maLags       = [],
    ): array {
        // Default lags: 1, 2, ..., p  and  1, 2, ..., q
        if ($arLags === []) {
            for ($i = 1; $i <= $p; $i++) { $arLags[] = $i; }
        }
        if ($maLags === []) {
            for ($j = 1; $j <= $q; $j++) { $maLags[] = $j; }
        }

        $n      = count($w);
        $maxAR  = ($arLags !== []) ? max($arLags) : 0;
        $maxMA  = ($maLags !== []) ? max($maLags) : 0;
        $maxLag = max($maxAR, $maxMA, 1);

        // ── Step 1: Fit long AR(m) to obtain preliminary innovations ──────
        //
        // Only needed when q > 0 (MA terms require innovation estimates).
        $longResiduals = array_fill(0, $n, 0.0);  // default: zero innovations

        if ($q > 0) {
            $m = min((int) floor($n / 4.0), max(10, 2 * ($p + $q)));
            $m = min($m, $n - 2);   // can't exceed series length

            [$longAR] = self::solveOLS($w, range(1, $m), [], false);
            // Compute long-AR residuals ê_t
            for ($t = $m; $t < $n; $t++) {
                $pred = 0.0;
                foreach ($longAR as $k => $phi) {
                    $pred += $phi * $w[$t - ($k + 1)];
                }
                $longResiduals[$t] = $w[$t] - $pred;
            }
            // Residuals before burn-in remain 0 (conservative: they don't
            // appear in the effective design matrix window anyway)
        }

        // ── Step 2: Solve for ARMA parameters via OLS ─────────────────────
        [$arCoeffs, $maCoeffs, $constant, $residuals] =
            self::solveOLS($w, $arLags, $maLags, $includeConst, $longResiduals);

        return [$arCoeffs, $maCoeffs, $constant, $residuals];
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * Build the ARMA design matrix and solve the normal equations.
     *
     * Design matrix X [n_eff × nparams]:
     *   columns 0..nAR-1   : w_{t-arLags[0]}, …, w_{t-arLags[nAR-1]}
     *   columns nAR..nAR+nMA-1 : ε̂_{t-maLags[0]}, …
     *   last column (if const): 1.0
     *
     * Normal equations solved via BLAS (X^T X, X^T y) + LinAlg LU.
     *
     * @param  float[] $w           Stationary series.
     * @param  int[]   $arLags      Active AR lags (1-indexed).
     * @param  int[]   $maLags      Active MA lags (1-indexed).
     * @param  bool    $includeConst
     * @param  float[] $innovations Preliminary innovations for MA columns.
     * @return array   [ar_coeffs[], ma_coeffs[], const, residuals[]]
     */
    private static function solveOLS(
        array $w,
        array $arLags,
        array $maLags,
        bool  $includeConst = true,
        array $innovations  = [],
    ): array {
        $n     = count($w);
        $nAR   = count($arLags);
        $nMA   = count($maLags);
        $nCst  = $includeConst ? 1 : 0;
        $nP    = $nAR + $nMA + $nCst;   // total parameters

        if ($nP === 0) {
            return [[], [], 0.0, $w];
        }

        // The effective window starts at the maximum lag required
        $maxAR   = ($arLags !== []) ? max($arLags) : 0;
        $maxMA   = ($maLags !== []) ? max($maLags) : 0;
        $maxLag  = max($maxAR, $maxMA);
        $startT  = $maxLag;              // first t we can build a full row for
        $nEff    = $n - $startT;

        if ($nEff <= $nP) {
            throw new \RuntimeException(
                "ARIMA: too few effective observations ({$nEff}) for {$nP} parameters. "
                . 'Reduce p, q, or d, or use more data.'
            );
        }

        $blas = BlasEngine::get()->ffi;

        // ── Allocate Tensors for BLAS normal equations ─────────────────────
        $X   = new Tensor([$nEff, $nP]);    // design matrix [n_eff × nP]
        $yT  = new Tensor([$nEff]);         // target vector w[startT..n-1]

        // ── Fill design matrix (PHP loop — scattered lag reads) ────────────
        // Innovations array: if empty, treat as all-zero
        $hasInno = ($innovations !== []);

        for ($t = $startT; $t < $n; $t++) {
            $row = $t - $startT;
            $off = $row * $nP;

            // AR columns
            for ($k = 0; $k < $nAR; $k++) {
                $X->buffer[$off + $k] = $w[$t - $arLags[$k]];
            }

            // MA columns (preliminary innovations from long AR)
            for ($k = 0; $k < $nMA; $k++) {
                $X->buffer[$off + $nAR + $k] =
                    $hasInno ? ($innovations[$t - $maLags[$k]] ?? 0.0) : 0.0;
            }

            // Constant column
            if ($includeConst) {
                $X->buffer[$off + $nP - 1] = 1.0;
            }

            $yT->buffer[$row] = $w[$t];
        }

        // ── BLAS: X^T X  [nP × nP] ────────────────────────────────────────
        //
        //   sgemm(RowMajor, Trans, NoTrans, M=nP, N=nP, K=nEff,
        //         alpha=1.0, A=X[nEff×nP], lda=nP,
        //                    B=X[nEff×nP], ldb=nP,
        //         beta=0.0,  C=XTX[nP×nP], ldc=nP)
        $XTX = new Tensor([$nP, $nP]);
        $blas->cblas_sgemm(
            101, 112, 111,
            $nP, $nP, $nEff,
            1.0, $X->buffer, $nP, $X->buffer, $nP,
            0.0, $XTX->buffer, $nP
        );

        // ── BLAS: X^T y  [nP] ─────────────────────────────────────────────
        //
        //   sgemv(RowMajor, Trans, M=nEff, N=nP,
        //         alpha=1.0, A=X[nEff×nP], lda=nP, x=y[nEff], incx=1,
        //         beta=0.0,  y=XTy[nP], incy=1)
        $XTy = new Tensor([$nP]);
        $blas->cblas_sgemv(
            101, 112,
            $nEff, $nP,
            1.0, $X->buffer, $nP, $yT->buffer, 1,
            0.0, $XTy->buffer, 1
        );

        // ── Ridge regularisation: XTX += λI ───────────────────────────────
        //
        // λ = 1e-6 × trace(X^T X) / nP   (scale-invariant Tikhonov)
        $trace = 0.0;
        for ($i = 0; $i < $nP; $i++) {
            $trace += (float) $XTX->buffer[$i * $nP + $i];
        }
        $lambda = 1e-6 * $trace / $nP;
        for ($i = 0; $i < $nP; $i++) {
            $XTX->buffer[$i * $nP + $i] = (float) $XTX->buffer[$i * $nP + $i] + $lambda;
        }

        // ── Convert to PHP arrays and solve with LinAlg::solve ─────────────
        $A = [];
        $b = [];
        for ($i = 0; $i < $nP; $i++) {
            $A[$i] = [];
            for ($j = 0; $j < $nP; $j++) {
                $A[$i][$j] = (float) $XTX->buffer[$i * $nP + $j];
            }
            $b[$i] = [(float) $XTy->buffer[$i]];
        }

        $betaMat = LinAlg::solve($A, $b, $nP, 1);
        $beta    = array_column($betaMat, 0);   // flat solution vector

        // ── Extract AR, MA, const from $beta ──────────────────────────────
        $arCoeffs = array_slice($beta, 0, $nAR);
        $maCoeffs = array_slice($beta, $nAR, $nMA);
        $constant = $includeConst ? $beta[$nP - 1] : 0.0;

        // ── Compute in-sample residuals ────────────────────────────────────
        //
        // Use the fitted β to re-compute residuals.  Unlike the long-AR
        // innovations used as regressors, these residuals feed back into the
        // MA columns for a single refinement pass (Hannan-Rissanen step 3).
        // We do one pass: residuals are computed sequentially, each ε̂_t
        // immediately available as the MA regressor for the next row.
        $residuals = array_fill(0, $n, 0.0);

        for ($t = $startT; $t < $n; $t++) {
            $pred = $constant;
            for ($k = 0; $k < $nAR; $k++) {
                $pred += $arCoeffs[$k] * $w[$t - $arLags[$k]];
            }
            for ($k = 0; $k < $nMA; $k++) {
                $pred += $maCoeffs[$k] * $residuals[$t - $maLags[$k]];
            }
            $residuals[$t] = $w[$t] - $pred;
        }

        // Return only the effective portion
        $residualsEff = array_slice($residuals, $startT);

        return [$arCoeffs, $maCoeffs, $constant, $residualsEff];
    }

    /**
     * Apply one order of (regular) differencing: z_t = y_t − y_{t-1}.
     * Returns an array of length n-1.
     *
     * @param  float[] $y
     * @return float[]
     */
    public static function diff1(array $y): array
    {
        $n   = count($y);
        $out = [];
        for ($i = 1; $i < $n; $i++) {
            $out[] = $y[$i] - $y[$i - 1];
        }
        return $out;
    }

    /**
     * Apply d orders of regular differencing.
     *
     * @param  float[] $y
     * @param  int     $d
     * @return float[]
     */
    public static function diffN(array $y, int $d): array
    {
        for ($i = 0; $i < $d; $i++) {
            $y = self::diff1($y);
        }
        return $y;
    }

    /**
     * Integrate (un-difference) forecast values using stored boundary values.
     *
     * Algorithm: for i = d-1 down to 0, apply a cumulative sum starting from
     * $this->undiffTails_[$i] (the last observed value of the i-th difference
     * series) to the current forecast array.
     *
     * @param  float[] $wFcst  Forecasts on the d-times-differenced scale.
     * @return float[]         Forecasts in the original (un-differenced) scale.
     */
    private function undiff(array $wFcst): array
    {
        $current = $wFcst;

        for ($i = $this->d - 1; $i >= 0; $i--) {
            $last   = $this->undiffTails_[$i];
            $result = [];
            foreach ($current as $delta) {
                $last     += $delta;
                $result[] = $last;
            }
            $current = $result;
        }

        return $current;
    }

    private function checkFitted(): void
    {
        if (!isset($this->ar_params_)) {
            throw new \RuntimeException('ARIMA is not fitted. Call fit() first.');
        }
    }
}
