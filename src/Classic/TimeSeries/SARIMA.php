<?php

declare(strict_types=1);

namespace Pml\Classic\TimeSeries;

use Pml\{Tensor, BlasEngine, LinAlg};

// ═══════════════════════════════════════════════════════════════════════════
//  SARIMA — Seasonal ARIMA(p,d,q)(P,D,Q,s)
//
//  Extends ARIMA with a multiplicative seasonal component using the
//  Box-Jenkins seasonal differencing and polynomial multiplication approach.
//
//  ── Model Definition ──────────────────────────────────────────────────────
//
//  Let:
//    ∇^d  = (1 − B)^d       regular difference operator (lag polynomial)
//    ∇_s^D = (1 − B^s)^D   seasonal difference operator  (period s, order D)
//
//  After applying both operators to y_t:
//    w_t = ∇^d ∇_s^D y_t
//
//  The SARIMA model for w_t is:
//    Φ_P(B^s) · φ_p(B) · w_t = Θ_Q(B^s) · θ_q(B) · ε_t
//
//  where:
//    φ_p(B)  = 1 − φ_1 B − … − φ_p B^p         (non-seasonal AR polynomial)
//    Φ_P(B^s)= 1 − Φ_1 B^s − … − Φ_P B^{Ps}   (seasonal AR polynomial)
//    θ_q(B)  = 1 + θ_1 B + … + θ_q B^q         (non-seasonal MA polynomial)
//    Θ_Q(B^s)= 1 + Θ_1 B^s + … + Θ_Q B^{Qs}   (seasonal MA polynomial)
//
//  ── Polynomial Multiplication → Expanded Lag Structure ────────────────────
//
//  The product φ_p(B) · Φ_P(B^s) is computed via polynomial convolution,
//  yielding an AR polynomial of degree p + P·s with specific non-zero coefficients:
//
//  Example — SARIMA(1,1,1)(1,1,1,12):
//    ar_nonseasonal = [1, −φ_1]                        (lags 0, 1)
//    ar_seasonal    = [1, 0,…,0, −Φ_1]                 (lags 0, 12)
//    expanded       = [1, −φ_1, 0,…,0, −Φ_1, φ_1·Φ_1] (lags 0, 1, 12, 13)
//
//    Active AR lags: {1, 12, 13}  (coefficients at non-zero positions)
//    Similarly for MA.
//
//  NOTE: This implementation uses an "unconstrained" approximation — the
//  interaction lag (e.g., lag 13 in the example above) is estimated freely
//  rather than enforcing coeff[13] = coeff[1]·coeff[12].  This is the
//  Hannan-Rissanen "conditional least squares on the expanded lag structure"
//  approximation, which is both fast and practically accurate for moderate
//  sample sizes.  For exact multiplicative constraint estimation, full MLE
//  with nonlinear optimisation would be required.
//
//  ── Differencing Order ────────────────────────────────────────────────────
//
//  Applied in the order: seasonal first, then regular.
//    1. z^(1)_t = ∇_s^D y_t   (D seasonal differences)
//    2. w_t     = ∇^d z^(1)_t  (d regular differences on result of step 1)
//
//  Undifferencing for forecasts reverses this: regular first, then seasonal.
//
//  ── Seasonal Difference Operator ─────────────────────────────────────────
//
//  One seasonal difference of period s:
//    (∇_s y)_t = y_t − y_{t-s}
//
//  Applied D times.  The boundary tail needed for undifferencing (seasonal
//  integration) is the last s values of each intermediate seasonal-diff series.
//
//  ── Complexity ────────────────────────────────────────────────────────────
//
//  fit:      O(n·m) PHP long-AR + O(n·nparams) BLAS sgemm
//  forecast: O(h·(p+P+q+Q)) PHP
//
//  where nparams ≈ p + q + P + Q + 1 (active lag count from expanded polys).
// ═══════════════════════════════════════════════════════════════════════════

final class SARIMA
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Coefficients at each active AR lag, indexed to match $arLags_.
     * @var float[]
     */
    public readonly array $ar_params_;

    /**
     * Coefficients at each active MA lag, indexed to match $maLags_.
     * @var float[]
     */
    public readonly array $ma_params_;

    /** Constant/intercept term. */
    public readonly float $const_;

    /** Corrected residual variance σ̂². */
    public readonly float $sigma2_;

    /**
     * In-sample innovations on the fully-differenced series.
     * @var float[]
     */
    public readonly array $residuals_;

    /** Active AR lags (1-indexed) from the expanded polynomial. @var int[] */
    public readonly array $arLags_;

    /** Active MA lags (1-indexed) from the expanded polynomial. @var int[] */
    public readonly array $maLags_;

    /** AIC. */
    public readonly float $aic_;

    /** BIC. */
    public readonly float $bic_;

    /** Length of the original training series. */
    public readonly int $n_obs_;

    // ── Private forecast state ─────────────────────────────────────────────

    /**
     * Last max(maxARLag, maxMALag) values of the fully-differenced series.
     * @var float[]
     */
    private array $diffedTail_;

    /**
     * Last maxMALag residuals from the ARMA fit.
     * @var float[]
     */
    private array $residTail_;

    /**
     * Boundary values for un-doing the d regular differences.
     * $regularUndiffTails_[$i] = last value of the $i-th regular diff of the
     * seasonal-differenced series.
     * @var float[]
     */
    private array $regularUndiffTails_;

    /**
     * Boundary values for un-doing the D seasonal differences.
     * $seasonalUndiffTails_[$i] = last $s values of the $i-th seasonal diff
     * of the original series.
     * @var float[][]
     */
    private array $seasonalUndiffTails_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int  $p             Non-seasonal AR order
     * @param int  $d             Non-seasonal differencing order
     * @param int  $q             Non-seasonal MA order
     * @param int  $P             Seasonal AR order
     * @param int  $D             Seasonal differencing order
     * @param int  $Q             Seasonal MA order
     * @param int  $s             Seasonal period (e.g., 12 for monthly, 4 for quarterly)
     * @param bool $includeConst  Include an intercept.
     */
    public function __construct(
        private readonly int  $p            = 1,
        private readonly int  $d            = 1,
        private readonly int  $q            = 1,
        private readonly int  $P            = 1,
        private readonly int  $D            = 1,
        private readonly int  $Q            = 1,
        private readonly int  $s            = 12,
        private readonly bool $includeConst = false,
    ) {
        if ($p < 0 || $d < 0 || $q < 0 || $P < 0 || $D < 0 || $Q < 0) {
            throw new \InvalidArgumentException('SARIMA: all orders must be non-negative.');
        }
        if ($s < 2) {
            throw new \InvalidArgumentException('SARIMA: seasonal period s must be ≥ 2.');
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Fit the SARIMA model to a univariate time series.
     *
     * @param float[] $y  1-D array of observations. Must be at least
     *                    (D+1)·s + d + max(p, q, P·s, Q·s) + 20 long.
     */
    public function fit(array $y): static
    {
        $y = array_values($y);
        $n = count($y);

        $maxExpLag = $this->p + $this->P * $this->s;
        $minLen    = ($this->D + 1) * $this->s + $this->d + $maxExpLag + 20;
        if ($n < $minLen) {
            throw new \InvalidArgumentException(
                "SARIMA: series too short. Need ≥ {$minLen} observations, got {$n}."
            );
        }

        $this->n_obs_ = $n;

        // ── Step 1: Seasonal differencing (D times, period s) ─────────────
        //
        // Store the last $s values of each intermediate series for
        // undifferencing during forecast (seasonal integration).
        $seasonalTails = [];   // $seasonalTails[$i] = last $s values before diff $i
        $series        = $y;

        for ($i = 0; $i < $this->D; $i++) {
            // Store last s values before applying this seasonal diff
            $seasonalTails[] = array_slice($series, -$this->s);
            $series          = self::seasonalDiff1($series, $this->s);
        }

        // ── Step 2: Regular differencing (d times) ─────────────────────────
        //
        // Store boundary value of each intermediate diff series.
        $regularTails = [];

        for ($i = 0; $i < $this->d; $i++) {
            $regularTails[] = $series[count($series) - 1];
            $series         = ARIMA::diff1($series);
        }

        // $series is now the fully-differenced series w.
        $w = $series;

        // ── Step 3: Compute expanded AR and MA lag sets ────────────────────
        //
        // Expand the multiplicative SARIMA polynomial to find which lags are
        // structurally non-zero.  Use polynomial convolution:
        //
        //   AR_full(B) = φ_p(B) · Φ_P(B^s)
        //   MA_full(B) = θ_q(B) · Θ_Q(B^s)
        //
        // A lag l is "active" if the coefficient at position l of the full
        // polynomial is non-zero (i.e., the corresponding monomial participates).
        $arPoly   = self::buildSeasonalPoly($this->p, $this->P, $this->s, type: 'ar');
        $maPoly   = self::buildSeasonalPoly($this->q, $this->Q, $this->s, type: 'ma');

        // Collect active lags (non-zero positions, 1-indexed, excluding lag 0)
        $arLags = self::activeLags($arPoly);
        $maLags = self::activeLags($maPoly);

        // ── Step 4: Fit ARMA with expanded lags ───────────────────────────
        [$arCoeffs, $maCoeffs, $constant, $residuals] =
            ARIMA::fitARMA($w, max($arLags ?: [0]), max($maLags ?: [0]),
                $this->includeConst, $arLags, $maLags);

        // ── Step 5: Compute diagnostics ───────────────────────────────────
        $nEff    = count($residuals);
        $nparams = count($arLags) + count($maLags) + ($this->includeConst ? 1 : 0);
        $df      = max(1, $nEff - $nparams);

        $rss = 0.0;
        foreach ($residuals as $e) { $rss += $e * $e; }

        $sigma2  = $rss / $df;
        $logLik  = -0.5 * $nEff * (1.0 + log(2.0 * M_PI * max(1e-15, $sigma2)));
        $k       = $nparams + 1;

        // ── Step 6: Store fitted attributes ───────────────────────────────
        $this->ar_params_ = $arCoeffs;
        $this->ma_params_ = $maCoeffs;
        $this->const_     = $constant;
        $this->sigma2_    = $sigma2;
        $this->residuals_ = $residuals;
        $this->arLags_    = $arLags;
        $this->maLags_    = $maLags;
        $this->aic_       = 2.0 * $k - 2.0 * $logLik;
        $this->bic_       = $k * log($nEff) - 2.0 * $logLik;

        // ── Step 7: Store forecast seeds ──────────────────────────────────
        $maxLag            = max(max($arLags ?: [1]), max($maLags ?: [1]));
        $this->diffedTail_ = array_slice($w, -$maxLag);
        $this->residTail_  = ($maLags !== [])
            ? array_slice($residuals, -max($maLags))
            : [];
        $this->regularUndiffTails_  = $regularTails;
        $this->seasonalUndiffTails_ = $seasonalTails;

        return $this;
    }

    /**
     * Generate h-step-ahead point forecasts.
     *
     * @param  int     $h  Number of steps ahead (≥ 1).
     * @return float[]     Forecasts in the original un-differenced scale.
     */
    public function forecast(int $h = 1): array
    {
        if ($h < 1) {
            throw new \InvalidArgumentException('SARIMA::forecast(): h must be ≥ 1.');
        }

        $this->checkFitted();

        $arLags   = $this->arLags_;
        $maLags   = $this->maLags_;
        $arCoeffs = $this->ar_params_;
        $maCoeffs = $this->ma_params_;
        $c        = $this->const_;

        $wObs  = $this->diffedTail_;
        $eObs  = $this->residTail_;

        $maxObsLag = count($wObs);     // how many observed w values we have
        $maxELag   = count($eObs);     // how many observed residuals we have

        $wFcst = [];

        for ($k = 0; $k < $h; $k++) {
            $val = $c;

            // AR terms at each active lag
            foreach ($arLags as $idx => $lag) {
                $pos = $k - $lag;   // position in forecast array (0-based)
                if ($pos >= 0) {
                    $val += $arCoeffs[$idx] * $wFcst[$pos];
                } else {
                    // Observed: lag steps back from n = k+1 steps into observed
                    $obsIdx = $maxObsLag + $pos;   // $pos is negative here
                    $val += $arCoeffs[$idx] * ($wObs[$obsIdx] ?? 0.0);
                }
            }

            // MA terms at each active lag (future shocks = 0)
            foreach ($maLags as $idx => $lag) {
                if ($k < $lag) {
                    $obsIdx = $maxELag - ($lag - $k);
                    $val += $maCoeffs[$idx] * ($eObs[$obsIdx] ?? 0.0);
                }
                // k >= lag → future shock = 0
            }

            $wFcst[] = $val;
        }

        // ── Undifference: regular first, then seasonal ────────────────────
        $undone = $this->undiffRegular($wFcst);
        $undone = $this->undiffSeasonal($undone);

        return $undone;
    }

    /**
     * Return a text summary of the fitted model.
     */
    public function summary(): string
    {
        $this->checkFitted();

        $label = sprintf(
            'SARIMA(%d,%d,%d)(%d,%d,%d)[%d]%s',
            $this->p, $this->d, $this->q,
            $this->P, $this->D, $this->Q, $this->s,
            $this->includeConst ? ' + const' : ''
        );

        $lines = [
            str_repeat('═', 60),
            " {$label}",
            str_repeat('─', 60),
            sprintf(' n_obs   : %d   n_eff : %d',
                $this->n_obs_, count($this->residuals_)),
            sprintf(' σ²      : %.6f   √σ² : %.6f',
                $this->sigma2_, sqrt($this->sigma2_)),
            sprintf(' AIC     : %.4f', $this->aic_),
            sprintf(' BIC     : %.4f', $this->bic_),
            str_repeat('─', 60),
            ' Active AR lags: [' . implode(', ', $this->arLags_) . ']',
            ' Active MA lags: [' . implode(', ', $this->maLags_) . ']',
            str_repeat('─', 60),
        ];

        if ($this->includeConst) {
            $lines[] = sprintf(' const   : %+.6f', $this->const_);
        }
        foreach ($this->arLags_ as $i => $lag) {
            $lines[] = sprintf(' ar.L%-3d : %+.6f', $lag, $this->ar_params_[$i]);
        }
        foreach ($this->maLags_ as $i => $lag) {
            $lines[] = sprintf(' ma.L%-3d : %+.6f', $lag, $this->ma_params_[$i]);
        }
        $lines[] = str_repeat('═', 60);

        return implode("\n", $lines) . "\n";
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * Build the full seasonal AR or MA polynomial by convolving the
     * non-seasonal polynomial with the seasonal polynomial.
     *
     * Non-seasonal AR polynomial (lag-polynomial form, coefficient at index i = lag i):
     *   [1, φ_1, φ_2, ..., φ_p]   (positive values; the model subtracts them)
     *
     * Seasonal AR polynomial (non-zero at positions 0, s, 2s, ..., P·s):
     *   [1, 0, ..., 0, Φ_1, 0, ..., 0, Φ_P]
     *
     * After convolution the zero-th coefficient is always 1 (normalised).
     * We return the coefficient magnitudes at each lag > 0 that are non-zero.
     * The caller uses this to enumerate active lags.
     *
     * @param  int    $ord    Non-seasonal order (p or q)
     * @param  int    $Sord   Seasonal order (P or Q)
     * @param  int    $s      Seasonal period
     * @param  string $type   'ar' or 'ma' (same logic, different sign convention)
     * @return float[]        Coefficient array indexed 0..maxLag; index 0 = lag 0 (always 1).
     */
    private static function buildSeasonalPoly(int $ord, int $Sord, int $s, string $type): array
    {
        // Non-seasonal polynomial: [1, 1, 1, ..., 1] at positions 0..ord
        // We use 1.0 as a placeholder "this lag is active" marker;
        // actual values are estimated by OLS, not encoded in the polynomial structure.
        $nsPoly = array_fill(0, $ord + 1, 1.0);
        $nsPoly[0] = 1.0;  // constant term

        // Seasonal polynomial: non-zero at 0, s, 2s, ..., P*s
        $sPolyLen = $Sord * $s + 1;
        $sPoly    = array_fill(0, $sPolyLen, 0.0);
        $sPoly[0] = 1.0;
        for ($k = 1; $k <= $Sord; $k++) {
            $sPoly[$k * $s] = 1.0;   // placeholder
        }

        // Convolve: the positions where the product is non-zero tell us
        // which lags to include in the OLS design matrix.
        return self::polyConvolve($nsPoly, $sPoly);
    }

    /**
     * Given a polynomial coefficient array (index = lag), return the list of
     * lags > 0 where the coefficient is structurally non-zero.
     *
     * @param  float[] $poly   Coefficient array (index = lag).
     * @return int[]           1-indexed active lag positions.
     */
    private static function activeLags(array $poly): array
    {
        $lags = [];
        for ($i = 1; $i < count($poly); $i++) {
            if (abs($poly[$i]) > 1e-12) {
                $lags[] = $i;
            }
        }
        return $lags;
    }

    /**
     * Polynomial multiplication (convolution).
     * a[i] is the coefficient of B^i.
     *
     * @param  float[] $a
     * @param  float[] $b
     * @return float[]     Product polynomial of length |a|+|b|-1.
     */
    private static function polyConvolve(array $a, array $b): array
    {
        $la = count($a);
        $lb = count($b);
        $c  = array_fill(0, $la + $lb - 1, 0.0);

        for ($i = 0; $i < $la; $i++) {
            if ($a[$i] == 0.0) { continue; }
            for ($j = 0; $j < $lb; $j++) {
                $c[$i + $j] += $a[$i] * $b[$j];
            }
        }

        return $c;
    }

    /**
     * Apply one seasonal difference: z_t = y_t − y_{t-s}.
     *
     * @param  float[] $y
     * @param  int     $s  Seasonal period.
     * @return float[]     Length n-s.
     */
    private static function seasonalDiff1(array $y, int $s): array
    {
        $n   = count($y);
        $out = [];
        for ($i = $s; $i < $n; $i++) {
            $out[] = $y[$i] - $y[$i - $s];
        }
        return $out;
    }

    /**
     * Un-do regular differencing using stored boundary values.
     *
     * @param  float[] $wFcst  Forecasts on the d-times-regular-differenced scale.
     * @return float[]         After reversing d regular differences.
     */
    private function undiffRegular(array $wFcst): array
    {
        $current = $wFcst;

        for ($i = $this->d - 1; $i >= 0; $i--) {
            $last   = $this->regularUndiffTails_[$i];
            $result = [];
            foreach ($current as $delta) {
                $last     += $delta;
                $result[] = $last;
            }
            $current = $result;
        }

        return $current;
    }

    /**
     * Un-do seasonal differencing using stored boundary windows.
     *
     * For each reversal step i (from D-1 down to 0):
     *   y_{n+k} = y_{n+k-s} + z_{n+k}
     *
     * The "previous-s" values come from $this->seasonalUndiffTails_[$i]
     * (length s), which are the last s observations of the $i-th seasonal
     * difference series.  As we forecast further ahead than s steps, we
     * use already-computed forecast values.
     *
     * @param  float[] $zFcst  Forecasts on the D-times-seasonal-differenced scale.
     * @return float[]         Forecasts after reversing D seasonal differences.
     */
    private function undiffSeasonal(array $zFcst): array
    {
        $current = $zFcst;

        for ($i = $this->D - 1; $i >= 0; $i--) {
            $tail = $this->seasonalUndiffTails_[$i];   // last $s observed values
            $s    = $this->s;
            $h    = count($current);

            // Build a combined array: [$tail..., $result...] so we can always
            // look back $s positions using a single index.
            $combined = $tail;   // length $s

            for ($k = 0; $k < $h; $k++) {
                // y_{n+k} = y_{n+k-s} + z_{n+k}
                // Index (k) into combined = s + k
                // Index s steps back = k  (in combined array)
                $prev       = $combined[$k];
                $combined[] = $prev + $current[$k];
            }

            // Result is the appended portion
            $current = array_slice($combined, $s);
        }

        return $current;
    }

    private function checkFitted(): void
    {
        if (!isset($this->ar_params_)) {
            throw new \RuntimeException('SARIMA is not fitted. Call fit() first.');
        }
    }
}
