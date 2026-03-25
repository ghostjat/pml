<?php

declare(strict_types=1);

/**
 * ════════════════════════════════════════════════════════════════════════════
 *  examples/ts_sarima_forecasting.php — ARIMA & SARIMA Forecasting Demo
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Demonstrates the Box-Jenkins workflow:
 *
 *   1. Generate a synthetic monthly series with trend + seasonality (12-month)
 *      plus AR(1) autocorrelation — an "airline passenger"-like series.
 *
 *   2. Fit ARIMA(1,1,1) on the non-seasonal version (first 60 months, no
 *      seasonal component) and show in-sample RMSE + 6-step forecast.
 *
 *   3. Fit SARIMA(1,1,1)(1,1,1,12) on the full 120-month series and produce
 *      a 12-month seasonal forecast with ASCII chart.
 *
 *   4. Print model summaries (coefficients, AIC, BIC).
 *
 * ── Synthetic Series Design ───────────────────────────────────────────────
 *
 *  The series is built as:
 *
 *    log(y_t) = 5.0                       (level ≈ 148)
 *             + 0.005 · t                 (slow upward trend)
 *             + 0.30 · sin(2π t / 12)    (seasonal amplitude ≈ ±0.30 on log scale)
 *             + 0.20 · cos(2π t / 12)    (phase shift → peak in Aug/Sep)
 *             + AR(1) noise with φ = 0.6 (persistent autocorrelation)
 *             + ε_t ~ N(0, 0.05²)
 *
 *  Taking the log ensures positivity; the seasonal differencing in SARIMA
 *  handles the combined trend + seasonal non-stationarity.
 *
 * ── Why SARIMA for monthly data? ─────────────────────────────────────────
 *
 *  ∇ (regular diff, d=1)   removes the linear trend.
 *  ∇₁₂ (seasonal diff, D=1) removes the repeating 12-month seasonal pattern.
 *  AR(1)/MA(1) terms       capture remaining short-memory autocorrelation.
 *  Seasonal AR(1)/MA(1)    capture year-over-year autocorrelation.
 *
 * Usage:
 *   php examples/ts_sarima_forecasting.php
 * ════════════════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../vendor/autoload.php';

use Pml\Classic\TimeSeries\{ARIMA, SARIMA};

// ─── 1. Generate synthetic monthly time series ────────────────────────────
//
//  We use a Box-Muller Gaussian for the innovations and AR(1) recursion.

mt_srand(42);   // reproducible

/**
 * Box-Muller: one N(0,1) variate.
 */
function randn_scalar(): float
{
    static $spare = null;
    static $hasSpare = false;
    if ($hasSpare) {
        $hasSpare = false;
        return $spare;
    }
    do { $u = mt_rand() / mt_getrandmax(); } while ($u === 0.0);
    $v        = mt_rand() / mt_getrandmax();
    $mag      = sqrt(-2.0 * log($u));
    $spare    = $mag * sin(2.0 * M_PI * $v);
    $hasSpare = true;
    return $mag * cos(2.0 * M_PI * $v);
}

$N     = 132;   // 11 years of monthly data; last 12 held out for evaluation
$logY  = [];
$ar1   = 0.0;   // AR(1) state

for ($t = 0; $t < $N; $t++) {
    $trend      = 5.0 + 0.005 * $t;
    $seasonal   = 0.30 * sin(2.0 * M_PI * $t / 12)
                + 0.20 * cos(2.0 * M_PI * $t / 12);
    $ar1        = 0.60 * $ar1 + 0.05 * randn_scalar();
    $logY[]     = $trend + $seasonal + $ar1;
}

// Back-transform to levels
$y = array_map('exp', $logY);

// Train/test split: train on first 120, hold out last 12 for SARIMA evaluation
$nTrain = 120;
$yTrain = array_slice($y, 0, $nTrain);
$yTest  = array_slice($y, $nTrain, 12);

echo "\n";
echo "════════════════════════════════════════════════════════════\n";
echo "  Time Series: ARIMA & SARIMA Forecasting Demo\n";
echo "  Series length : {$N} months  (train={$nTrain}, test=12)\n";
echo sprintf("  Level range   : [%.1f, %.1f]\n", min($y), max($y));
echo "════════════════════════════════════════════════════════════\n\n";

// ─── 2. ARIMA(1,1,1) on a de-seasonalised excerpt ────────────────────────
//
//  For a clean ARIMA demo, use the first 60 months of the LOG series
//  (the log removes multiplicative seasonality, but trend remains →
//  d=1 differencing handles it).

echo "── Part A: ARIMA(1,1,1) on first 60 months of log-series ──\n\n";

$logYTrain60 = array_slice($logY, 0, 60);

$arima = new ARIMA(p: 1, d: 1, q: 1, includeConst: true);
$arima->fit($logYTrain60);

echo $arima->summary();

// In-sample RMSE on differenced residuals
$residuals = $arima->residuals_;
$rmseResid = sqrt(array_sum(array_map(fn($e) => $e * $e, $residuals)) / count($residuals));
echo sprintf("  In-sample RMSE (differenced residuals) : %.6f\n\n", $rmseResid);

// 6-step-ahead forecast (on log scale)
$fcstLog = $arima->forecast(6);
echo "  6-step-ahead log-scale forecasts:\n";
foreach ($fcstLog as $k => $fv) {
    $actual = $logY[60 + $k] ?? null;
    $err    = $actual !== null ? sprintf('  actual=%.4f  err=%+.4f', $actual, $fv - $actual) : '';
    echo sprintf("    t+%-2d : %.4f%s\n", $k + 1, $fv, $err);
}

$mae60 = 0.0;
for ($k = 0; $k < 6; $k++) {
    $mae60 += abs($fcstLog[$k] - $logY[60 + $k]);
}
echo sprintf("  6-step MAE (log scale) : %.4f\n\n", $mae60 / 6);

// ─── 3. SARIMA(1,1,1)(1,1,1,12) on full 120-month level series ───────────
//
//  We work on the raw levels (not log).  The model handles:
//    d=1  → remove linear trend
//    D=1  → remove annual seasonality
//    AR/MA(1) + Seasonal AR/MA(1) → capture autocorrelation

echo "── Part B: SARIMA(1,1,1)(1,1,1,12) on 120-month level series ──\n\n";

$sarima = new SARIMA(p: 1, d: 1, q: 1, P: 1, D: 1, Q: 1, s: 12, includeConst: false);
$sarima->fit($yTrain);

echo $sarima->summary();

// 12-month-ahead forecast
$fcst12 = $sarima->forecast(12);

// Evaluation against held-out 12 months
$maeSarima = 0.0;
$rmseSarima = 0.0;
for ($k = 0; $k < 12; $k++) {
    $err        = $fcst12[$k] - $yTest[$k];
    $maeSarima += abs($err);
    $rmseSarima += $err * $err;
}
$maeSarima  /= 12;
$rmseSarima  = sqrt($rmseSarima / 12);

$meanActual = array_sum($yTest) / 12;
$mape       = 0.0;
for ($k = 0; $k < 12; $k++) {
    $mape += abs(($fcst12[$k] - $yTest[$k]) / max(1e-6, abs($yTest[$k])));
}
$mape = 100.0 * $mape / 12;

echo "  12-month forecast vs. held-out actuals:\n";
echo "  " . str_repeat('─', 56) . "\n";
echo sprintf("  %-6s  %-10s  %-10s  %-10s\n", 'Step', 'Forecast', 'Actual', 'Error');
echo "  " . str_repeat('─', 56) . "\n";

$months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
for ($k = 0; $k < 12; $k++) {
    echo sprintf("  %-6s  %-10.2f  %-10.2f  %+-.2f\n",
        $months[$k], $fcst12[$k], $yTest[$k], $fcst12[$k] - $yTest[$k]);
}

echo "  " . str_repeat('─', 56) . "\n";
echo sprintf("  MAE   : %.2f\n", $maeSarima);
echo sprintf("  RMSE  : %.2f\n", $rmseSarima);
echo sprintf("  MAPE  : %.1f%%\n\n", $mape);

// ─── 4. ASCII chart: last 24 observed + 12 forecast ──────────────────────

echo "── Part C: ASCII chart — last 24 observed + 12 forecast ──\n\n";

$chartObs  = array_slice($yTrain, -24);    // last 24 observed
$chartAll  = array_merge($chartObs, $fcst12);
$allVals   = $chartAll;

$chartMin  = min($allVals) * 0.97;
$chartMax  = max($allVals) * 1.03;
$chartH    = 16;
$chartW    = count($chartAll);   // 36 columns

// Map value to row (0=top)
$mapRow = function(float $v) use ($chartMin, $chartMax, $chartH): int {
    $frac = ($v - $chartMin) / max(1e-6, $chartMax - $chartMin);
    return (int) round((1.0 - $frac) * ($chartH - 1));
};

// Fill chart grid
$grid = [];
for ($r = 0; $r < $chartH; $r++) {
    $grid[$r] = array_fill(0, $chartW, ' ');
}

for ($col = 0; $col < $chartW; $col++) {
    $row  = $mapRow($chartAll[$col]);
    $row  = max(0, min($chartH - 1, $row));
    $char = ($col < 24) ? '●' : '◆';   // observed = ●, forecast = ◆
    $grid[$row][$col] = $char;
}

// Print with Y-axis labels
$step = ($chartMax - $chartMin) / ($chartH - 1);
for ($r = 0; $r < $chartH; $r++) {
    $label = $chartMax - $r * $step;
    echo sprintf(' %7.1f │%s', $label, implode('', $grid[$r])) . "\n";
}
echo '         └' . str_repeat('─', $chartW) . "\n";
echo '          ' . str_repeat(' ', 21) . "↑ forecast starts\n";
echo "\n  Legend: ● observed   ◆ forecast\n\n";

// ─── 5. ARIMA order selection hint ───────────────────────────────────────

echo "── Part D: ARIMA order comparison (AIC) ──\n\n";
echo "  Fitting ARIMA(p,1,q) for p,q ∈ {0,1,2} on log-series (n=60):\n\n";
echo sprintf("  %-12s  %10s  %10s\n", 'Model', 'AIC', 'BIC');
echo '  ' . str_repeat('─', 36) . "\n";

$bestAIC   = INF;
$bestModel = '';

foreach ([0, 1, 2] as $p) {
    foreach ([0, 1, 2] as $q) {
        if ($p === 0 && $q === 0) { continue; }  // degenerate
        try {
            $m = new ARIMA(p: $p, d: 1, q: $q, includeConst: true);
            $m->fit($logYTrain60);
            $tag = "ARIMA({$p},1,{$q})";
            echo sprintf("  %-12s  %10.4f  %10.4f\n", $tag, $m->aic_, $m->bic_);
            if ($m->aic_ < $bestAIC) { $bestAIC = $m->aic_; $bestModel = $tag; }
        } catch (\Throwable $e) {
            echo sprintf("  %-12s  %10s\n", "ARIMA({$p},1,{$q})", 'error: ' . $e->getMessage());
        }
    }
}

echo '  ' . str_repeat('─', 36) . "\n";
echo sprintf("  Best by AIC: %s  (AIC=%.4f)\n\n", $bestModel, $bestAIC);
echo "════════════════════════════════════════════════════════════\n";
echo "  Done.\n";
echo "════════════════════════════════════════════════════════════\n\n";
