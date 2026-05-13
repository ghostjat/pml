<?php
declare(strict_types=1);
/**
 * CRYPTOCURRENCY PRICE FORECASTING
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Forecast next-day BTC/USD closing price direction
 *            (up / down) using technical indicators as features.
 * Model    : GBDTClassifier — handles non-stationarity via lag
 *            features, no distributional assumptions, fast retrain.
 * Business : A directional accuracy of 55 %+ is statistically
 *            significant and tradeable with proper risk management.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Estimators\Classifiers\GBDTClassifier;
use Pml\Metrics\Classification\Accuracy;
use Pml\Metrics\Classification\RocAuc;

section('Crypto Price Direction Forecasting — GBDT');

// ── 1. Simulate daily BTC OHLCV price history ─────────────────────────────────
mt_srand(77);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

$nDays = 800;
$prices = [30000.0];
$volumes = [];
for ($i = 1; $i < $nDays; $i++) {
    $ret      = $rng(-0.06, 0.06) + 0.0002;   // slight upward drift
    $prices[] = max(1000.0, end($prices) * (1 + $ret));
    $volumes[] = $rng(1e9, 8e9);
}
$volumes[] = $rng(1e9, 8e9);

// ── 2. Build feature matrix from technical indicators ─────────────────────────
// Features: ret_1d, ret_3d, ret_7d, ret_14d, volatility_7d,
//           rsi_14, macd_signal, volume_ratio, price_vs_sma20,
//           price_vs_sma50, bb_position (Bollinger Band)

$rows = [];
$lbls = [];
$window = 50;  // need 50 days of history for longest indicator

for ($i = $window; $i < $nDays - 1; $i++) {
    $p  = $prices[$i];
    $p1 = $prices[$i - 1];
    $p3 = $prices[$i - 3];
    $p7 = $prices[$i - 7];
    $p14= $prices[$i - 14];

    // Returns
    $ret1  = ($p - $p1) / $p1;
    $ret3  = ($p - $p3) / $p3;
    $ret7  = ($p - $p7) / $p7;
    $ret14 = ($p - $p14) / $p14;

    // 7-day volatility (std dev of returns)
    $rets7 = array_map(fn($j) => ($prices[$j] - $prices[$j-1]) / $prices[$j-1],
                       range($i-6, $i));
    $mean7 = array_sum($rets7) / 7;
    $vol7  = sqrt(array_sum(array_map(fn($r) => ($r - $mean7) ** 2, $rets7)) / 7);

    // RSI-14
    $gains = $losses = [];
    for ($j = $i-13; $j <= $i; $j++) {
        $d = $prices[$j] - $prices[$j-1];
        $d > 0 ? $gains[] = $d : $losses[] = abs($d);
    }
    $avgGain = $gains  ? array_sum($gains)  / 14 : 0.001;
    $avgLoss = $losses ? array_sum($losses) / 14 : 0.001;
    $rsi = 100 - 100 / (1 + $avgGain / $avgLoss);

    // MACD (12/26 EMA signal approximated with SMA ratio)
    $sma12 = array_sum(array_slice($prices, $i-11, 12)) / 12;
    $sma26 = array_sum(array_slice($prices, $i-25, 26)) / 26;
    $macd  = ($sma12 - $sma26) / $sma26;

    // Volume ratio
    $avgVol = array_sum(array_slice($volumes, $i-9, 10)) / 10;
    $volRatio = $volumes[$i] / max(1, $avgVol);

    // Price vs SMA20 / SMA50
    $sma20 = array_sum(array_slice($prices, $i-19, 20)) / 20;
    $sma50 = array_sum(array_slice($prices, $i-49, 50)) / 50;
    $vs20  = ($p - $sma20) / $sma20;
    $vs50  = ($p - $sma50) / $sma50;

    // Bollinger Band position
    $sma20std = sqrt(array_sum(array_map(
        fn($j) => ($prices[$j] - $sma20) ** 2,
        range($i-19, $i)
    )) / 20);
    $bbPos = $sma20std > 0 ? ($p - ($sma20 - 2 * $sma20std)) / (4 * $sma20std) : 0.5;

    $rows[] = [$ret1, $ret3, $ret7, $ret14, $vol7,
               $rsi / 100, $macd, $volRatio, $vs20, $vs50, $bbPos];

    // Label: 1 if tomorrow is up, 0 if down
    $lbls[] = $prices[$i + 1] > $p ? 1.0 : 0.0;
}

// Walk-forward split: train on 80 %, test on most recent 20 %
$cutoff  = (int)(count($rows) * 0.8);
$trainDs = Dataset::fromArray(array_slice($rows, 0, $cutoff), array_slice($lbls, 0, $cutoff));
$testDs  = Dataset::fromArray(array_slice($rows, $cutoff),    array_slice($lbls, $cutoff));

metric('Training days', $trainDs->numRows());
metric('Test days',     $testDs->numRows());

// ── 3. Train ──────────────────────────────────────────────────────────────────
section('Training');
$t0 = microtime(true);

$model = new GBDTClassifier(nEstimators: 300, maxDepth: 4, lr: 0.05, lambda: 2.0);
$model->train($trainDs);

metric('Training time', elapsed($t0));

// ── 4. Evaluate ───────────────────────────────────────────────────────────────
section('Evaluation');
$pred   = $model->predict($testDs);
$labels = $testDs->labels();

$acc = (new Accuracy())->score($pred, $labels);
$auc = (new RocAuc())->score($pred, $labels);

metric('Directional Accuracy', $acc);
metric('ROC-AUC',              $auc);

$baseline = max(
    array_sum($lbls) / count($lbls),
    1 - array_sum($lbls) / count($lbls)
);
metric('Naive baseline (majority class)', round($baseline, 4));
metric('Edge over baseline', round(($acc - $baseline) * 100, 2), ' pp');

// ── 5. Live signal ────────────────────────────────────────────────────────────
section('Trading Signal');
$lastRow   = $rows[count($rows) - 1];
$lastPrice = $prices[count($prices) - 1];
$signal    = $model->proba(Dataset::fromArray([$lastRow]))->toFlatArray()[0] ?? 0.5;

printf("  Latest BTC price  : $%s\n", number_format($lastPrice, 2));
printf("  P(up tomorrow)    : %.1f%%\n", $signal * 100);
printf("  Signal            : %s\n", $signal > 0.55 ? '📈 BUY  / HOLD' : ($signal < 0.45 ? '📉 SELL / SHORT' : '⏸️  NEUTRAL'));

echo "\n✓ Done\n";

/*
 * IMPORTANT DISCLAIMER
 * Past price patterns do not guarantee future returns.
 * This model is for educational purposes — apply proper
 * risk management, position sizing, and out-of-sample testing
 * before using any signal in production trading.
 */
