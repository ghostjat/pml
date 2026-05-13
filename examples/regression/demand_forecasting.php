<?php
declare(strict_types=1);
/**
 * RETAIL DEMAND FORECASTING
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Predict next-week unit sales per SKU per store.
 * Model    : GBDTRegressor with lag features and calendar signals.
 * Business : Over-stocking = working capital trapped in inventory.
 *            Under-stocking = lost sales + customer churn.
 *            A 10 % improvement in forecast accuracy for a retailer
 *            with $1 B inventory can free $80–120 M in cash.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Estimators\Regression\GBDTRegressor;
use Pml\Metrics\Regression\MeanAbsoluteError;
use Pml\Metrics\Regression\RootMeanSquaredError;
use Pml\Metrics\Regression\SMAPE;

section('Retail Demand Forecasting — GBDT');

// ── 1. Build time-series feature matrix ──────────────────────────────────────
// Features per (sku, store, week):
//   lag_1..lag_4 (previous 4 weeks sales), rolling_mean_4, rolling_std_4,
//   price, is_promoted, week_of_year, is_holiday_week, store_size_tier,
//   sku_category (encoded 0-4)

mt_srand(123);
$rng  = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);
$rows = [];
$lbls = [];

$nSkus   = 20;
$nStores = 10;
$nWeeks  = 60;   // 60 weeks of history per sku/store combo

for ($sku = 0; $sku < $nSkus; $sku++) {
    $baseVolume = $rng(50, 800);         // SKU popularity
    $category   = $sku % 5;

    for ($store = 0; $store < $nStores; $store++) {
        $storeMult = $rng(0.5, 2.0);     // store size effect
        $history   = [];

        // Seed 4 weeks of warm-up
        for ($w = 0; $w < 4; $w++) {
            $history[] = max(0.0, $baseVolume * $storeMult * (0.8 + $rng(0, 0.4)));
        }

        for ($w = 4; $w < $nWeeks; $w++) {
            $lag1   = $history[$w - 1];
            $lag2   = $history[$w - 2];
            $lag3   = $history[$w - 3];
            $lag4   = $history[$w - 4];
            $mean4  = ($lag1 + $lag2 + $lag3 + $lag4) / 4;
            $std4   = sqrt((($lag1-$mean4)**2 + ($lag2-$mean4)**2 + ($lag3-$mean4)**2 + ($lag4-$mean4)**2) / 4);

            $isPromo    = (mt_rand(0, 5) === 0) ? 1 : 0;
            $isHoliday  = in_array($w % 52, [0, 51, 25, 26]) ? 1 : 0;
            $price      = $rng(5, 150);
            $weekOfYear = ($w % 52) + 1;

            // Seasonal + promo + noise
            $seasonal = 1.0 + 0.3 * sin(2 * M_PI * $weekOfYear / 52);
            $sales    = $mean4 * $seasonal
                * (1 + $isPromo * $rng(0.2, 0.5))
                * (1 + $isHoliday * $rng(0.1, 0.3))
                * $rng(0.85, 1.15);
            $sales = max(0.0, round($sales));

            $rows[] = [$lag1, $lag2, $lag3, $lag4, $mean4, $std4,
                       $price, (float)$isPromo, (float)$weekOfYear,
                       (float)$isHoliday, $storeMult, (float)$category];
            $lbls[] = $sales;
            $history[] = $sales;
        }
    }
}

// Train on first 80 % of timeline, test on last 20 % (walk-forward)
$cutoff  = (int)(count($rows) * 0.8);
$trainDs = Dataset::fromArray(array_slice($rows, 0, $cutoff),       array_slice($lbls, 0, $cutoff));
$testDs  = Dataset::fromArray(array_slice($rows, $cutoff), array_slice($lbls, $cutoff));

metric('Train weeks', $trainDs->numRows());
metric('Test weeks',  $testDs->numRows());

// ── 2. Train ──────────────────────────────────────────────────────────────────
section('Training');
$t0 = microtime(true);

$model = new GBDTRegressor(nEstimators: 300, maxDepth: 6, lr: 0.05);
$model->train($trainDs);

metric('Training time', elapsed($t0));

// ── 3. Evaluate ───────────────────────────────────────────────────────────────
section('Evaluation');
$pred   = $model->predict($testDs);
$labels = $testDs->labels();

metric('MAE',   (new MeanAbsoluteError())->score($pred, $labels),     ' units');
metric('RMSE',  (new RootMeanSquaredError())->score($pred, $labels),  ' units');
metric('SMAPE', (new SMAPE())->score($pred, $labels), '%');

// ── 4. Next-week forecast for a single SKU/store ──────────────────────────────
section('Next-Week Forecast Sample');
$sampleHistory = [320, 290, 310, 340];
$mean4  = array_sum($sampleHistory) / 4;
$var    = array_sum(array_map(fn($x) => ($x - $mean4) ** 2, $sampleHistory)) / 4;
$sample = Dataset::fromArray([[
    $sampleHistory[3], $sampleHistory[2], $sampleHistory[1], $sampleHistory[0],
    $mean4, sqrt($var),
    29.99, 1.0, 47.0, 0.0, 1.5, 2.0
]]);
$forecast = $model->predict($sample)->toFlatArray()[0];
printf("  History (last 4w): %s\n", implode(', ', $sampleHistory));
printf("  Forecast next week: %.0f units\n", $forecast);
printf("  Recommended reorder: %.0f units\n", $forecast * 1.15);  // 15 % safety stock

echo "\n✓ Done\n";
