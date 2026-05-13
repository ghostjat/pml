<?php
declare(strict_types=1);
/**
 * HOUSE PRICE PREDICTION — Real Estate AVM
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Predict residential property sale price (regression).
 * Model    : GBDTRegressor — state-of-the-art for tabular regression,
 *            handles nonlinear interactions and mixed feature types.
 * Business : Banks use Automated Valuation Models (AVMs) to appraise
 *            collateral in seconds. Zillow, Redfin, and mortgage
 *            lenders all run models like this at millions of req/day.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Estimators\Regression\GBDTRegressor;
use Pml\Metrics\Regression\RootMeanSquaredError;
use Pml\Metrics\Regression\RSquared;
use Pml\Metrics\Regression\MeanAbsoluteError;

section('House Price AVM — GBDT Regressor');

// ── 1. Property dataset ───────────────────────────────────────────────────────
// Features: sqft, bedrooms, bathrooms, lot_size, age_years,
//           garage (0/1), pool (0/1), school_rating (1-10),
//           distance_downtown_km, neighborhood_score (1-10)

mt_srand(7);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

$rows = [];
$lbls = [];

for ($i = 0; $i < 6000; $i++) {
    $sqft       = $rng(600, 4500);
    $beds       = max(1, (int)$rng(1, 6));
    $baths      = max(1, (int)$rng(1, 4));
    $lot        = $rng(2000, 20000);
    $age        = $rng(0, 80);
    $garage     = mt_rand(0, 1);
    $pool       = (mt_rand(0, 4) === 0) ? 1 : 0;
    $school     = $rng(3, 10);
    $downtown   = $rng(1, 30);
    $nbhd       = $rng(4, 10);

    // Realistic price model (USD)
    $price = 50000
        + $sqft       * $rng(80, 180)
        + $beds       * 8000
        + $baths      * 12000
        + $lot        * 2.5
        - $age        * 800
        + $garage     * 15000
        + $pool       * 25000
        + $school     * 18000
        - $downtown   * 3500
        + $nbhd       * 12000
        + $rng(-30000, 30000);   // market noise

    $rows[] = [(float)$sqft, (float)$beds, (float)$baths, $lot, $age,
               (float)$garage, (float)$pool, $school, $downtown, $nbhd];
    $lbls[] = max(50000.0, $price);
}

$dataset = Dataset::fromArray($rows, $lbls);
[$train, $test] = $dataset->randomize()->split(0.8);

metric('Properties (train)', $train->numRows());
metric('Properties (test)',  $test->numRows());
metric('Median price',       '$' . number_format((float)array_sum(array_slice($lbls, 0, 100)) / 100, 0));

// ── 2. Train ──────────────────────────────────────────────────────────────────
section('Training');
$t0 = microtime(true);

$model = new GBDTRegressor(nEstimators: 400, maxDepth: 6, lr: 0.05, lambda: 1.0);
$model->train($train);

metric('Training time', elapsed($t0));

// ── 3. Evaluate ───────────────────────────────────────────────────────────────
section('Evaluation');
$pred   = $model->predict($test);
$labels = $test->labels();

$rmse = (new RootMeanSquaredError())->score($pred, $labels);
$mae  = (new MeanAbsoluteError())->score($pred, $labels);
$r2   = (new RSquared())->score($pred, $labels);

metric('RMSE',     '$' . number_format($rmse, 0));
metric('MAE',      '$' . number_format($mae, 0));
metric('R²',       $r2);
metric('MAPE est', round($mae / (array_sum($lbls) / count($lbls)) * 100, 2), '%');

// ── 4. Instant property appraisal ─────────────────────────────────────────────
section('Instant Appraisals');

$listings = [
    'Downtown studio'   => [680, 1, 1, 3200, 5,  0, 0, 7.5, 2.0, 8.5],
    'Suburban 4-bed'    => [2400, 4, 3, 8500, 12, 1, 1, 8.2, 12.0, 7.0],
    'Rural farmhouse'   => [1800, 3, 2, 18000, 45, 1, 0, 5.0, 28.0, 5.5],
];

$inputRows = array_values($listings);
$batch     = Dataset::fromArray($inputRows);
$prices    = $model->predict($batch)->toFlatArray();

foreach (array_keys($listings) as $k => $name) {
    printf("  %-22s → AVM: $%s\n", $name, number_format($prices[$k], 0));
}

echo "\n✓ Done\n";

/*
 * PRODUCTION NOTES
 * ────────────────
 * • Retrain quarterly on MLS transaction data.
 * • Add geospatial features (census tract, walk score, flood zone).
 * • Log prediction intervals using quantile regression for confidence bands.
 * • Serve via REST: POST /appraise → JSON with price + confidence_interval.
 */
