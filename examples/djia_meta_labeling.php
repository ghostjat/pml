<?php

declare(strict_types=1);

/**
 * examples/djia_meta_labeling.php
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * TIER-1 HEDGE FUND ARCHITECTURE: DUAL-CLASSIFIER META-LABELING
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Two-Stage Architecture:
 *
 * PRIMARY MODEL  (The Strategist)
 * ────────────────────────────────
 * Trained on 70% of data.
 * A classification model that predicts if the 10-day forward return 
 * will be strictly positive (1.0 = Up, 0.0 = Down/Flat).
 * Pipeline: StandardScaler → XGBClassifier
 *
 * META-MODEL  (The Risk Manager)
 * ────────────────────────────────
 * Trained on the next 15% of data (the "meta-train" window).
 * Receives an AUGMENTED feature set: original indicators + the Primary
 * Model's PREDICTION PROBABILITY as an extra column.
 * Its target is a binary META-LABEL:
 * 1 = "Primary Model was correct (Hard guess matched actual target)"
 * 0 = "Primary Model was wrong"
 * Pipeline: SimpleImputer(median) → StandardScaler → XGBClassifier
 *
 * INFERENCE  (The Execution Layer)
 * ─────────────────────────────────
 * Final 15% of data. Both models run in sequence:
 * 1. Primary predicts hard binary direction AND probability.
 * 2. Meta-Model predicts P(primary is correct) via predict_proba().
 * 3. ONLY execute the trade if P(correct) > 0.60 (60% confidence gate).
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../vendor/autoload.php';

use Pml\Classic\Exploration\DataProfiler;
use Pml\Classic\Preprocess\StandardScaler;
use Pml\Classic\Impute\SimpleImputer;
use Pml\Classic\Ensemble\XGBClassifier;
use Pml\Classic\Pipeline\Pipeline;
use Pml\Tensor;

// ── Utility ──────────────────────────────────────────────────────────────────

function banner(string $title, string $phase): void
{
    $line = str_repeat('═', 72);
    echo "\n{$line}\n  {$phase}: {$title}\n{$line}\n";
}

function loadTsv(string $path): array
{
    $fh = fopen($path, 'r');
    if ($fh === false) {
        throw new \RuntimeException("Cannot open: {$path}");
    }

    $headers = [];
    $rows    = [];
    $first   = true;
    
    while (($row = fgetcsv($fh, 0, "\t")) !== false) {
        if ($row === [null]) continue;
        $row = array_map('trim', $row);
        if ($first) {
            $headers = $row;
            $first   = false;
        } else {
            $rows[] = $row;
        }
    }
    fclose($fh);
    return ['data' => $rows, 'feature_names' => $headers];
}

// ─────────────────────────────────────────────────────────────────────────────
//  PHASE 1 — Ingestion, EDA & Cross-Asset Feature Engineering
// ─────────────────────────────────────────────────────────────────────────────

banner('Ingestion, EDA & Feature Engineering', 'Phase 1');

$csvPath = $argv[1] ?? __DIR__ . '/../datasets/stocks/tcs.csv';

if (!file_exists($csvPath)) {
    die("Error: Dataset not found at {$csvPath}\n");
}

$dataset  = loadTsv($csvPath);
$raw_data = $dataset['data'];

echo "  Dataset      : {$csvPath}\n";
echo "  Total rows   : " . count($raw_data) . "\n\n";

// ── Load Market Index & Build O(1) Date → Close Hash Map ─────────────────────
$marketCsvPath = __DIR__ . '/../datasets/stocks/nifty50.csv';
$market_closes = [];

if (file_exists($marketCsvPath)) {
    $marketDataset = loadTsv($marketCsvPath);
    foreach ($marketDataset['data'] as $row) {
        $dateKey  = trim((string) $row[0]);
        $closeVal = (float) $row[4];
        if ($dateKey !== '' && $closeVal > 0.0) {
            $market_closes[$dateKey] = $closeVal;
        }
    }
    printf("  Market index : %s  (%d trading days indexed)\n", $marketCsvPath, count($market_closes));
} else {
    echo "  [WARNING] nifty50.csv not found — market return defaults to 0.0 for all rows.\n";
}

// ── Institutional Feature Explosion ──────────────────────────────────────────

$engineered_X = [];
$target_y     = [];
$n_raw        = count($raw_data);

echo "\n  Engineering features (10-Day Horizon | RVOL | OBV | Trend Distances)...\n";

// Start at day 200 to allow for the 200-day SMA, end 10 days early for the target
for ($i = 200; $i < $n_raw - 10; $i++) {

    $close   = (float) $raw_data[$i][4];
    $vol     = (float) $raw_data[$i][5];
    $volPrev = (float) $raw_data[$i - 1][5];

    if ($close <= 0.0 || $volPrev <= 0.0) {
        continue;
    }

    // F1: 14-Day RSI Proxy
    $upSum = 0.0; $downSum = 0.0; $upCnt = 0; $downCnt = 0;
    for ($k = 0; $k < 14; $k++) {
        $c1 = (float) $raw_data[$i - $k][4];
        $c0 = (float) $raw_data[$i - $k - 1][4];
        if ($c0 <= 0.0) continue;
        $delta = $c1 - $c0;
        if ($delta > 0.0) { $upSum += $delta; $upCnt++; } 
        elseif ($delta < 0.0) { $downSum += abs($delta); $downCnt++; }
    }
    $avgUp   = ($upCnt > 0) ? $upSum / $upCnt : 0.0;
    $avgDown = ($downCnt > 0) ? $downSum / $downCnt : 0.0;
    $rsiProxy = ($avgDown < 1e-12) ? 100.0 : 100.0 - (100.0 / (1.0 + ($avgUp / $avgDown)));

    // F2: 10-Day Volatility
    $returns10 = []; $rMean = 0.0; $volDegenerate = false;
    for ($k = 0; $k < 10; $k++) {
        $c1 = (float) $raw_data[$i - $k][4];
        $c0 = (float) $raw_data[$i - $k - 1][4];
        if ($c0 <= 0.0) { $volDegenerate = true; break; }
        $r = ($c1 - $c0) / $c0;
        $returns10[] = $r;
        $rMean += $r;
    }
    if ($volDegenerate || count($returns10) < 10) {
        $vol10d = NAN;
    } else {
        $rMean /= 10.0; $variance = 0.0;
        foreach ($returns10 as $r) $variance += ($r - $rMean) ** 2;
        $vol10d = sqrt($variance / 10.0);
    }

    // F3: Volume Momentum (1 Day)
    $volumeMomentum = ($vol - $volPrev) / $volPrev;

    // F4: True Alpha (10-Day vs Market)
    $closeMinus10 = (float) $raw_data[$i - 10][4];
    $dateToday    = trim((string) $raw_data[$i][0]);
    $datePast10   = trim((string) $raw_data[$i - 10][0]);

    if ($closeMinus10 <= 0.0) {
        $relStrength = NAN;
    } else {
        $stock10d = ($close - $closeMinus10) / $closeMinus10;
        $market10d = (isset($market_closes[$dateToday], $market_closes[$datePast10]) && $market_closes[$datePast10] > 0.0)
            ? ($market_closes[$dateToday] - $market_closes[$datePast10]) / $market_closes[$datePast10]
            : 0.0;
        $relStrength = $stock10d - $market10d;
    }

    // F5: RVOL (Relative Volume - 20 Day)
    $volSum20 = 0.0;
    for ($k = 0; $k < 20; $k++) $volSum20 += (float) $raw_data[$i - $k][5];
    $avgVol20 = $volSum20 / 20.0;
    $rvol = ($avgVol20 > 0) ? $vol / $avgVol20 : 1.0;

    // F6: On-Balance Volume Proxy (10 Day)
    $obvSum = 0.0; $volSum10 = 0.0;
    for ($k = 0; $k < 10; $k++) {
        $v = (float) $raw_data[$i - $k][5];
        $volSum10 += $v;
        $c0 = (float) $raw_data[$i - $k - 1][4];
        $c1 = (float) $raw_data[$i - $k][4];
        if ($c1 > $c0) $obvSum += $v;
        elseif ($c1 < $c0) $obvSum -= $v;
    }
    $obvProxy = ($volSum10 > 0) ? $obvSum / $volSum10 : 0.0;

    // F7 & F8: Trend Distances (50 & 200 Day SMA)
    $sum50 = 0.0; for ($k = 0; $k < 50; $k++) $sum50 += (float) $raw_data[$i - $k][4];
    $trendDist50 = $close / ($sum50 / 50.0);

    $sum200 = 0.0; for ($k = 0; $k < 200; $k++) $sum200 += (float) $raw_data[$i - $k][4];
    $trendDist200 = $close / ($sum200 / 200.0);

    // F9: Bollinger Band Width (20 Day)
    $sum20 = 0.0; $high20 = -1.0; $low20 = 9999999.0;
    for ($k = 0; $k < 20; $k++) {
        $c = (float) $raw_data[$i - $k][4];
        $h = (float) $raw_data[$i - $k][2];
        $l = (float) $raw_data[$i - $k][3];
        $sum20 += $c;
        if ($h > $high20) $high20 = $h;
        if ($l < $low20) $low20 = $l;
    }
    $sma20 = $sum20 / 20.0;
    $bbWidth = ($sma20 > 0) ? ($high20 - $low20) / $sma20 : 0.0;

    // TARGET: 10-Day Forward Return (Binary)
    $closeFuture   = (float) $raw_data[$i + 10][4];
    $forwardReturn = ($closeFuture - $close) / $close;
    $binaryTarget  = ($forwardReturn > 0.0) ? 1.0 : 0.0;

    $engineered_X[] = [
        $rsiProxy, 
        $vol10d, 
        $volumeMomentum, 
        $relStrength, 
        $rvol, 
        $obvProxy, 
        $trendDist50, 
        $trendDist200, 
        $bbWidth
    ];
    $target_y[] = $binaryTarget;
}

$num_samples  = count($engineered_X);
$num_features = count($engineered_X[0]);

printf(
    "  Engineered matrix : [%d samples × %d features]\n  Target Format     : Binary Classification (1.0 = Up, 0.0 = Down)\n",
    $num_samples,
    $num_features
);


// ─────────────────────────────────────────────────────────────────────────────
//  PHASE 2 — Sequential Tri-Split & Primary Model
// ─────────────────────────────────────────────────────────────────────────────

banner('Sequential Tri-Split & Primary Model (Direction)', 'Phase 2');

$n_primary = (int) ($num_samples * 0.70);
$n_meta    = (int) ($num_samples * 0.85);
$n_test    = $num_samples - $n_meta;

printf("  Total : %d | Primary train: %d | Meta train: %d | Test: %d\n", $num_samples, $n_primary, $n_meta - $n_primary, $n_test);

$X_primary_arr = array_slice($engineered_X, 0, $n_primary);
$y_primary_arr = array_slice($target_y,     0, $n_primary);

$X_meta_arr    = array_slice($engineered_X, $n_primary, $n_meta - $n_primary);
$y_meta_arr    = array_slice($target_y,     $n_primary, $n_meta - $n_primary);

$X_test_arr    = array_slice($engineered_X, $n_meta);
$y_test_arr    = array_slice($target_y,     $n_meta);

$X_train_primary = Tensor::fromArray($X_primary_arr, [$n_primary, $num_features]);
$y_train_primary = Tensor::fromArray($y_primary_arr, [$n_primary]);

$primaryPipeline = new Pipeline([
    ['scaler', new StandardScaler()],
    ['xgb',   new XGBClassifier(
        n_estimators:  100,      // Heavily Regularized
        max_depth:     2,        // Shallow trees
        learning_rate: 0.05,
        subsample:     0.50,     // 50% bagging
        random_state:  42,
    )],
]);

echo "  Fitting Primary Pipeline (StandardScaler → XGBClassifier)...\n";
$primaryPipeline->fit($X_train_primary, $y_train_primary);


// ─────────────────────────────────────────────────────────────────────────────
//  PHASE 3 — Meta-Label Construction & Meta-Model Training
// ─────────────────────────────────────────────────────────────────────────────

banner('Meta-Label Construction & Meta-Model Training', 'Phase 3');

$n_meta_size   = $n_meta - $n_primary;
$X_meta_tensor = Tensor::fromArray($X_meta_arr, [$n_meta_size, $num_features]);
$y_meta_tensor = Tensor::fromArray($y_meta_arr, [$n_meta_size]);

echo "  Generating Primary Model predictions & probabilities on meta-train...\n";
$meta_primary_preds  = $primaryPipeline->predict($X_meta_tensor);
$meta_primary_probas = $primaryPipeline->predict_proba($X_meta_tensor);

$meta_labels = []; $metaCorrect = 0;

for ($i = 0; $i < $n_meta_size; $i++) {
    $actual       = (float) $y_meta_tensor->buffer[$i];
    $primaryGuess = (float) $meta_primary_preds->buffer[$i];

    $correct = ($actual === $primaryGuess);
    $meta_labels[] = $correct ? 1.0 : 0.0;
    if ($correct) $metaCorrect++;
}

printf("  Primary directional accuracy on meta-train : %.2f%%\n", ($metaCorrect / $n_meta_size) * 100.0);

$X_meta_augmented = [];
for ($i = 0; $i < $n_meta_size; $i++) {
    $row   = $X_meta_arr[$i];
    $row[] = (float) $meta_primary_probas->buffer[$i]; // Append probability
    $X_meta_augmented[] = $row;
}

$n_meta_features = $num_features + 1;
$X_meta_aug_tensor = Tensor::fromArray($X_meta_augmented, [$n_meta_size, $n_meta_features]);
$y_meta_labels     = Tensor::fromArray($meta_labels,      [$n_meta_size]);

$metaPipeline = new Pipeline([
    ['imputer', new SimpleImputer(strategy: 'median')],
    ['scaler',  new StandardScaler()],
    ['xgb',     new XGBClassifier(
        n_estimators:  50,       // Very Regularized
        max_depth:     2,
        learning_rate: 0.02,
        subsample:     0.50,
        random_state:  99,
    )],
]);

echo "\n  Fitting Meta Pipeline (SimpleImputer → StandardScaler → XGBClassifier)...\n";
$metaPipeline->fit($X_meta_aug_tensor, $y_meta_labels);


// ─────────────────────────────────────────────────────────────────────────────
//  PHASE 4 — Production Inference & Confidence-Gated Execution
// ─────────────────────────────────────────────────────────────────────────────

banner('Production Inference & Confidence-Gated Execution', 'Phase 4');

$X_test_tensor = Tensor::fromArray($X_test_arr, [$n_test, $num_features]);
$y_test_tensor = Tensor::fromArray($y_test_arr, [$n_test]);

echo "  Step A: Primary Model predicting on {$n_test} test samples...\n";
$test_primary_preds  = $primaryPipeline->predict($X_test_tensor);
$test_primary_probas = $primaryPipeline->predict_proba($X_test_tensor);

$X_test_augmented = [];
for ($i = 0; $i < $n_test; $i++) {
    $row   = $X_test_arr[$i];
    $row[] = (float) $test_primary_probas->buffer[$i];
    $X_test_augmented[] = $row;
}

$X_test_aug_tensor = Tensor::fromArray($X_test_augmented, [$n_test, $n_meta_features]);

echo "  Step B: Meta-Model computing confidence probabilities...\n";
$meta_probas = $metaPipeline->predict_proba($X_test_aug_tensor);

$META_CONFIDENCE_THRESHOLD = 0.60;

$tradesTaken = 0; $tradesCorrect = 0; $correctWithoutFilter = 0;

for ($i = 0; $i < $n_test; $i++) {
    $actual       = (float) $y_test_tensor->buffer[$i];
    $primaryGuess = (float) $test_primary_preds->buffer[$i];
    $confidence   = (float) $meta_probas->buffer[$i];

    $isPrimaryCorrect = ($actual === $primaryGuess);
    if ($isPrimaryCorrect) $correctWithoutFilter++;

    if ($confidence > $META_CONFIDENCE_THRESHOLD) {
        $tradesTaken++;
        if ($isPrimaryCorrect) $tradesCorrect++;
    }
}

$baselineAcc = ($n_test > 0) ? ($correctWithoutFilter / $n_test) * 100.0 : 0.0;
$filteredAcc = ($tradesTaken > 0) ? ($tradesCorrect / $tradesTaken) * 100.0 : 0.0;
$selectivity = ($n_test > 0) ? ($tradesTaken / $n_test) * 100.0 : 0.0;

echo "\n";
echo str_repeat('─', 72) . "\n";
echo "  META-LABELING PERFORMANCE REPORT\n";
echo str_repeat('─', 72) . "\n";
printf("  Total Test Days                        : %d\n",      $n_test);
printf("  Primary (unfiltered) Directional Acc.  : %.2f%%\n",  $baselineAcc);
printf("  Confidence Threshold                   : %.0f%%\n",  $META_CONFIDENCE_THRESHOLD * 100);
printf("  Trades Taken (passed Meta-Filter)      : %d / %d  (%.1f%% selectivity)\n", $tradesTaken, $n_test, $selectivity);
printf("  Directional Accuracy (filtered trades) : %.2f%%\n",  $filteredAcc);
echo str_repeat('─', 72) . "\n";

$lift = $filteredAcc - $baselineAcc;

echo "\n  INTERPRETATION:\n";
if ($tradesTaken === 0) {
    echo "  [CAUTION] Meta-Filter rejected ALL trades at the 60% threshold.\n";
} elseif ($filteredAcc > 60.0 && $lift > 3.0) {
    printf("  [ELITE]  Filtered accuracy %.2f%% with +%.2f%% lift over baseline.\n", $filteredAcc, $lift);
} elseif ($filteredAcc > 55.0) {
    printf("  [STRONG] Filtered accuracy %.2f%% — a solid quantitative edge.\n", $filteredAcc);
} elseif ($filteredAcc > 52.0) {
    printf("  [MODEST] Filtered accuracy %.2f%% — marginal edge above coin-flip.\n", $filteredAcc);
} else {
    printf("  [BELOW BASELINE] Filtered accuracy %.2f%%.\n", $filteredAcc);
}
echo str_repeat('─', 72) . "\n";