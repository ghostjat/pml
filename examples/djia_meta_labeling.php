<?php

declare(strict_types=1);

/**
 * examples/djia_meta_labeling.php
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *  TIER-1 HEDGE FUND ARCHITECTURE: META-LABELING
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  Meta-Labeling is a technique pioneered by Marcos Lopez de Prado
 *  (Advances in Financial Machine Learning, 2018). The core insight is:
 *
 *    "Technical indicators tell us WHAT direction to trade.
 *     Meta-Labeling tells us WHETHER to actually place the bet."
 *
 *  Two-Stage Architecture:
 *
 *    PRIMARY MODEL  (The Strategist)
 *    ────────────────────────────────
 *    Trained on 70% of data.
 *    A regression model that predicts the continuous 5-day forward return.
 *    Its sign (+ / –) gives the trade direction (Long / Short).
 *    Pipeline: StandardScaler → XGBRegressor
 *
 *    META-MODEL  (The Risk Manager)
 *    ────────────────────────────────
 *    Trained on the next 15% of data (the "meta-train" window).
 *    Receives a AUGMENTED feature set: original indicators + the Primary
 *    Model's predicted return as an extra column.
 *    Its target is a binary META-LABEL:
 *      1 = "Primary Model was correct (signs matched)"
 *      0 = "Primary Model was wrong"
 *    Pipeline: SimpleImputer(median) → StandardScaler → XGBClassifier
 *
 *    INFERENCE  (The Execution Layer)
 *    ─────────────────────────────────
 *    Final 15% of data. Both models run in sequence:
 *      1. Primary predicts direction.
 *      2. Meta-Model predicts P(primary is correct) via predict_proba().
 *      3. ONLY execute the trade if P(correct) > 0.75 (75% confidence gate).
 *
 *  Why does this work? By training a second model to identify WHEN the first
 *  model is reliable, we filter out the coin-flip trades and only take
 *  positions when the ensemble has a genuine statistical edge.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../vendor/autoload.php';

use Pml\Classic\Exploration\DataProfiler;
use Pml\Classic\Preprocess\StandardScaler;
use Pml\Classic\Impute\SimpleImputer;
use Pml\Classic\Ensemble\XGBRegressor;
use Pml\Classic\Ensemble\XGBClassifier;
use Pml\Classic\Pipeline\Pipeline;
use Pml\Tensor;

// ── Utility ──────────────────────────────────────────────────────────────────

function banner(string $title, string $phase): void
{
    $line = str_repeat('═', 72);
    echo "\n{$line}\n  {$phase}: {$title}\n{$line}\n";
}

/**
 * Load a TSV (tab-separated) file and return the same Bunch-style array that
 * DataLoader::load_csv() produces.  DataLoader defaults to comma delimiter;
 * the stock CSVs (tcs.csv, nifty50.csv) use tabs, so they require this loader.
 *
 * @return array{data: list<list<string>>, feature_names: list<string>}
 */
function loadTsv(string $path): array
{
    $fh = fopen($path, 'r');
    if ($fh === false) {
        throw new \RuntimeException("Cannot open: {$path}");
    }

    $headers = [];
    $rows    = [];

    $first = true;
    while (($row = fgetcsv($fh, 0, "\t")) !== false) {
        if ($row === [null]) {
            continue;   // blank line
        }
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

// Quick EDA — mirror pandas df.info() / df.describe()
$featureNames = $dataset['feature_names'] ?? ['date', 'open', 'high', 'low', 'close', 'volume'];
DataProfiler::info($raw_data, $featureNames);

$descStats = DataProfiler::describe($raw_data, $featureNames);
DataProfiler::print_describe($descStats);

// ── Load Market Index & Build O(1) Date → Close Hash Map ─────────────────────
//
//  WHY A HASH MAP?
//  ───────────────
//  A naive approach would search the index CSV row-by-row for each stock date,
//  producing O(n × m) complexity — unacceptable for datasets with thousands of rows.
//  By pre-indexing the market data into an associative array keyed on the date
//  string, every lookup in the feature loop costs O(1) (PHP hash table).
//
//  WHY TRUE ALPHA INSTEAD OF A HARDCODED DRIFT?
//  ──────────────────────────────────────────────
//  A static 0.25% 5-day baseline is meaningless during:
//    • Market crashes  (index −10%): every stock looks like an "outperformer"
//    • Bull runs       (index +5%):  every stock looks like an "underperformer"
//  Both scenarios inject systematic bias that the Meta-Model will learn as
//  a spurious pattern, degrading live performance.  True Alpha = stock return
//  minus the contemporaneous index return, stripped of broad-market noise.

$marketCsvPath = __DIR__ . '/../datasets/stocks/nifty50.csv';

/** @var array<string, float> $market_closes  date-string → closing price */
$market_closes = [];

if (file_exists($marketCsvPath)) {
    $marketDataset = loadTsv($marketCsvPath);

    foreach ($marketDataset['data'] as $row) {
        // Column layout (nifty50.csv): [0]=Date  [1]=Open  [2]=High  [3]=Low  [4]=Close
        $dateKey  = trim((string) $row[0]);   // e.g. "28/03/2006"
        $closeVal = (float) $row[4];
        if ($dateKey !== '' && $closeVal > 0.0) {
            $market_closes[$dateKey] = $closeVal;
        }
    }

    printf("  Market index : %s  (%d trading days indexed)\n", $marketCsvPath, count($market_closes));
} else {
    echo "  [WARNING] nifty50.csv not found — market return defaults to 0.0 for all rows.\n";
}

// ── Feature Engineering ───────────────────────────────────────────────────────
//
//  9-feature institutional-grade arsenal:
//
//  F1  RSI Proxy (14-day)          — momentum overbought/oversold
//  F2  5-Day Historical Volatility — price compression / expansion
//  F3  True Cross-Asset Alpha      — company alpha vs broad market
//  F4  RVOL (Relative Volume 20d)  — institutional accumulation signal
//  F5  OBV Proxy (10-day)          — directional volume pressure
//  F6  SMA-50 Distance             — medium-term trend position
//  F7  SMA-200 Distance            — macro-trend position (bull/bear regime)
//  F8  Bollinger Band Width (20d)  — volatility compression; precedes breakouts
//
//  TARGET  10-Day Forward Return
//      (close[i+10] − close[i]) / close[i]
//      Wider horizon reduces noise vs the 5-day target.
//
//  Loop constraints:
//    Start: i = 200  → SMA-200 requires 200 prior closes (binding constraint)
//    End:   i < n − 10  → need 10 future closes for the forward-return target

$engineered_X = [];
$target_y     = [];

$n_raw = count($raw_data);

echo "\n  Engineering 8-feature quant arsenal (RSI | Vol5d | Alpha | RVOL | OBV | SMA50 | SMA200 | BBW)...\n";

for ($i = 200; $i < $n_raw - 10; $i++) {

    $close   = (float) $raw_data[$i][4];
    $vol     = (float) $raw_data[$i][5];
    $volPrev = (float) $raw_data[$i - 1][5];

    // Skip degenerate rows early — dirty data guard
    if ($close <= 0.0 || $volPrev <= 0.0) {
        continue;
    }

    // ── F1: 14-Day RSI Proxy ──────────────────────────────────────────────
    //
    // RS = avg_up_move / avg_down_move over 14 daily price changes.
    // RSI = 100 − 100/(1+RS).  If no down moves at all → RSI = 100.
    $upSum   = 0.0;
    $downSum = 0.0;
    $upCnt   = 0;
    $downCnt = 0;

    for ($k = 0; $k < 14; $k++) {
        $c1 = (float) $raw_data[$i - $k][4];
        $c0 = (float) $raw_data[$i - $k - 1][4];
        if ($c0 <= 0.0) {
            continue;
        }
        $delta = $c1 - $c0;
        if ($delta > 0.0) {
            $upSum += $delta;
            $upCnt++;
        } elseif ($delta < 0.0) {
            $downSum += abs($delta);
            $downCnt++;
        }
    }

    $avgUp   = ($upCnt   > 0) ? $upSum   / $upCnt   : 0.0;
    $avgDown = ($downCnt > 0) ? $downSum / $downCnt : 0.0;

    $rsiProxy = ($avgDown < 1e-12)
        ? 100.0
        : 100.0 - 100.0 / (1.0 + $avgUp / $avgDown);

    // ── F2: 5-Day Historical Volatility ───────────────────────────────────
    //
    // Population std dev of the last 5 daily returns.
    // Contracting volatility often precedes directional breakouts.
    $returns5      = [];
    $rMean5        = 0.0;
    $volDegenerate = false;

    for ($k = 0; $k < 5; $k++) {
        $c1 = (float) $raw_data[$i - $k][4];
        $c0 = (float) $raw_data[$i - $k - 1][4];
        if ($c0 <= 0.0) {
            $volDegenerate = true;
            break;
        }
        $r          = ($c1 - $c0) / $c0;
        $returns5[] = $r;
        $rMean5    += $r;
    }

    if ($volDegenerate || count($returns5) < 5) {
        $vol5d = NAN;
    } else {
        $rMean5  /= 5.0;
        $variance = 0.0;
        foreach ($returns5 as $r) {
            $variance += ($r - $rMean5) ** 2;
        }
        $vol5d = sqrt($variance / 5.0);
    }

    // ── F3: True Cross-Asset Alpha (5-Day) ────────────────────────────────
    //
    // Alpha = stock_5d_return − market_5d_return.
    // O(1) date-keyed lookup — holiday gaps default market return to 0.0.
    $dateToday    = trim((string) $raw_data[$i][0]);
    $datePastFive = trim((string) $raw_data[$i - 5][0]);
    $closeM5      = (float) $raw_data[$i - 5][4];

    if ($closeM5 <= 0.0) {
        $relStrength = NAN;
    } else {
        $stock5d  = ($close - $closeM5) / $closeM5;
        $market5d = (isset($market_closes[$dateToday], $market_closes[$datePastFive])
            && $market_closes[$datePastFive] > 0.0)
            ? ($market_closes[$dateToday] - $market_closes[$datePastFive]) / $market_closes[$datePastFive]
            : 0.0;
        $relStrength = $stock5d - $market5d;
    }

    // ── F4: Relative Volume — RVOL (20-Day) ───────────────────────────────
    //
    // RVOL = Today_Volume / Avg_Volume_20d.
    // RVOL > 2.0 = strong institutional participation (accumulation or distribution).
    // RVOL < 0.5 = thin, low-conviction tape.
    // We guard against a zero average (pathological data) with a 1e-9 floor.
    $volSum20 = 0.0;
    for ($k = 0; $k < 20; $k++) {
        $volSum20 += (float) $raw_data[$i - $k][5];
    }
    $avgVol20 = $volSum20 / 20.0;
    $rvol     = ($avgVol20 > 1e-9) ? $vol / $avgVol20 : 1.0;

    // ── F5: On-Balance Volume Proxy (10-Day) ──────────────────────────────
    //
    // Over the last 10 days:
    //   If close[d] > close[d-1]  → add volume[d]   (buying pressure)
    //   If close[d] < close[d-1]  → subtract it      (selling pressure)
    //   If equal                  → neutral (0 contribution)
    //
    // Normalise by total 10-day volume → range (-1, +1).
    // A strongly positive OBV proxy means buyers are in control on up-days;
    // negative means sellers dominate.  Guard against zero total volume.
    $obvSum   = 0.0;
    $volSum10 = 0.0;

    for ($k = 0; $k < 10; $k++) {
        $cd  = (float) $raw_data[$i - $k][4];
        $cp  = (float) $raw_data[$i - $k - 1][4];
        $vd  = (float) $raw_data[$i - $k][5];
        $volSum10 += $vd;
        if ($cd > $cp) {
            $obvSum += $vd;
        } elseif ($cd < $cp) {
            $obvSum -= $vd;
        }
    }
    $obvProxy = ($volSum10 > 1e-9) ? $obvSum / $volSum10 : 0.0;

    // ── F6: SMA-50 Distance ───────────────────────────────────────────────
    //
    // Today_Close / 50_Day_SMA.
    // > 1.0 → price above mid-term average (bullish momentum regime).
    // < 1.0 → price below mid-term average (bearish / mean-reversion setup).
    $sma50Sum = 0.0;
    for ($k = 0; $k < 50; $k++) {
        $sma50Sum += (float) $raw_data[$i - $k][4];
    }
    $sma50    = $sma50Sum / 50.0;
    $sma50Dist = ($sma50 > 1e-9) ? $close / $sma50 : NAN;

    // ── F7: SMA-200 Distance (Macro Trend) ───────────────────────────────
    //
    // Today_Close / 200_Day_SMA.
    // The 200-day SMA is the canonical bull/bear regime divider used by every
    // institutional desk.  Distance from it encodes whether we are in a
    // structural uptrend (> 1) or a bear market (< 1).
    $sma200Sum = 0.0;
    for ($k = 0; $k < 200; $k++) {
        $sma200Sum += (float) $raw_data[$i - $k][4];
    }
    $sma200     = $sma200Sum / 200.0;
    $sma200Dist = ($sma200 > 1e-9) ? $close / $sma200 : NAN;

    // ── F8: Bollinger Band Width (20-Day) ─────────────────────────────────
    //
    // BBW = (Highest_High_20d − Lowest_Low_20d) / SMA_20_Close
    //
    // Uses the price channel (high/low extremes) rather than std dev of closes,
    // which is more sensitive to intraday volatility and gap events.
    // BBW shrinking → price compression → breakout imminent.
    // BBW expanding → trend extension or panic selling.
    $sma20Sum  = 0.0;
    $high20Max = PHP_FLOAT_MIN;
    $low20Min  = PHP_FLOAT_MAX;

    for ($k = 0; $k < 20; $k++) {
        $h = (float) $raw_data[$i - $k][2];
        $l = (float) $raw_data[$i - $k][3];
        $c20 = (float) $raw_data[$i - $k][4];
        $sma20Sum += $c20;
        if ($h > $high20Max) {
            $high20Max = $h;
        }
        if ($l < $low20Min) {
            $low20Min = $l;
        }
    }
    $sma20 = $sma20Sum / 20.0;
    $bbw   = ($sma20 > 1e-9) ? ($high20Max - $low20Min) / $sma20 : NAN;

    // ── TARGET: 10-Day Forward Return ─────────────────────────────────────
    //
    // Wider 10-day horizon reduces single-day noise vs the 5-day target,
    // giving the model a cleaner signal to learn from.
    $closeFuture   = (float) $raw_data[$i + 10][4];
    $forwardReturn = ($close > 1e-9) ? ($closeFuture - $close) / $close : NAN;

    // Skip rows where critical features are fully undefined
    if (is_nan($forwardReturn)) {
        continue;
    }

    $engineered_X[] = [
        $rsiProxy,    // F1
        $vol5d,       // F2
        $relStrength, // F3
        $rvol,        // F4
        $obvProxy,    // F5
        $sma50Dist,   // F6
        $sma200Dist,  // F7
        $bbw,         // F8
    ];
    $target_y[] = $forwardReturn;
}

$num_samples  = count($engineered_X);
$num_features = count($engineered_X[0]);

printf(
    "  Engineered matrix : [%d samples × %d features]\n  Features          : RSI_14 | Vol_5d | Alpha | RVOL | OBV | SMA50D | SMA200D | BBW\n  Target range      : [%.4f, %.4f]  (10-day forward return)\n",
    $num_samples,
    $num_features,
    min($target_y),
    max($target_y)
);


// ─────────────────────────────────────────────────────────────────────────────
//  PHASE 2 — Sequential Tri-Split & Primary Model Training
// ─────────────────────────────────────────────────────────────────────────────

banner('Sequential Tri-Split & Primary Model (Direction)', 'Phase 2');

// ── Tri-Split: 70% / 15% / 15% ───────────────────────────────────────────────
//
//  NEVER shuffle time-series data — future cannot precede past.
//  We use three non-overlapping sequential windows:
//
//   [0 … n_primary)       → Primary model training set
//   [n_primary … n_meta)  → Meta-model training set (out-of-sample for Primary)
//   [n_meta … n)          → Final blind test set
//
//  The Primary model NEVER sees the meta-train or test windows during fit().
//  The Meta-model is trained on predictions the Primary makes on UNSEEN data,
//  which ensures the meta-labels are drawn from the true out-of-sample
//  error distribution — not the training distribution.

$n_primary = (int) ($num_samples * 0.70);
$n_meta    = (int) ($num_samples * 0.85);   // primary + meta = 85%
$n_test    = $num_samples - $n_meta;

printf(
    "  Total  : %d  |  Primary train : %d  |  Meta train : %d  |  Test : %d\n",
    $num_samples,
    $n_primary,
    $n_meta - $n_primary,
    $n_test
);

// Split using array_slice() BEFORE building Tensors — avoids allocating a
// monolithic tensor and then re-slicing (wastes memory on large datasets)
$X_primary_arr = array_slice($engineered_X, 0, $n_primary);
$y_primary_arr = array_slice($target_y,     0, $n_primary);

$X_meta_arr    = array_slice($engineered_X, $n_primary, $n_meta - $n_primary);
$y_meta_arr    = array_slice($target_y,     $n_primary, $n_meta - $n_primary);

$X_test_arr    = array_slice($engineered_X, $n_meta);
$y_test_arr    = array_slice($target_y,     $n_meta);

// ── Pack Primary split into Tensors ──────────────────────────────────────────
$X_train_primary = Tensor::fromArray($X_primary_arr, [$n_primary, $num_features]);
$y_train_primary = Tensor::fromArray($y_primary_arr, [$n_primary]);

// ── Primary Pipeline: StandardScaler → XGBRegressor ──────────────────────────
//
//  Regularization rationale (overfitting prevention):
//    n_estimators = 100  → fewer trees; each tree covers broader patterns
//    max_depth    = 2    → stumps — can only learn one interaction per tree,
//                          forcing the ensemble to model broad regimes rather
//                          than memorising market micro-noise
//    learning_rate = 0.05 → moderate shrinkage; fast enough to converge in 100 rounds
//    subsample    = 0.5  → heavy row bagging: each tree sees only half the data,
//                          creating maximum diversity and variance reduction
$primaryPipeline = new Pipeline([
    ['scaler', new StandardScaler()],
    ['xgb',   new XGBRegressor(
        n_estimators:  100,
        max_depth:     2,
        learning_rate: 0.05,
        subsample:     0.5,
        random_state:  42,
    )],
]);

echo "  Fitting Primary Pipeline (StandardScaler → XGBRegressor)...\n";
$primaryPipeline->fit($X_train_primary, $y_train_primary);
echo "  Primary model fitted on {$n_primary} training days.\n";


// ─────────────────────────────────────────────────────────────────────────────
//  PHASE 3 — Meta-Label Construction & Meta-Model Training
// ─────────────────────────────────────────────────────────────────────────────

banner('Meta-Label Construction & Meta-Model Training', 'Phase 3');

// ── Step 3A: Generate Primary Predictions on the Meta-Train window ────────────
//
//  CRITICAL: We predict on the meta-train data which the Primary model has
//  NEVER seen.  This gives us a realistic sample of the Primary model's
//  out-of-sample accuracy, from which the Meta-Model learns to distinguish
//  confident correct calls from unreliable noisy ones.

$n_meta_size  = $n_meta - $n_primary;
$X_meta_tensor = Tensor::fromArray($X_meta_arr, [$n_meta_size, $num_features]);
$y_meta_tensor = Tensor::fromArray($y_meta_arr, [$n_meta_size]);

echo "  Generating Primary Model predictions on {$n_meta_size} meta-train samples...\n";
$meta_primary_preds = $primaryPipeline->predict($X_meta_tensor);

// ── Step 3B: Build the Binary Meta-Labels ─────────────────────────────────────
//
//  For each sample i in the meta-train window:
//    1  → Primary Model was CORRECT: predicted return sign matches actual sign
//    0  → Primary Model was WRONG:   signs conflict
//
//  This binary signal teaches the Meta-Model to recognise the *features*
//  and *market conditions* where the Primary Model can be trusted.

$meta_labels   = [];   // float: 0.0 or 1.0
$metaCorrect   = 0;

for ($i = 0; $i < $n_meta_size; $i++) {
    $actual    = (float) $y_meta_tensor->buffer[$i];
    $predicted = (float) $meta_primary_preds->buffer[$i];

    // Signs match → Primary was directionally correct → Meta-Label = 1
    $correct = ($actual > 0.0 && $predicted > 0.0) || ($actual < 0.0 && $predicted < 0.0);
    $label   = $correct ? 1.0 : 0.0;
    $meta_labels[] = $label;
    if ($correct) {
        $metaCorrect++;
    }
}

$primaryAccOnMeta = ($metaCorrect / $n_meta_size) * 100.0;
printf(
    "  Primary directional accuracy on meta-train : %.2f%%\n",
    $primaryAccOnMeta
);
printf(
    "  Meta-label distribution                    : %d correct (1) / %d incorrect (0)\n",
    $metaCorrect,
    $n_meta_size - $metaCorrect
);

// ── Step 3C: Augment Meta Feature Matrix ──────────────────────────────────────
//
//  The Primary Model's predicted return is appended as an EXTRA FEATURE to
//  the meta-train matrix.  This allows the Meta-Model to learn:
//
//    "When RSI=72 AND vol is low AND the Primary Model predicts +2.5% return,
//     what is the probability that the Primary Model is actually correct?"
//
//  This is the key architectural insight: the predicted return IS a feature,
//  not just a threshold.  It encodes the Primary Model's confidence magnitude.

$X_meta_augmented = [];
for ($i = 0; $i < $n_meta_size; $i++) {
    $row   = $X_meta_arr[$i];
    $row[] = (float) $meta_primary_preds->buffer[$i];   // append primary prediction
    $X_meta_augmented[] = $row;
}

$n_meta_features = $num_features + 1;   // original features + primary prediction

$X_meta_aug_tensor = Tensor::fromArray($X_meta_augmented, [$n_meta_size, $n_meta_features]);
$y_meta_labels     = Tensor::fromArray($meta_labels,      [$n_meta_size]);

// ── Step 3D: Fit the Meta Pipeline ────────────────────────────────────────────
//
//  SimpleImputer(median) → handles any NaN features from degenerate market days
//  StandardScaler       → normalises the augmented feature space
//  XGBClassifier        → binary:logistic outputs P(class=1) = P(primary correct)
//
//  XGBClassifier automatically selects 'binary:logistic' when it detects 2 classes.
//  predict_proba() returns a Tensor[n_samples] with P(correct | features).

// ── Meta-Pipeline: SimpleImputer → StandardScaler → XGBClassifier ────────────
//
//  Regularization rationale (small meta-train set; high overfitting risk):
//    n_estimators = 50   → tiny ensemble; meta-labels have very weak signal,
//                          more trees just memorise the noise
//    max_depth    = 2    → stumps only — binary label with 9 features needs
//                          almost no depth to find the key split
//    learning_rate = 0.02 → very slow shrinkage; prevents the model from
//                           chasing the last few meta-train examples
//    subsample    = 0.5  → same heavy bagging as the Primary Model
$metaPipeline = new Pipeline([
    ['imputer', new SimpleImputer(strategy: 'median')],
    ['scaler',  new StandardScaler()],
    ['xgb',     new XGBClassifier(
        n_estimators:  50,
        max_depth:     2,
        learning_rate: 0.02,
        subsample:     0.5,
        random_state:  99,
    )],
]);

echo "\n  Fitting Meta Pipeline (SimpleImputer → StandardScaler → XGBClassifier)...\n";
$metaPipeline->fit($X_meta_aug_tensor, $y_meta_labels);
echo "  Meta-Model fitted on {$n_meta_size} augmented samples.\n";


// ─────────────────────────────────────────────────────────────────────────────
//  PHASE 4 — Production Inference & Confidence-Gated Execution
// ─────────────────────────────────────────────────────────────────────────────

banner('Production Inference & Confidence-Gated Execution', 'Phase 4');

// ── Step 4A: Primary Model predicts on unseen test data ───────────────────────

$X_test_tensor = Tensor::fromArray($X_test_arr, [$n_test, $num_features]);
$y_test_tensor = Tensor::fromArray($y_test_arr, [$n_test]);

echo "  Step A: Primary Model predicting on {$n_test} test samples...\n";
$test_primary_preds = $primaryPipeline->predict($X_test_tensor);

// ── Step 4B: Augment test features & Meta-Model computes confidence ────────────
//
//  The test set receives the SAME augmentation as meta-train:
//  append the Primary Model's predicted return as the last column.
//  Then the Meta-Model produces P(primary correct) for each test day.

$X_test_augmented = [];
for ($i = 0; $i < $n_test; $i++) {
    $row   = $X_test_arr[$i];
    $row[] = (float) $test_primary_preds->buffer[$i];
    $X_test_augmented[] = $row;
}

$X_test_aug_tensor = Tensor::fromArray($X_test_augmented, [$n_test, $n_meta_features]);

echo "  Step B: Meta-Model computing confidence probabilities...\n";

// predict_proba() → binary:logistic → Tensor[n_test] with P(correct) per sample
$meta_probas = $metaPipeline->predict_proba($X_test_aug_tensor);

// ── Step 4C: Apply the 75% Confidence Gate ────────────────────────────────────
//
//  ONLY execute a trade when the Meta-Model's confidence exceeds 0.75.
//  This filters out the "coin-flip" trades (P ≈ 0.5) where the Primary
//  Model has no genuine edge — the trades most likely to produce losses.
//
//  The threshold of 0.75 is not arbitrary: at P > 0.75, the Meta-Model
//  has learned a signal 3× stronger than random.  In a Sharpe-ratio
//  framework, reducing trade frequency while improving hit-rate is the
//  correct tradeoff.

// Regularized models produce less extreme probabilities (compressed toward 0.5),
// so a 0.75 gate would reject nearly all trades.  0.60 captures the regime where
// the Meta-Model has a genuine 3:2 odds edge while still allowing enough trades
// to measure the directional accuracy meaningfully.
$META_CONFIDENCE_THRESHOLD = 0.60;

$totalDays       = $n_test;
$tradesTaken     = 0;
$tradesCorrect   = 0;

// Track filtered-out stats for comparison
$totalWithoutFilter = 0;
$correctWithoutFilter = 0;

for ($i = 0; $i < $n_test; $i++) {
    $actual    = (float) $y_test_tensor->buffer[$i];
    $predicted = (float) $test_primary_preds->buffer[$i];
    $confidence = (float) $meta_probas->buffer[$i];

    // Unfiltered baseline
    $dirMatch = ($actual > 0.0 && $predicted > 0.0) || ($actual < 0.0 && $predicted < 0.0);
    if ($actual != 0.0 && $predicted != 0.0) {
        $totalWithoutFilter++;
        if ($dirMatch) {
            $correctWithoutFilter++;
        }
    }

    // Confidence-gated trades
    if ($confidence > $META_CONFIDENCE_THRESHOLD) {
        $tradesTaken++;
        if ($dirMatch) {
            $tradesCorrect++;
        }
    }
}

// ── Terminal Report ───────────────────────────────────────────────────────────

$baselineAcc = ($totalWithoutFilter > 0)
    ? ($correctWithoutFilter / $totalWithoutFilter) * 100.0
    : 0.0;

$filteredAcc = ($tradesTaken > 0)
    ? ($tradesCorrect / $tradesTaken) * 100.0
    : 0.0;

$selectivity = ($totalDays > 0) ? ($tradesTaken / $totalDays) * 100.0 : 0.0;

echo "\n";
echo str_repeat('─', 72) . "\n";
echo "  META-LABELING PERFORMANCE REPORT\n";
echo str_repeat('─', 72) . "\n";
printf("  Total Test Days                        : %d\n",      $totalDays);
printf("  Primary (unfiltered) Directional Acc.  : %.2f%%\n",  $baselineAcc);
printf("  Confidence Threshold                   : %.0f%%\n",   $META_CONFIDENCE_THRESHOLD * 100);
printf("  Trades Taken (passed Meta-Filter)      : %d / %d  (%.1f%% selectivity)\n",
    $tradesTaken, $totalDays, $selectivity);
printf("  Directional Accuracy (filtered trades) : %.2f%%\n",  $filteredAcc);
echo str_repeat('─', 72) . "\n";

// ── Interpretation ────────────────────────────────────────────────────────────

$lift = $filteredAcc - $baselineAcc;

echo "\n  INTERPRETATION:\n";

if ($tradesTaken === 0) {
    echo "  [CAUTION] Meta-Filter rejected ALL trades at the 75% threshold.\n";
    echo "  The Meta-Model found no high-confidence setups.  Consider lowering\n";
    echo "  the threshold or re-examining the feature engineering pipeline.\n";
} elseif ($filteredAcc > 60.0 && $lift > 3.0) {
    printf("  [ELITE]  Filtered accuracy %.2f%% with +%.2f%% lift over baseline.\n", $filteredAcc, $lift);
    echo "  Meta-Labeling has significantly improved the Primary Model's edge.\n";
    echo "  This trade-selectivity profile is consistent with Tier-1 execution.\n";
} elseif ($filteredAcc > 55.0) {
    printf("  [STRONG] Filtered accuracy %.2f%% — a solid quantitative edge.\n", $filteredAcc);
    echo "  In high-frequency finance, >55% directional accuracy is profitable.\n";
    echo "  Meta-filter is adding value — consider tuning the confidence threshold.\n";
} elseif ($filteredAcc > 52.0) {
    printf("  [MODEST] Filtered accuracy %.2f%% — marginal edge above coin-flip.\n", $filteredAcc);
    echo "  Add macro features (VIX proxy, sector rotation) or widen the lookback.\n";
} else {
    printf("  [BELOW BASELINE] Filtered accuracy %.2f%%.\n", $filteredAcc);
    echo "  Meta-Model is not adding value.  Check: class imbalance in meta-labels,\n";
    echo "  data leakage, or insufficient meta-train set size.\n";
}

echo str_repeat('─', 72) . "\n";
