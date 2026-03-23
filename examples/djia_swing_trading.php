<?php

declare(strict_types=1);

/**
 * ════════════════════════════════════════════════════════════════════════════
 *  examples/djia_swing_trading.php — Weekly Swing Trading Strategy
 * ════════════════════════════════════════════════════════════════════════════
 *
 * WHY WE PREDICT WEEKLY SWINGS, NOT DAILY NOISE
 * ───────────────────────────────────────────────
 * Daily returns are dominated by random micro-events: order-flow imbalances,
 * news sentiment spikes, and intraday volatility that no model can reliably
 * capture.  The signal-to-noise ratio on a 1-day horizon is near zero.
 *
 * The 5-session (one trading week) horizon smooths that noise and exposes
 * structural, momentum-driven patterns that repeat across market cycles:
 *
 *   RSI momentum     — is the asset overbought or oversold?
 *   Stochastic 15,3,3 — where is price relative to its recent range?
 *   Turtle 55-day high — has price broken out of a major structural level?
 *   Weekly volatility  — are we in a wide-range or coiling week?
 *   MACD proxy         — is short-term momentum above or below medium-term?
 *
 * All five features are STATIONARY — they hover around a constant mean
 * regardless of whether GOOGL trades at $200 (2006) or $1,000 (2017).
 * This allows XGBoost to learn from patterns in 2008 and apply them to 2016.
 *
 * CONFIDENCE THRESHOLDING (THE QUANT EDGE)
 * ─────────────────────────────────────────
 * We do not trade every signal.  A predicted return near 0% means the model
 * is uncertain — acting on that signal loses money to spread and commission.
 * We only trade when |predicted_return| ≥ 1.5%: a move large enough to
 * overcome friction and represent a genuine high-conviction call.
 *
 * GOLD STANDARD — 55%+ Win Rate on Filtered, High-Conviction Swing Trades
 * ─────────────────────────────────────────────────────────────────────────
 * A 55%+ directional win rate on confidence-filtered swing trades using
 * macro technical features (Stochastic, Turtle breakout, MACD) is the gold
 * standard for systematic quantitative funds.  Most institutional trend-
 * following strategies target 55–65% directional accuracy after filtering.
 * Below 50% the strategy is loss-making; 50–55% barely covers transaction
 * costs; above 55% represents an exploitable statistical edge.
 *
 * USAGE
 * ──────
 *   php examples/djia_swing_trading.php
 *   php examples/djia_swing_trading.php path/to/stock.csv
 *
 * DATASET
 * ────────
 *   datasets/stocks/GOOGL.csv  (Kaggle DJIA 2006–2018)
 *   Columns: Date, Open, High, Low, Close, Volume, Name
 */

// ── Autoloader ────────────────────────────────────────────────────────────────

foreach ([__DIR__ . '/../vendor/autoload.php', __DIR__ . '/vendor/autoload.php'] as $al) {
    if (file_exists($al)) { require_once $al; break; }
}

use Pml\Tensor;
use Pml\Classic\Datasets\DataLoader;
use Pml\Classic\Exploration\DataProfiler;
use Pml\Classic\Impute\SimpleImputer;
use Pml\Classic\Preprocess\StandardScaler;
use Pml\Classic\Ensemble\XGBRegressor;
use Pml\Classic\Pipeline\Pipeline;
use Pml\Classic\Metrics\Metrics;

// ── Helper ────────────────────────────────────────────────────────────────────

function banner(string $title, string $phase): void
{
    $line = str_repeat('═', 70);
    echo "\n{$line}\n  {$phase}: {$title}\n{$line}\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 1: Ingestion & Exploratory Data Analysis
//
//  We profile the raw CSV BEFORE feature engineering so any structural issues
//  (missing values, wrong dtypes, unexpected ranges) are visible to the user
//  before we commit to the pipeline.  This mirrors the pandas workflow:
//    df.info()      → column types and null counts
//    df.describe()  → distributional statistics for numeric columns
// ─────────────────────────────────────────────────────────────────────────────

banner('Ingestion & Exploratory Data Analysis', 'Phase 1');

$csvPath = $argv[1] ?? __DIR__ . '/../datasets/stocks/APPL.csv';

if (!file_exists($csvPath)) {
    fwrite(STDERR,
        "ERROR: Dataset not found at '{$csvPath}'.\n"
        . "Pass the path as an argument:\n"
        . "  php examples/djia_swing_trading.php /path/to/GOOGL.csv\n"
    );
    exit(1);
}

// DataLoader::load_csv() streams via fgetcsv(), casting numeric cells to float.
// With header: true the first row becomes $feature_names; data rows are
// returned as integer-indexed arrays: [0]=Date [1]=Open [2]=High [3]=Low
//                                      [4]=Close [5]=Volume [6]=Name
$bunch        = DataLoader::load_csv($csvPath, header: true);
$raw_data     = $bunch['data'];
$feature_names = $bunch['feature_names'];
$n_total      = count($raw_data);

printf("  Loaded %d trading sessions from %s\n", $n_total, basename($csvPath));

// ── Column overview (mirrors df.info()) ──────────────────────────────────────
echo "\n── Column Overview (info) ────────────────────────────────────────────\n";
DataProfiler::info($raw_data, $feature_names);

// ── Distributional statistics (mirrors df.describe()) ────────────────────────
echo "\n── Numeric Summary (describe) ────────────────────────────────────────\n";
$stats = DataProfiler::describe($raw_data, $feature_names);
DataProfiler::print_describe($stats, precision: 2);

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 2: Advanced Feature Engineering & The 5-Day Horizon
//
//  Loop bounds:
//    start = 55  — Turtle 55-day high is the longest-lookback feature
//                  (needs high[i−54..i]).  All other features fit within this:
//                    Stochastic 15,3,3 → oldest needed: i−18  (at i=55 → 37)
//                    MACD 26-day SMA   → i−25              (at i=55 → 30)
//                    RSI 14-day proxy  → i−14              (at i=55 → 41)
//                    Weekly vol        → i−4               (at i=55 → 51)
//    end   = n−5 (exclusive) — forward return needs close[i+5]
//
//  NaN STRATEGY
//  Degenerate conditions (zero price range in Stochastic, zero SMA in MACD)
//  are encoded as NAN.  The SimpleImputer in Phase 3 fills these with the
//  per-column median computed on the training set — exactly the sklearn
//  SimpleImputer(strategy='median') behaviour.
// ─────────────────────────────────────────────────────────────────────────────

banner('Advanced Feature Engineering (5 Macro Features)', 'Phase 2');

$engineered_X = [];   // 2D: one row per sample, 5 features per row
$target_y     = [];   // 1D: 5-day forward return per sample

for ($i = 55; $i < $n_total - 5; $i++) {

    // ── Core price extractions ─────────────────────────────────────────────

    $today_close  = (float) $raw_data[$i][4];
    $close_5d_ago = (float) $raw_data[$i - 5][4];
    $close_in_5   = (float) $raw_data[$i + 5][4];

    // Skip rows with corrupt zero-price data (extremely rare in DJIA datasets)
    if ($today_close <= 0.0 || $close_5d_ago <= 0.0) {
        continue;
    }

    // ── TARGET: 5-Day Forward Return ──────────────────────────────────────
    //
    // "If I buy at today's close, what % return do I earn by closing the
    //  position exactly 5 sessions later?"
    //
    // Stationary: hovers around 0% with consistent variance whether GOOGL
    // trades at $200 or $1,000 — safe to model across the full 12-year window.
    $target_return = ($close_in_5 - $today_close) / $today_close;

    // ── FEATURE 1: 14-Day RSI Proxy ───────────────────────────────────────
    //
    // The Relative Strength Index (RSI) signals whether recent buying pressure
    // (up-days) outweighs selling pressure (down-days) over the last fortnight.
    //
    //   ratio > 1 → more up-days → potentially overbought (short setup)
    //   ratio = 1 → balanced market
    //   ratio < 1 → more down-days → potentially oversold (long setup)
    //
    // Using max($down_days, 1) prevents zero-division when all 14 sessions
    // move in the same direction, while preserving the extreme signal.
    $up_days   = 0;
    $down_days = 0;
    for ($j = 1; $j <= 14; $j++) {
        $c_curr = (float) $raw_data[$i - $j + 1][4];
        $c_prev = (float) $raw_data[$i - $j][4];
        if ($c_curr > $c_prev)     $up_days++;
        elseif ($c_curr < $c_prev) $down_days++;
    }
    $rsi_proxy = $up_days / max($down_days, 1);

    // ── FEATURE 2: Stochastic Oscillator 15,3,3 — Slow %D ────────────────
    //
    // The Stochastic measures where today's close sits within its 15-session
    // high-low channel:
    //
    //   Fast %K = ((Close − Low₁₅) / (High₁₅ − Low₁₅)) × 100
    //
    //   100 = close at the very top of its 15-day range → momentum peak
    //     0 = close at the very bottom                  → momentum trough
    //
    // Raw %K is too volatile to trade directly.  Double smoothing with
    // 3-day SMAs produces the Slow %D signal:
    //
    //   %D      = 3-day SMA of %K       (first smooth)
    //   Slow %D = 3-day SMA of %D       (second smooth)
    //
    // Slow %D is the standard output of the Stochastic 15,3,3 and is the
    // version plotted in most professional charting platforms.  Values above
    // 80 indicate an overbought condition; below 20 indicates oversold.
    //
    // Derivation: to compute Slow %D at day i we need:
    //   %K[i], %K[i−1], %K[i−2]  → %D[i]
    //   %K[i−1], %K[i−2], %K[i−3] → %D[i−1]
    //   %K[i−2], %K[i−3], %K[i−4] → %D[i−2]
    //   Oldest data consumed: High/Low at i−4−14 = i−18  (day 37 at i=55 ✓)

    // Compute Fast %K for the 5 days needed: indices i−4 .. i
    $fastK = [];
    for ($offset = 4; $offset >= 0; $offset--) {
        $day     = $i - $offset;
        $close_k = (float) $raw_data[$day][4];
        $low15   = PHP_FLOAT_MAX;
        $high15  = PHP_FLOAT_MIN;
        for ($k = 0; $k < 15; $k++) {
            $h = (float) $raw_data[$day - $k][2];   // High column
            $l = (float) $raw_data[$day - $k][3];   // Low column
            if ($h > $high15) $high15 = $h;
            if ($l < $low15)  $low15  = $l;
        }
        $range = $high15 - $low15;
        // Zero range means the stock traded completely flat for 15 sessions —
        // theoretically impossible for liquid stocks, so NAN triggers imputation.
        $fastK[] = ($range > 1e-9) ? (($close_k - $low15) / $range) * 100.0 : NAN;
    }
    // $fastK[0] = %K[i−4] ... $fastK[4] = %K[i]

    // %D = 3-day SMA of %K  (three values needed for Slow %D)
    $percentD = [];
    for ($k = 0; $k <= 2; $k++) {
        $sum = $fastK[$k] + $fastK[$k + 1] + $fastK[$k + 2];
        // Propagate NaN through the averages so imputer handles them cleanly
        $percentD[$k] = (is_nan($sum)) ? NAN : $sum / 3.0;
    }
    // $percentD[0] = %D[i−2], $percentD[1] = %D[i−1], $percentD[2] = %D[i]

    // Slow %D = 3-day SMA of %D
    $dSum  = $percentD[0] + $percentD[1] + $percentD[2];
    $slowD = (is_nan($dSum)) ? NAN : $dSum / 3.0;

    // ── FEATURE 3: Turtle 55-Day Breakout ─────────────────────────────────
    //
    // The Turtle Trading system (Richard Dennis, 1983) is one of the most
    // famous trend-following strategies in quantitative finance.  The core
    // rule: BUY when price breaks above the highest high of the last 55 days.
    //
    // We encode this as a continuous ratio:
    //   > 1.0 → price has broken above the 55-day structural high  (breakout)
    //   = 1.0 → price is exactly at the 55-day high
    //   < 1.0 → price is within the 55-day consolidation range (no signal)
    //
    // This feature captures macro trend breakouts that drive sustained weekly
    // moves — the type of directional signal most difficult to achieve with
    // short-window indicators alone.
    $high55 = PHP_FLOAT_MIN;
    for ($j = 0; $j < 55; $j++) {
        $h = (float) $raw_data[$i - $j][2];
        if ($h > $high55) $high55 = $h;
    }
    // $high55 > 0 guaranteed for real price data; ratio is safe without guard
    $turtle_breakout = $today_close / $high55;

    // ── FEATURE 4: Weekly Volatility Regime ───────────────────────────────
    //
    // The 5-session true range (today inclusive), normalised by today's close.
    //
    //   High value → wide-range week: large swings possible, higher risk.
    //   Low value  → tight coil: possible breakout setup but smaller swings.
    //
    // This feature acts as a REGIME SELECTOR: XGBoost can learn that the
    // Turtle breakout signal is more reliable in low-volatility consolidations
    // than in already high-volatility weeks.
    $week_high = PHP_FLOAT_MIN;
    $week_low  = PHP_FLOAT_MAX;
    for ($j = 0; $j < 5; $j++) {
        $h = (float) $raw_data[$i - $j][2];
        $l = (float) $raw_data[$i - $j][3];
        if ($h > $week_high) $week_high = $h;
        if ($l < $week_low)  $week_low  = $l;
    }
    $weekly_volatility = ($week_high - $week_low) / $today_close;

    // ── FEATURE 5: MACD Proxy (12-Day SMA − 26-Day SMA) / 26-Day SMA ─────
    //
    // The MACD (Moving Average Convergence Divergence) is the most widely
    // used momentum indicator by institutional traders.  It measures whether
    // short-term momentum (SMA₁₂) is accelerating above or decelerating
    // below medium-term momentum (SMA₂₆).
    //
    //   > 0 → short-term trend faster than medium-term → bullish momentum
    //   < 0 → short-term trend slower than medium-term → bearish momentum
    //   = 0 → crossover point → potential trend change
    //
    // Normalising by SMA₂₆ makes the ratio scale-invariant, so a $5 stock
    // and a $500 stock produce comparable MACD signals.
    $sum12 = 0.0;
    for ($j = 0; $j < 12; $j++) $sum12 += (float) $raw_data[$i - $j][4];
    $sma12 = $sum12 / 12.0;

    $sum26 = 0.0;
    for ($j = 0; $j < 26; $j++) $sum26 += (float) $raw_data[$i - $j][4];
    $sma26 = $sum26 / 26.0;

    // SMA₂₆ of real prices cannot be 0; NAN guard is a safety net only.
    $macd_proxy = ($sma26 > 1e-9) ? ($sma12 - $sma26) / $sma26 : NAN;

    $engineered_X[] = [$rsi_proxy, $slowD, $turtle_breakout, $weekly_volatility, $macd_proxy];
    $target_y[]     = $target_return;
}

$num_samples  = count($engineered_X);
$num_features = 5;

if ($num_samples < 50) {
    fwrite(STDERR, "ERROR: Too few samples ({$num_samples}) after feature engineering.\n");
    exit(1);
}

printf(
    "  Engineered matrix : [%d samples × %d features]\n  Features          : RSI_proxy | Stoch_SlowD | Turtle_55 | WeekVol | MACD_proxy\n  Target range      : [%.4f, %.4f]  (5-day forward return)\n",
    $num_samples,
    $num_features,
    min($target_y),
    max($target_y)
);

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 3: Imputation, Pipeline & XGBoost Tuning
//
//  STRICT RULE: Never shuffle financial time series.
//  Train on the oldest 80% of samples, test on the most recent 20%.
//  Any other split introduces LOOK-AHEAD BIAS — the model would learn from
//  data it could not have had during a real live trade.
//
//  MEMORY RULE: Split PHP arrays FIRST, then build Tensors.
//  Avoids allocating one monolithic [n_total × 5] Tensor that gets
//  immediately discarded after slicing.
//
//  PIPELINE STEPS
//  1. SimpleImputer(median)  — Fills any NaN values (Stochastic zero-range,
//                              MACD safety) with the per-column median.
//                              Medians are computed on the TRAINING SET ONLY —
//                              no data leakage from the test period.
//  2. StandardScaler         — Zero-mean, unit-variance normalisation.
//                              Ensures each of the 5 features contributes
//                              equally to the initial splits in XGBoost.
//  3. XGBRegressor           — Gradient-boosted trees tuned for weekly swings.
// ─────────────────────────────────────────────────────────────────────────────

banner('Imputation + Pipeline Training (SimpleImputer → Scaler → XGBoost)', 'Phase 3');

$split_idx = (int) ($num_samples * 0.8);
$test_size = $num_samples - $split_idx;

printf("  Sequential split  → Train: %d days | Test: %d days\n", $split_idx, $test_size);

// Split raw PHP arrays sequentially, then materialise as Tensors.
$X_train_arr = array_slice($engineered_X, 0, $split_idx);
$y_train_arr = array_slice($target_y,     0, $split_idx);
$X_test_arr  = array_slice($engineered_X, $split_idx);
$y_test_arr  = array_slice($target_y,     $split_idx);

$X_train = Tensor::fromArray($X_train_arr, [$split_idx, $num_features]);
$y_train = Tensor::fromArray($y_train_arr, [$split_idx]);
$X_test  = Tensor::fromArray($X_test_arr,  [$test_size,  $num_features]);
$y_test  = Tensor::fromArray($y_test_arr,  [$test_size]);

// ── XGBoost hyperparameter rationale ─────────────────────────────────────
//
//  n_estimators = 200  — more trees to capture the slow, broad weekly
//                        patterns; with lr=0.01 each tree contributes only
//                        1% of its weight, so convergence is gradual.
//
//  max_depth = 4       — depth-4 trees have max 16 leaves, enough to learn
//                        interactions (e.g. "overbought RSI AND Turtle
//                        breakout → strong sell") without memorising days.
//
//  learning_rate = 0.01 — very slow shrinkage suppresses daily-noise
//                         overfitting in favour of weekly-trend consensus
//                         across the full 200-tree ensemble.
//
//  subsample = 0.7     — each tree sees a random 70% of rows (Friedman-
//                        style stochastic boosting), breaking the serial
//                        autocorrelation in financial time series and
//                        decorrelating individual trees.

$pipeline = new Pipeline([
    ['imputer', new SimpleImputer(strategy: 'median')],
    ['scaler',  new StandardScaler()],
    ['xgb',     new XGBRegressor(
        n_estimators:  200,
        max_depth:     4,
        learning_rate: 0.01,
        subsample:     0.7,
        random_state:  42,
    )],
]);

echo "  Fitting pipeline (SimpleImputer → StandardScaler → XGBRegressor)...\n";
$pipeline->fit($X_train, $y_train);
echo "  Pipeline fitted.\n";

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 4: The "Quant" Evaluation — Confidence Thresholding
//
//  NAIVE EVALUATION PROBLEM
//  Raw directional accuracy treats a predicted +0.01% the same as +3.0%.
//  The model is far less certain about the direction of a near-zero
//  prediction than a large one.  Acting on every signal — including the
//  whispers — loses money to spread and commission.
//
//  SOLUTION: TRADE ONLY ON HIGH-CONVICTION SIGNALS
//  We define confidence_threshold = 1.5% (≈ one weekly σ of GOOGL returns).
//  When |predicted_return| < 1.5% → no trade, signal is too weak.
//  When |predicted_return| ≥ 1.5% → trade taken, check direction.
//
//  WIN CONDITION: actual return moved in the predicted direction.
//  We require only directional correctness — sufficient to build a
//  profitable systematic long/short strategy.
//
//  55%+ WIN RATE = GOLD STANDARD
//  A 55%+ win rate on confidence-filtered swing trades using these five
//  macro features (RSI, Stochastic, Turtle 55-day, weekly vol, MACD) is
//  the gold standard for quantitative hedge funds.  Most top-tier systematic
//  trend-following strategies operate at 55–65% directional accuracy after
//  filtering.  Below 50% the strategy is loss-making regardless of leverage.
// ─────────────────────────────────────────────────────────────────────────────

banner("Quant Evaluation — Confidence Thresholding (1.5% Filter)", 'Phase 4');

$predictions = $pipeline->predict($X_test);

// ── Standard regression metrics (all test samples) ───────────────────────

$mae  = Metrics::mean_absolute_error($y_test, $predictions);
$mse  = Metrics::mean_squared_error($y_test, $predictions);
$rmse = sqrt($mse);
$r2   = Metrics::r2_score($y_test, $predictions);

// ── Confidence-filtered directional accuracy ─────────────────────────────

$confidence_threshold = 0.015;   // 1.5% minimum predicted move to trade

$skipped      = 0;   // predictions within ±1.5% — no trade taken
$trades_taken = 0;   // predictions outside ±1.5% — trade executed
$trades_won   = 0;   // trades where direction matched actual move

for ($i = 0; $i < $test_size; $i++) {
    $predicted = (float) $predictions->buffer[$i];
    $actual    = (float) $y_test->buffer[$i];

    if (abs($predicted) < $confidence_threshold) {
        // The model predicts a near-zero return.  This is the "I don't know"
        // zone — the signal is weaker than typical market friction costs.
        $skipped++;
        continue;
    }

    // High-conviction signal: the model sees a meaningful weekly swing.
    $trades_taken++;

    // Win: predicted direction (positive or negative) matches actual direction.
    if (($predicted > 0.0) === ($actual > 0.0)) {
        $trades_won++;
    }
}

// ── Derived metrics ───────────────────────────────────────────────────────

$win_rate    = ($trades_taken > 0) ? ($trades_won  / $trades_taken) * 100.0 : 0.0;
$filter_rate = ($test_size    > 0) ? ($skipped      / $test_size)   * 100.0 : 0.0;
$trade_rate  = ($test_size    > 0) ? ($trades_taken / $test_size)   * 100.0 : 0.0;

// ── Terminal report ───────────────────────────────────────────────────────

echo "\n── Regression Quality (all test samples) ─────────────────────────────\n";
printf("  MAE (avg error per week)    : %.5f  (%.3f%%)\n", $mae,  $mae  * 100);
printf("  RMSE                        : %.5f  (%.3f%%)\n", $rmse, $rmse * 100);
printf("  R² score                    : %.4f\n",            $r2);

echo "\n── Confidence-Filtered Trade Simulation ──────────────────────────────\n";
printf("  Total days in test set      : %d\n",      $test_size);
printf("  Skipped (|pred| < 1.5%%)   : %d  (%.1f%% of signals filtered)\n", $skipped, $filter_rate);
printf("  Trades taken (|pred| ≥ 1.5%%): %d  (%.1f%% of test days)\n",       $trades_taken, $trade_rate);
printf("  Trades won (correct dir.)   : %d\n",      $trades_won);
printf("  Directional Win Rate        : %.2f%%\n",   $win_rate);

echo "\n── Interpretation ────────────────────────────────────────────────────\n";

if ($trades_taken === 0) {
    echo "  [NOTE] Zero high-confidence signals generated.\n";
    echo "  All predictions fall within the ±1.5% noise band.\n";
    echo "  Consider lowering confidence_threshold or adding features\n";
    echo "  (e.g. earnings proximity, sector rotation, implied volatility).\n";
} elseif ($win_rate >= 60.0) {
    printf("  [OUTSTANDING] %.2f%% win rate — institutional-grade alpha.\n", $win_rate);
    echo "  This edge generates positive risk-adjusted returns even after\n";
    echo "  bid-ask spread and commission friction.\n";
    echo "  Next steps: walk-forward validation, Kelly position sizing.\n";
} elseif ($win_rate >= 55.0) {
    printf("  [EDGE DETECTED] %.2f%% win rate exceeds the 55%% quant benchmark.\n", $win_rate);
    echo "  A genuine, exploitable statistical edge exists on swing trades.\n";
    echo "  A live deployment would add stop-losses and position sizing.\n";
} elseif ($win_rate >= 50.0) {
    printf("  [MARGINAL] %.2f%% win rate — above chance, below the 55%% benchmark.\n", $win_rate);
    echo "  Transaction costs (spread + commission) would likely erode this edge.\n";
    echo "  Consider tightening the threshold or adding macro-economic features.\n";
} else {
    printf("  [NO EDGE] %.2f%% win rate — no directional advantage at 1.5%% threshold.\n", $win_rate);
    echo "  Filtered signals are no better than a coin flip.\n";
    echo "  Options: wider train window, higher threshold, earnings calendar.\n";
}

echo "──────────────────────────────────────────────────────────────────────\n\n";
