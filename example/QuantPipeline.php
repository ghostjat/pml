<?php

declare(strict_types=1);

/**
 * QuantPipeline.php
 *
 * Production-ready stock price prediction pipeline using Pml.
 * Ports the sigma_coding Random Forest Price Prediction notebook to PHP.
 *
 * Pipeline stages:
 *   1. ETL          – Stream-load OHLC CSV grouped by ticker
 *   2. Features     – Lag, SMA, EMA, RSI, rolling-std, price ratios (no lookahead)
 *   3. Dataset      – Build Pml\Dataset from PHP arrays
 *   4. Scaling      – StandardScaler fitted on train split only
 *   5. Training     – GradientBoostingRegressor (next-close) + RandomForestClassifier (direction)
 *   6. Evaluation   – MAE, RMSE, R², Accuracy
 *
 * Dataset format: Ticker,Date,Open,High,Low,Close,Volume  (weekly bars, lm250.csv)
 */

require_once __DIR__ . '/../vendor/autoload.php';

use Pml\Dataset;
use Pml\Tensor;
use Pml\Transformers\StandardScaler;
use Pml\Estimators\Regression\GradientBoostingRegressor;
use Pml\Estimators\Classifiers\RandomForestClassifier;
use Pml\Metrics\Regression\MeanAbsoluteError;
use Pml\Metrics\Regression\RootMeanSquaredError;
use Pml\Metrics\Regression\RSquared;
use Pml\Metrics\Classification\Accuracy;

// ============================================================
// 1. ETL — OhlcLoader
// ============================================================

/**
 * Streams an OHLC CSV file and groups rows by ticker symbol.
 * Uses a generator internally to stay memory-efficient even for
 * very large files; the final grouped array is the only structure
 * retained in memory.
 *
 * Expected CSV header: Ticker,Date,Open,High,Low,Close,Volume
 */
final class OhlcLoader
{
    /**
     * Load and group all rows by ticker.
     *
     * @return array<string, list<array{date:string,open:float,high:float,low:float,close:float,volume:float}>>
     */
    public static function loadGrouped(string $csvPath): array
    {
        $handle = fopen($csvPath, 'r');
        if ($handle === false) {
            throw new \RuntimeException("Cannot open CSV: {$csvPath}");
        }

        // Discard header row
        fgetcsv($handle);

        $byTicker = [];

        while (($row = fgetcsv($handle)) !== false) {
            // Guard against malformed / short rows
            if (count($row) < 7) {
                continue;
            }

            [$ticker, $date, $open, $high, $low, $close, $volume] = $row;

            // Skip rows with missing or non-numeric OHLCV values
            if (!is_numeric($open) || !is_numeric($high) || !is_numeric($low)
                    || !is_numeric($close) || !is_numeric($volume)) {
                continue;
            }

            $close = (float) $close;
            if ($close <= 0.0) {
                continue; // Ignore zero/negative prices (bad data)
            }

            $byTicker[$ticker][] = [
                'date'   => $date,
                'open'   => (float) $open,
                'high'   => (float) $high,
                'low'    => (float) $low,
                'close'  => $close,
                'volume' => (float) $volume,
            ];
        }

        fclose($handle);

        return $byTicker;
    }
}

// ============================================================
// 2. Feature Engineering
// ============================================================

/**
 * Computes a 16-feature vector for each weekly bar of a single ticker.
 *
 * Features (all computed at time t using only data up to t — no lookahead):
 *   [0]  lag1_close     – Close price one period ago
 *   [1]  lag2_close     – Close price two periods ago
 *   [2]  return_1w      – Weekly return: (close[t] - close[t-1]) / close[t-1]
 *   [3]  return_2w      – Two-week return
 *   [4]  sma5           – 5-period simple moving average of Close
 *   [5]  sma10          – 10-period SMA
 *   [6]  sma20          – 20-period SMA
 *   [7]  ema10          – 10-period exponential moving average (Wilder)
 *   [8]  rsi14          – 14-period RSI
 *   [9]  rolling_std10  – 10-period rolling standard deviation of Close
 *   [10] hl_range       – (High - Low) / Close  (normalised range)
 *   [11] oc_diff        – (Open - Close) / Close (candle body)
 *   [12] volume_change  – (Volume[t] - Volume[t-1]) / Volume[t-1]
 *   [13] dist_sma5      – (Close - SMA5) / Close
 *   [14] dist_sma10     – (Close - SMA10) / Close
 *   [15] dist_sma20     – (Close - SMA20) / Close
 *
 * Targets:
 *   Regression     – next period's raw Close price
 *   Classification – 1 (up) / 0 (down) based on next close vs current
 *
 * The minimum warm-up requirement is 20 periods (SMA20 + lag2), so the
 * first 20 rows of each ticker are discarded, and the last row is dropped
 * because no target is available.
 */
final class FeatureEngineer
{
    private const WARMUP = 20;

    /**
     * @param  list<array{date:string,open:float,high:float,low:float,close:float,volume:float}> $rows
     * @return array{0: list<float[]>, 1: list<float>, 2: list<float>}
     *         [samples, regression_labels, classification_labels]
     */
    public static function engineer(array $rows): array
    {
        $n = count($rows);

        // Need at least WARMUP + 1 rows to produce any valid sample with a target
        if ($n < self::WARMUP + 2) {
            return [[], [], []];
        }

        // Pre-extract price/volume columns for cache-friendly iteration
        $closes  = array_column($rows, 'close');
        $opens   = array_column($rows, 'open');
        $highs   = array_column($rows, 'high');
        $lows    = array_column($rows, 'low');
        $volumes = array_column($rows, 'volume');

        // Pre-compute series-level indicators (full series, O(N))
        $ema10 = self::computeEma($closes, 10);
        $rsi14 = self::computeRsi($closes, 14);

        $samples   = [];
        $labelsReg = [];
        $labelsCls = [];

        // Valid index range: t in [WARMUP .. n-2]
        // – t >= WARMUP  : ensures lag2, SMA20, EMA10 seed, RSI14 seed are available
        // – t <= n-2     : ensures close[t+1] (target) exists
        for ($t = self::WARMUP; $t <= $n - 2; $t++) {

            $c  = $closes[$t];
            $c1 = $closes[$t - 1];
            $c2 = $closes[$t - 2];
            $v  = $volumes[$t];
            $v1 = $volumes[$t - 1];

            // Returns (guard against zero-price edge cases in dirty data)
            $ret1 = $c1 > 0.0 ? ($c - $c1) / $c1 : 0.0;
            $ret2 = $c2 > 0.0 ? ($c1 - $c2) / $c2 : 0.0;

            // Moving averages
            $sma5  = self::sma($closes, $t, 5);
            $sma10 = self::sma($closes, $t, 10);
            $sma20 = self::sma($closes, $t, 20);

            // Rolling volatility
            $std10 = self::rollingStd($closes, $t, 10);

            // Price-based features (normalised to be scale-invariant)
            $hlRange = $c > 0.0 ? ($highs[$t] - $lows[$t]) / $c : 0.0;
            $ocDiff  = $c > 0.0 ? ($opens[$t] - $c) / $c        : 0.0;
            $volChg  = $v1 > 0.0 ? ($v - $v1) / $v1              : 0.0;

            // Distance from price to moving averages
            $distSma5  = $c > 0.0 ? ($c - $sma5)  / $c : 0.0;
            $distSma10 = $c > 0.0 ? ($c - $sma10) / $c : 0.0;
            $distSma20 = $c > 0.0 ? ($c - $sma20) / $c : 0.0;

            $samples[] = [
                $c1,            // lag1_close
                $c2,            // lag2_close
                $ret1,          // return_1w
                $ret2,          // return_2w
                $sma5,          // sma5
                $sma10,         // sma10
                $sma20,         // sma20
                $ema10[$t],     // ema10
                $rsi14[$t],     // rsi14
                $std10,         // rolling_std10
                $hlRange,       // hl_range
                $ocDiff,        // oc_diff
                $volChg,        // volume_change
                $distSma5,      // dist_sma5
                $distSma10,     // dist_sma10
                $distSma20,     // dist_sma20
            ];

            $nextClose   = $closes[$t + 1];
            $labelsReg[] = $nextClose;
            $labelsCls[] = $nextClose > $c ? 1.0 : 0.0;
        }

        return [$samples, $labelsReg, $labelsCls];
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    /** Simple Moving Average of the last $period values ending at index $t */
    private static function sma(array $series, int $t, int $period): float
    {
        $sum = 0.0;
        for ($i = $t - $period + 1; $i <= $t; $i++) {
            $sum += $series[$i];
        }
        return $sum / $period;
    }

    /** Population standard deviation of the last $period values ending at $t */
    private static function rollingStd(array $series, int $t, int $period): float
    {
        $window = array_slice($series, $t - $period + 1, $period);
        $mean   = array_sum($window) / $period;
        $var    = 0.0;
        foreach ($window as $v) {
            $var += ($v - $mean) ** 2;
        }
        return sqrt($var / $period);
    }

    /**
     * Computes the full EMA series using standard exponential smoothing.
     * Seeded with the SMA of the first $period values.
     *
     * @return float[]  Index-aligned with $series
     */
    private static function computeEma(array $series, int $period): array
    {
        $n   = count($series);
        $ema = array_fill(0, $n, 0.0);
        $k   = 2.0 / ($period + 1);

        // Seed: SMA of the first `period` values
        $seed = 0.0;
        for ($i = 0; $i < $period; $i++) {
            $seed += $series[$i];
        }
        $ema[$period - 1] = $seed / $period;

        for ($i = $period; $i < $n; $i++) {
            $ema[$i] = $series[$i] * $k + $ema[$i - 1] * (1.0 - $k);
        }

        return $ema;
    }

    /**
     * Computes Wilder's RSI for the full series.
     * Returns 50.0 (neutral) for indices before the first valid RSI value.
     *
     * @return float[]  Index-aligned with $series
     */
    private static function computeRsi(array $series, int $period): array
    {
        $n   = count($series);
        $rsi = array_fill(0, $n, 50.0);

        if ($n < $period + 1) {
            return $rsi;
        }

        // Seed average gain / loss over the first $period price changes
        $avgGain = 0.0;
        $avgLoss = 0.0;
        for ($i = 1; $i <= $period; $i++) {
            $delta = $series[$i] - $series[$i - 1];
            if ($delta > 0.0) {
                $avgGain += $delta;
            } else {
                $avgLoss += -$delta;
            }
        }
        $avgGain /= $period;
        $avgLoss /= $period;

        $rsi[$period] = $avgLoss < 1e-10
            ? 100.0
            : 100.0 - (100.0 / (1.0 + $avgGain / $avgLoss));

        // Wilder's smoothed rolling RSI
        for ($i = $period + 1; $i < $n; $i++) {
            $delta = $series[$i] - $series[$i - 1];
            $gain  = max(0.0, $delta);
            $loss  = max(0.0, -$delta);

            $avgGain = ($avgGain * ($period - 1) + $gain) / $period;
            $avgLoss = ($avgLoss * ($period - 1) + $loss) / $period;

            $rsi[$i] = $avgLoss < 1e-10
                ? 100.0
                : 100.0 - (100.0 / (1.0 + $avgGain / $avgLoss));
        }

        return $rsi;
    }
}

// ============================================================
// 3. QuantPipeline — Orchestrator
// ============================================================

/**
 * Orchestrates the full ETL → Feature Engineering → Dataset →
 * Scaling → Training → Evaluation pipeline.
 */
final class QuantPipeline
{
    /** Feature names — used only for reporting */
    private const FEATURE_NAMES = [
        'lag1_close', 'lag2_close', 'return_1w', 'return_2w',
        'sma5', 'sma10', 'sma20', 'ema10', 'rsi14', 'rolling_std10',
        'hl_range', 'oc_diff', 'volume_change',
        'dist_sma5', 'dist_sma10', 'dist_sma20',
    ];

    /**
     * @param string   $csvPath     Path to the OHLC CSV file
     * @param bool     $verbose     Print progress to stdout
     * @param int|null $maxSamples  Cap total feature rows (null = use all).
     *                              Set to ~5000 for a quick smoke-test run.
     * @param int      $gbrTrees    GBR estimators  (production: 100–300)
     * @param int      $gbrDepth    GBR max depth   (production: 3–5)
     * @param int      $rfcTrees    RFC estimators  (production: 100–200)
     * @param int      $rfcDepth    RFC max depth   (production: 6–10)
     */
    public function __construct(
        private readonly string $csvPath,
        private readonly bool   $verbose    = true,
        private readonly ?int   $maxSamples = null,
        private readonly int    $gbrTrees   = 100,
        private readonly int    $gbrDepth   = 4,
        private readonly int    $rfcTrees   = 100,
        private readonly int    $rfcDepth   = 8,
    ) {}

    public function run(): void
    {
        $this->log("╔══════════════════════════════════════════════════╗");
        $this->log("║        QuantPipeline — Stock Price Forecast       ║");
        $this->log("╚══════════════════════════════════════════════════╝");
        $this->log("Dataset : {$this->csvPath}");
        $this->log("Features: " . implode(', ', self::FEATURE_NAMES));

        // ── Stage 1: ETL ─────────────────────────────────────────────────────
        $this->log("\n[1/6] Loading OHLC data...");
        $t0       = microtime(true);
        $byTicker = OhlcLoader::loadGrouped($this->csvPath);
        $elapsed  = round((microtime(true) - $t0) * 1000, 1);

        $tickerCount = count($byTicker);
        $totalBars   = array_sum(array_map('count', $byTicker));
        $this->log("  Tickers  : {$tickerCount}");
        $this->log("  Total bars: {$totalBars}  ({$elapsed} ms)");

        // ── Stage 2: Feature Engineering ─────────────────────────────────────
        $this->log("\n[2/6] Engineering features...");
        $t0 = microtime(true);

        $allSamples   = [];
        $allLabelsReg = [];
        $allLabelsCls = [];
        $skippedTickers = 0;

        foreach ($byTicker as $ticker => $rows) {
            [$samples, $labelsReg, $labelsCls] = FeatureEngineer::engineer($rows);

            if (empty($samples)) {
                $skippedTickers++;
                continue;
            }

            foreach ($samples as $idx => $row) {
                $allSamples[]   = $row;
                $allLabelsReg[] = $labelsReg[$idx];
                $allLabelsCls[] = $labelsCls[$idx];
            }
        }

        $elapsed  = round((microtime(true) - $t0) * 1000, 1);
        $rowCount = count($allSamples);
        $featCount = count($allSamples[0] ?? []);

        // Optional cap: randomly sub-sample to keep training fast during testing.
        // Uses a stride sample (not random) to preserve some temporal diversity.
        if ($this->maxSamples !== null && $rowCount > $this->maxSamples) {
            $step   = (int) ceil($rowCount / $this->maxSamples);
            $keys   = range(0, $rowCount - 1, $step);
            $allSamples   = array_values(array_intersect_key($allSamples,   array_flip($keys)));
            $allLabelsReg = array_values(array_intersect_key($allLabelsReg, array_flip($keys)));
            $allLabelsCls = array_values(array_intersect_key($allLabelsCls, array_flip($keys)));
            $rowCount = count($allSamples);
            $this->log("  ⚠  Sampled down to {$rowCount} rows (maxSamples={$this->maxSamples})");
        }

        $this->log("  Rows generated : {$rowCount}");
        $this->log("  Features/row   : {$featCount}");
        $this->log("  Skipped tickers: {$skippedTickers} (insufficient history)");
        $this->log("  Elapsed        : {$elapsed} ms");

        if ($rowCount < 100) {
            throw new \RuntimeException("Insufficient data: only {$rowCount} rows after feature engineering.");
        }

        // ── Stage 3: Dataset Construction ────────────────────────────────────
        $this->log("\n[3/6] Building Pml\\Dataset objects...");

        $datasetReg = Dataset::fromArray($allSamples, $allLabelsReg);
        $datasetCls = Dataset::fromArray($allSamples, $allLabelsCls);

        // Time-ordered 80/20 split — no shuffle to prevent data leakage
        [$trainReg, $testReg] = $datasetReg->split(0.8);
        [$trainCls, $testCls] = $datasetCls->split(0.8);

        $this->log("  Train rows : {$trainReg->numRows()}");
        $this->log("  Test rows  : {$testReg->numRows()}");

        // ── Stage 4: Scaling ─────────────────────────────────────────────────
        $this->log("\n[4/6] Fitting StandardScaler on training data...");

        $scaler = new StandardScaler();
        $scaler->fit($trainReg);  // Fit on TRAINING set only — no test leakage

        $trainRegScaled = $scaler->transform($trainReg);
        $testRegScaled  = $scaler->transform($testReg);

        // Classification tasks share the same feature space → reuse scaler
        $trainClsScaled = new Dataset($trainRegScaled->samples(), $trainCls->labels());
        $testClsScaled  = new Dataset($testRegScaled->samples(),  $testCls->labels());

        $this->log("  StandardScaler fitted (mean-centred, unit-variance).");

        // ── Stage 5: Training ─────────────────────────────────────────────────
        $this->log("\n[5/6] Training models...");

        // GradientBoostingRegressor — predicts next-period Close price
        $this->log("  [GBR] GradientBoostingRegressor(n={$this->gbrTrees}, lr=0.05, depth={$this->gbrDepth}, subsample=0.8)");
        $t0 = microtime(true);

        $gbr = new GradientBoostingRegressor(
            nEstimators: $this->gbrTrees,
            learningRate: 0.05,
            maxDepth: $this->gbrDepth,
            subsample: 0.8,
        );
        $gbr->train($trainRegScaled);

        $elapsed = round((microtime(true) - $t0) * 1000, 1);
        $this->log("  [GBR] Done in {$elapsed} ms");

        // RandomForestClassifier — predicts price direction (Up / Down)
        $this->log("  [RFC] RandomForestClassifier(n={$this->rfcTrees}, depth={$this->rfcDepth}, minSplit=5)");
        $t0 = microtime(true);

        $rfc = new RandomForestClassifier(
            nEstimators: $this->rfcTrees,
            maxDepth: $this->rfcDepth,
            minSamplesSplit: 5,
        );
        $rfc->train($trainClsScaled);

        $elapsed = round((microtime(true) - $t0) * 1000, 1);
        $this->log("  [RFC] Done in {$elapsed} ms");

        // ── Stage 6: Evaluation ───────────────────────────────────────────────
        $this->log("\n[6/6] Evaluating on held-out test set...");

        // Regression
        $predReg = $gbr->predict($testRegScaled);
        $trueReg = $testRegScaled->labels();

        $mae  = (new MeanAbsoluteError())->score($predReg, $trueReg);
        $rmse = (new RootMeanSquaredError())->score($predReg, $trueReg);
        $r2   = (new RSquared())->score($predReg, $trueReg);

        $this->log("\n  ┌─────────────────────────────────────────────┐");
        $this->log("  │  Regression — Next-Close Price Prediction   │");
        $this->log("  ├─────────────────────────────────────────────┤");
        $this->log(sprintf("  │  MAE  : %10.4f                         │", $mae));
        $this->log(sprintf("  │  RMSE : %10.4f                         │", $rmse));
        $this->log(sprintf("  │  R²   : %10.4f                         │", $r2));
        $this->log("  └─────────────────────────────────────────────┘");

        // Classification
        $predCls = $rfc->predict($testClsScaled);
        $trueCls = $testClsScaled->labels();

        $accuracy = (new Accuracy())->score($predCls, $trueCls);

        $this->log("\n  ┌─────────────────────────────────────────────┐");
        $this->log("  │  Classification — Price Direction (Up/Down) │");
        $this->log("  ├─────────────────────────────────────────────┤");
        $this->log(sprintf("  │  Accuracy : %6.2f%%                         │", $accuracy * 100));
        $this->log("  └─────────────────────────────────────────────┘");

        // Sample output table
        $this->printSamplePredictions($predReg, $trueReg, $predCls, $trueCls, 8);

        // Prediction pipeline demo
        $this->log("\n═══════════ Prediction Pipeline Demo ═══════════");
        $this->demoPredict($testRegScaled, $gbr, $rfc, $scaler);

        $this->log("\n✓ Pipeline complete.");
    }

    // ── Reporting helpers ────────────────────────────────────────────────────

    private function printSamplePredictions(
        Tensor $predReg,
        Tensor $trueReg,
        Tensor $predCls,
        Tensor $trueCls,
        int    $n = 8,
    ): void {
        $this->log("\n  Sample predictions (first {$n} test rows):");
        $this->log(sprintf(
            "  %-5s  %-12s %-12s %-5s  %-5s  %-6s",
            "Row", "True Close", "Pred Close", "T.Dir", "P.Dir", "Error%"
        ));
        $this->log("  " . str_repeat("─", 52));

        $prA = $predReg->toFlatArray();
        $trA = $trueReg->toFlatArray();
        $pcA = $predCls->toFlatArray();
        $tcA = $trueCls->toFlatArray();

        for ($i = 0; $i < min($n, count($prA)); $i++) {
            $errPct = $trA[$i] > 0.0 ? abs($prA[$i] - $trA[$i]) / $trA[$i] * 100 : 0.0;
            $this->log(sprintf(
                "  %-5d  %-12.2f %-12.2f %-5s  %-5s  %5.2f%%",
                $i + 1,
                $trA[$i],
                $prA[$i],
                $tcA[$i] > 0.5 ? "UP" : "DOWN",
                $pcA[$i] > 0.5 ? "UP" : "DOWN",
                $errPct,
            ));
        }
    }

    /**
     * Demonstrates how to use the trained models for single-row inference.
     * Takes the last bar of the test set and predicts the following week.
     */
    private function demoPredict(
        Dataset                  $testScaled,
        GradientBoostingRegressor $gbr,
        RandomForestClassifier    $rfc,
        StandardScaler            $scaler,
    ): void {
        $this->log("  Predicting the next bar after the last test row...");

        // Extract the last row as a 1-sample Dataset
        $lastRow     = $testScaled->tail(1);
        $predPrice   = $gbr->predict($lastRow)->toFlatArray()[0];
        $predDir     = $rfc->predict($lastRow)->toFlatArray()[0];
        $truePrice   = $testScaled->labels()?->toFlatArray()[array_key_last($testScaled->labels()->toFlatArray())] ?? 0.0;

        $this->log(sprintf("  True next close : %.2f", $truePrice));
        $this->log(sprintf("  Predicted close : %.2f", $predPrice));
        $this->log(sprintf("  Predicted dir   : %s", $predDir > 0.5 ? "UP ▲" : "DOWN ▼"));
    }

    private function log(string $msg): void
    {
        if ($this->verbose) {
            echo $msg . PHP_EOL;
        }
    }
}

// ============================================================
// Entry Point
// ============================================================
//
// Usage:
//   php QuantPipeline.php              # Full production run (~162k rows)
//   php QuantPipeline.php --quick      # Quick smoke-test (~5k rows, 20 trees)
//   php QuantPipeline.php --max 20000  # Custom row cap
//

$csvPath = __DIR__ . '/../datasets/stocks/lm250.csv';

if (!file_exists($csvPath)) {
    fwrite(STDERR, "Error: Dataset not found at {$csvPath}\n");
    exit(1);
}

// Parse CLI flags
$opts       = getopt('', ['quick', 'max:']);
$quickRun   = isset($opts['quick']);
$maxSamples = isset($opts['max']) ? (int) $opts['max'] : null;

if ($quickRun) {
    // Quick mode: ~5k rows, smaller models — full pipeline verified in ~30 seconds
    $pipeline = new QuantPipeline(
        csvPath:    $csvPath,
        verbose:    true,
        maxSamples: 5000,
        gbrTrees:   20,
        gbrDepth:   3,
        rfcTrees:   20,
        rfcDepth:   5,
    );
} else {
    // Production mode: all rows, full model capacity
    $pipeline = new QuantPipeline(
        csvPath:    $csvPath,
        verbose:    true,
        maxSamples: $maxSamples,   // null = use all 162k rows
        gbrTrees:   100,
        gbrDepth:   4,
        rfcTrees:   100,
        rfcDepth:   8,
    );
}

$pipeline->run();
