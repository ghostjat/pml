<?php
declare(strict_types=1);
/**
 * MINI BLOOMBERG AI TERMINAL
 * ═══════════════════════════════════════════════════════════════════
 * A self-contained quantitative intelligence system that:
 *   • Detects market regimes (bull/bear/sideways) via clustering
 *   • Scores equity signals (momentum, value, quality) via GBDT
 *   • Flags volatility anomalies via IsolationForest
 *   • Generates a portfolio allocation recommendation
 *
 * This is what a quant startup prototype looks like in PML.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Estimators\Clusterers\KMeans;
use Pml\Estimators\Classifiers\GBDTClassifier;
use Pml\Estimators\AnomalyDetectors\IsolationForest;
use Pml\Transformers\StandardScaler;

section('Mini Bloomberg AI Terminal');
echo "  Initialising quantitative intelligence modules...\n";

mt_srand(2024);
$rng    = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);
$randn  = fn() => sqrt(-2 * log(lcg_value() + 1e-10)) * cos(2 * M_PI * lcg_value());

// ═══════════════════════════════════════════════════════════════════
// MODULE 1: MARKET REGIME DETECTION
// ═══════════════════════════════════════════════════════════════════
section('Module 1: Market Regime Detection');

// Daily macro features: sp500_ret, vix, credit_spread, yield_curve,
// dollar_index, commodity_index, breadth
$regimeData = [];
for ($d = 0; $d < 2000; $d++) {
    $regime = mt_rand(0, 2);  // 0=bull, 1=bear, 2=sideways
    $regimeData[] = match($regime) {
        0 => [$rng(0.0, 0.8), $rng(10, 18), $rng(0.5, 1.0), $rng(0.5, 1.5), $rng(-0.5, 0.5), $rng(0.5, 2.0), $rng(0.6, 0.9)],
        1 => [$rng(-0.8, 0.1), $rng(25, 50), $rng(2.0, 5.0), $rng(-1.5, 0.0), $rng(0.5, 2.0), $rng(-2.0, 0.0), $rng(0.2, 0.5)],
        2 => [$rng(-0.2, 0.2), $rng(15, 25), $rng(0.8, 1.8), $rng(-0.3, 0.5), $rng(-0.3, 0.3), $rng(-0.5, 0.5), $rng(0.45, 0.6)],
    };
}

$regimeScaler = new StandardScaler();
$regimeDs     = Dataset::fromArray($regimeData);
$regimeScaler->fit($regimeDs);
$regimeScaled = $regimeScaler->transform($regimeDs);

$regimeClustering = new KMeans(k: 3, maxIter: 200);
$regimeClustering->train($regimeScaled);

// Score today's conditions
$todayMacro   = [$rng(0.2, 0.5), 16.5, 0.8, 0.9, -0.1, 1.2, 0.72];
$todayScaled  = $regimeScaler->transform(Dataset::fromArray([$todayMacro]));
$todayRegime  = (int)$regimeClustering->predict($todayScaled)->toFlatArray()[0];

$regimeNames  = ['Bull Market 🐂', 'Bear Market 🐻', 'Sideways / Range 〰️'];
// Map cluster id to most-likely regime based on VIX (feature[1])
$clusterMeans = [];
$clusterData  = $regimeClustering->predict($regimeScaled)->toFlatArray();
foreach ($regimeData as $i => $row) {
    $clusterMeans[(int)$clusterData[$i]][] = $row[1]; // VIX
}
$clusterVix = array_map(fn($v) => array_sum($v) / count($v), $clusterMeans);
arsort($clusterVix);  // highest VIX = bear, lowest = bull
$vixRanks   = array_keys($clusterVix);
$regimeMap  = [$vixRanks[2] => 0, $vixRanks[0] => 1, $vixRanks[1] => 2];
$mappedRegime = $regimeMap[$todayRegime] ?? 0;

printf("  Current market regime : %s\n", $regimeNames[$mappedRegime]);
printf("  VIX reading            : %.1f\n", 16.5);

// ═══════════════════════════════════════════════════════════════════
// MODULE 2: EQUITY SIGNAL SCORING
// ═══════════════════════════════════════════════════════════════════
section('Module 2: Equity Signal Model');

// Build factor model (momentum, value, quality signals → next month return quintile)
$factorRows = []; $factorLbls = [];
for ($i = 0; $i < 5000; $i++) {
    $mom = $randn() * 0.3; $val = $randn() * 0.25; $qlty = $randn() * 0.4 + 0.1;
    $vol = abs($randn()) * 0.15 + 0.05; $size = $randn() * 0.3; $earns = $randn() * 0.2;
    $ret = 0.002 + $mom*0.07 + $val*0.04 + $qlty*0.03 - $vol*0.06 + $earns*0.05 + $randn()*0.08;
    $factorRows[] = [$mom, $val, $qlty, $vol, $size, $earns];
    $factorLbls[] = $ret;
}

// Convert to quintiles for classification
$sorted = $factorLbls; sort($sorted); $n = count($sorted);
$edges  = [$sorted[(int)($n*0.2)], $sorted[(int)($n*0.4)], $sorted[(int)($n*0.6)], $sorted[(int)($n*0.8)]];
$qLbls  = array_map(fn($r) => match(true) {
    $r < $edges[0] => 0.0, $r < $edges[1] => 1.0, $r < $edges[2] => 2.0,
    $r < $edges[3] => 3.0, default => 4.0,
}, $factorLbls);

$factorDs = Dataset::fromArray($factorRows, $qLbls);
$signalModel = new GBDTClassifier(nEstimators: 100, maxDepth: 4, lr: 0.08);
$signalModel->train($factorDs);

// Score a universe of 20 equities
$tickers = ['AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','JPM','BAC','GS',
            'XOM','CVX','JNJ','PFE','UNH','WMT','HD','MCD','V','MA'];
$universe = [];
foreach ($tickers as $t) {
    $universe[] = [$randn()*0.3, $randn()*0.25, abs($randn())*0.4+0.05,
                   abs($randn())*0.15+0.05, $randn()*0.3, $randn()*0.2];
}
$univDs  = Dataset::fromArray($universe);
$signals = $signalModel->predict($univDs)->toFlatArray();

printf("\n  %-8s | Q | Signal    %-8s | Q | Signal\n", 'Ticker', 'Ticker');
printf("  %s\n", str_repeat('-', 55));
for ($i = 0; $i < 10; $i++) {
    $q1 = (int)round($signals[$i]) + 1;
    $q2 = (int)round($signals[$i + 10]) + 1;
    $s1 = $q1 >= 4 ? 'LONG ↑' : ($q1 <= 2 ? 'SHORT ↓' : 'NEUTRAL');
    $s2 = $q2 >= 4 ? 'LONG ↑' : ($q2 <= 2 ? 'SHORT ↓' : 'NEUTRAL');
    printf("  %-8s | %d | %-10s %-8s | %d | %s\n",
           $tickers[$i], $q1, $s1, $tickers[$i+10], $q2, $s2);
}

// ═══════════════════════════════════════════════════════════════════
// MODULE 3: VOLATILITY ANOMALY DETECTION
// ═══════════════════════════════════════════════════════════════════
section('Module 3: Volatility Anomaly Alert');

// Normal vol regime features
$volNormal = [];
for ($i = 0; $i < 5000; $i++) {
    $volNormal[] = [$rng(0.10, 0.25), $rng(12, 22), $rng(0.5, 1.5), $rng(0.8, 1.2), $rng(-0.3, 0.3)];
}
$volModel = new IsolationForest(nEstimators: 100, sampleSize: 256, contamination: 0.02);
$volModel->train(Dataset::fromArray($volNormal));

$volScenarios = [
    '2024-11-05 (election day)' => [0.42, 35.0, 3.2, 0.6, 0.8],
    '2024-03-15 (normal day)'   => [0.16, 15.5, 0.9, 1.1, 0.1],
    '2024-08-05 (vol shock)'    => [0.75, 52.0, 5.1, 0.3, 1.5],
];

foreach ($volScenarios as $date => $features) {
    $score = $volModel->predict(Dataset::fromArray([$features]))->toFlatArray()[0] ?? 0.0;
    $alert = $score > 0.5 ? '🚨 VOLATILITY ALERT — reduce risk exposure' : '✅ Normal';
    printf("  %-30s anomaly=%.3f  %s\n", $date, $score, $alert);
}

// ═══════════════════════════════════════════════════════════════════
// FINAL: PORTFOLIO ALLOCATION RECOMMENDATION
// ═══════════════════════════════════════════════════════════════════
section('Portfolio Allocation Recommendation');

$topLongs  = array_keys(array_filter($signals, fn($q) => $q >= 3.5), true);
$topShorts = array_keys(array_filter($signals, fn($q) => $q <= 0.5), true);

$riskMult = match($mappedRegime) { 1 => 0.5, 2 => 0.75, default => 1.0 };

printf("  Market regime : %s  →  Risk multiplier: %.2f×\n", $regimeNames[$mappedRegime], $riskMult);
printf("  Long book     : %s\n", implode(', ', array_map(fn($i) => $tickers[$i], array_slice($topLongs, 0, 5))));
printf("  Short book    : %s\n", implode(', ', array_map(fn($i) => $tickers[$i], array_slice($topShorts, 0, 3))));
printf("  Gross exposure: %.0f%%  Net exposure: %.0f%%\n", 120 * $riskMult, 70 * $riskMult);

echo "\n✓ Terminal ready\n";
