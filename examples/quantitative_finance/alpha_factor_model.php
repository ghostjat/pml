<?php
declare(strict_types=1);
/**
 * ALPHA FACTOR MODEL — Quantitative Equity Research
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Predict next-month stock return quintile (1=bottom → 5=top)
 *            from factor exposures: momentum, value, quality, size.
 * Model    : GBDTClassifier — captures nonlinear factor interactions
 *            missed by linear factor models (Fama-French style).
 * Business : A long/short equity fund separating top/bottom quintile
 *            with 60 %+ accuracy on a $100 M book generates
 *            significant alpha net of transaction costs.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Estimators\Classifiers\GBDTClassifier;
use Pml\Metrics\Classification\Accuracy;
use Pml\Metrics\Classification\RocAuc;

section('Alpha Factor Model — Quant Equity');

// ── 1. Cross-sectional factor dataset ─────────────────────────────────────────
// Factors (normalised within each monthly cross-section):
//   mom_12m_1m   : 12-month momentum minus last month (skip-month)
//   value_pb     : Price-to-Book (inverted: low PB = high value score)
//   quality_roe  : Return on Equity
//   quality_debt : Debt-to-Equity (inverted)
//   size_logmcap : Log market cap (inverted: small = high score)
//   vol_1m       : 1-month realised volatility
//   rev_1m       : 1-month reversal
//   earns_surp   : Earnings surprise (actual vs consensus)
//
// Label: return quintile in next month (0..4 mapped to 1..5)

mt_srand(2024);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);
$randn = function() {
    // Box-Muller
    static $next = null;
    if ($next !== null) { $v = $next; $next = null; return $v; }
    $u = lcg_value(); $v = lcg_value();
    $mag = sqrt(-2 * log($u + 1e-10));
    $next = $mag * sin(2 * M_PI * $v);
    return $mag * cos(2 * M_PI * $v);
};

$rows = []; $lbls = [];
$nStocks = 500; $nMonths = 36;

for ($m = 0; $m < $nMonths; $m++) {
    // Simulate cross-section for this month
    $stockFeatures = [];
    $returns = [];

    for ($s = 0; $s < $nStocks; $s++) {
        $mom    = $randn() * 0.4;
        $value  = $randn() * 0.3;
        $roe    = $randn() * 0.5 + 0.1;
        $debt   = $randn() * 0.3;
        $size   = $randn() * 0.5;
        $vol    = abs($randn()) * 0.2 + 0.05;
        $rev    = $randn() * 0.15;
        $earns  = $randn() * 0.2;

        // Return has nonlinear factor dependencies
        $ret = 0.003                       // market return
             + $mom   * 0.06              // momentum premium
             + $value * 0.04              // value premium
             + $roe   * 0.03              // quality
             - abs($debt) * 0.02          // debt penalty
             - $size  * 0.02              // size premium (small-cap)
             - $vol   * 0.05              // low-vol anomaly
             + $mom * $value * 0.01       // nonlinear interaction
             + $earns * 0.04              // earnings momentum
             + $randn() * 0.08;           // idiosyncratic noise

        $stockFeatures[] = [$mom, $value, $roe, $debt, $size, $vol, $rev, $earns];
        $returns[] = $ret;
    }

    // Assign quintiles (0..4) based on that month's cross-sectional returns
    $sorted = $returns;
    sort($sorted);
    $quintileEdges = [
        $sorted[(int)($nStocks * 0.20)],
        $sorted[(int)($nStocks * 0.40)],
        $sorted[(int)($nStocks * 0.60)],
        $sorted[(int)($nStocks * 0.80)],
    ];

    foreach ($stockFeatures as $k => $feats) {
        $r = $returns[$k];
        $q = match(true) {
            $r < $quintileEdges[0] => 0.0,
            $r < $quintileEdges[1] => 1.0,
            $r < $quintileEdges[2] => 2.0,
            $r < $quintileEdges[3] => 3.0,
            default                => 4.0,
        };
        $rows[] = $feats;
        $lbls[] = $q;
    }
}

// Walk-forward: train on first 80 % of months
$cutoff  = (int)(count($rows) * 0.8);
$trainDs = Dataset::fromArray(array_slice($rows, 0, $cutoff), array_slice($lbls, 0, $cutoff));
$testDs  = Dataset::fromArray(array_slice($rows, $cutoff),    array_slice($lbls, $cutoff));

metric('Training obs', $trainDs->numRows());
metric('Test obs',     $testDs->numRows());

// ── 2. Train ──────────────────────────────────────────────────────────────────
section('Training');
$t0 = microtime(true);

$model = new GBDTClassifier(nEstimators: 300, maxDepth: 4, lr: 0.05, lambda: 2.0);
$model->train($trainDs);

metric('Training time', elapsed($t0));

// ── 3. Evaluate ───────────────────────────────────────────────────────────────
section('Evaluation');
$pred   = $model->predict($testDs);
$labels = $testDs->labels();

metric('Accuracy (5-class)',     (new Accuracy())->score($pred, $labels));
metric('Random baseline',        '0.2000 (1/5 quintiles)');

// Rank IC: check if top predicted quintile (Q5) has higher avg return than bottom (Q1)
$predsArr  = $pred->toFlatArray();
$labelsArr = $labels->toFlatArray();

$q1hits = $q5hits = $q1count = $q5count = 0;
foreach ($predsArr as $i => $p) {
    if ((int)round($p) === 4) { $q5hits += ($labelsArr[$i] >= 3) ? 1 : 0; $q5count++; }
    if ((int)round($p) === 0) { $q1hits += ($labelsArr[$i] <= 1) ? 1 : 0; $q1count++; }
}

metric('Q5 signal precision (high → high)', $q5count ? round($q5hits / $q5count, 4) : 0);
metric('Q1 signal precision (low → low)',   $q1count ? round($q1hits / $q1count, 4) : 0);

// ── 4. Live factor score ───────────────────────────────────────────────────────
section('Factor Scoring Sample Stocks');

$universe = [
    'AAPL' => [ 0.35, -0.2, 0.8, -0.1,  0.6, 0.12, -0.1, 0.3],  // momentum + quality
    'GME'  => [-0.1,  0.4, -0.3, 0.5,  -0.8, 0.45,  0.2, -0.1],  // volatile, value-trap
    'BRK'  => [ 0.1,  0.6,  0.6, -0.3,  1.2, 0.05,  0.0, 0.1],   // value + quality
];

$stockDs = Dataset::fromArray(array_values($universe));
$scores  = $model->predict($stockDs)->toFlatArray();

printf("  %-6s | Quintile | Signal\n", 'Ticker');
printf("  %s\n", str_repeat('-', 35));
foreach (array_keys($universe) as $k => $ticker) {
    $q = (int)round($scores[$k]) + 1;
    $signal = match(true) {
        $q >= 4 => 'LONG  ↑',
        $q <= 2 => 'SHORT ↓',
        default  => 'NEUTRAL',
    };
    printf("  %-6s | Q%-7d | %s\n", $ticker, $q, $signal);
}

echo "\n✓ Done\n";
