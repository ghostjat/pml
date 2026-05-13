<?php
declare(strict_types=1);
/**
 * REAL-TIME FRAUD SCORING API — Production Pattern
 * ═══════════════════════════════════════════════════════════════════
 * This script shows the FULL production lifecycle:
 *   1. Train a GBDT fraud model
 *   2. Save to disk
 *   3. Simulate an HTTP API endpoint loading the model once (cold)
 *      and serving thousands of requests (warm)
 *   4. Show latency distribution
 *
 * Pattern  : Train offline → Save → Load once → Score in-process.
 *            No Python, no TensorFlow Serving, no external inference
 *            server — PML scores in < 1 ms per transaction.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Pipeline;
use Pml\Estimators\Classifiers\GBDTClassifier;
use Pml\Transformers\StandardScaler;
use Pml\Metrics\Classification\RocAuc;

// ═══════════════════════════════════════════════════════════════════
// PHASE 1: TRAINING (run once, e.g. nightly cron job)
// ═══════════════════════════════════════════════════════════════════
section('Phase 1 — Offline Training');

mt_srand(55);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

function makeTxns(callable $rng, int $n, float $fraudRate): array
{
    $rows = []; $lbls = [];
    for ($i = 0; $i < $n; $i++) {
        $isFraud = (mt_rand() / mt_getrandmax()) < $fraudRate;
        $rows[]  = $isFraud
            ? [$rng(200, 3000), (int)$rng(0, 4), $rng(5,15), 1.0, 1.0, $rng(20,100), (int)$rng(1,3), $rng(0.7,1.0)]
            : [$rng(5, 250),   (int)$rng(7,22),  $rng(0,3),  0.0, 0.0, $rng(50,600), 0.0,            $rng(0, 0.2)];
        $lbls[] = $isFraud ? 1.0 : 0.0;
    }
    return [$rows, $lbls];
}

[$trainRows, $trainLbls] = makeTxns($rng, 10000, 0.04);
$trainDs = Dataset::fromArray($trainRows, $trainLbls);

$t0 = microtime(true);
$pipeline = new Pipeline(
    [new StandardScaler()],
    new GBDTClassifier(nEstimators: 200, maxDepth: 4, lr: 0.05)
);
$pipeline->train($trainDs);
metric('Training time', elapsed($t0));

// Validate
[$testRows, $testLbls] = makeTxns($rng, 2000, 0.04);
$testDs  = Dataset::fromArray($testRows, $testLbls);
$pred    = $pipeline->predict($testDs);
metric('Validation AUC', (new RocAuc())->score($pred, $testDs->labels()));

// Save model artifact
$modelDir = sys_get_temp_dir() . '/fraud_api_model';
$pipeline->save($modelDir);
metric('Model saved', $modelDir);

// ═══════════════════════════════════════════════════════════════════
// PHASE 2: API SERVER COLD-START (runs once when process starts)
// ═══════════════════════════════════════════════════════════════════
section('Phase 2 — API Cold-Start (model load)');

$t0     = microtime(true);
$scorer = Pipeline::load($modelDir);
metric('Model load (cold-start)', elapsed($t0));

// ═══════════════════════════════════════════════════════════════════
// PHASE 3: SERVING SIMULATION (thousands of requests, model warm)
// ═══════════════════════════════════════════════════════════════════
section('Phase 3 — Serving 5000 Transactions');

[$liveRows] = makeTxns($rng, 5000, 0.04);
$latencies  = [];

foreach ($liveRows as $txn) {
    $t0   = microtime(true);
    $ds   = Dataset::fromArray([$txn]);
    $score = $scorer->predict($ds)->toFlatArray()[0] ?? 0.0;
    $latencies[] = (microtime(true) - $t0) * 1000;  // ms
}

sort($latencies);
$p50  = $latencies[(int)(count($latencies) * 0.50)];
$p95  = $latencies[(int)(count($latencies) * 0.95)];
$p99  = $latencies[(int)(count($latencies) * 0.99)];
$mean = array_sum($latencies) / count($latencies);
$tps  = 1000 / $mean;

metric('Transactions served',   count($latencies));
metric('Mean latency',          round($mean, 4), ' ms');
metric('p50  latency',          round($p50,  4), ' ms');
metric('p95  latency',          round($p95,  4), ' ms');
metric('p99  latency',          round($p99,  4), ' ms');
metric('Throughput (1 process)',round($tps, 0),  ' TPS');
metric('Throughput (16 FPM)',   round($tps * 16, 0), ' TPS (16 PHP-FPM workers)');

// ── Sample decisions ──────────────────────────────────────────────────────────
section('Sample API Responses');

$samples = [
    ['txn_id' => 'TXN-001', 'features' => [2500.0, 2, 12.0, 1, 1, 80.0, 2, 0.9]],
    ['txn_id' => 'TXN-002', 'features' => [45.0, 14, 1.0, 0, 0, 220.0, 0, 0.1]],
    ['txn_id' => 'TXN-003', 'features' => [1200.0, 1, 8.0, 1, 0, 60.0, 1, 0.7]],
];

foreach ($samples as $req) {
    $ds    = Dataset::fromArray([$req['features']]);
    $score = $scorer->predict($ds)->toFlatArray()[0] ?? 0.0;
    printf("  POST /score  txn_id=%-10s  score=%.4f  action=%s\n",
           $req['txn_id'], $score, $score > 0.5 ? 'DECLINE' : 'APPROVE');
}

echo <<<TXT

  ┌─────────────────────────────────────────────────────────┐
  │  DEPLOYMENT RECIPE                                      │
  │                                                         │
  │  # Nginx → PHP-FPM (16 workers) → PML model in-process │
  │                                                         │
  │  1. Run training once (cron):                           │
  │     php train.php && php deploy.php /fraud_api_model    │
  │                                                         │
  │  2. PHP-FPM worker (opcache) loads model on cold-start  │
  │     → reuses for every subsequent request (warm)        │
  │                                                         │
  │  3. POST /score { features: [...] }                     │
  │     → < 1 ms inference → { score: 0.92, action: DECLINE}│
  └─────────────────────────────────────────────────────────┘

TXT;

echo "✓ Done\n";
