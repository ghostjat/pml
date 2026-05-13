<?php
declare(strict_types=1);
/**
 * STREAMING FRAUD DETECTION — Online Learning
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Detect payment fraud on a real-time transaction stream
 *            where the model must update continuously as new fraud
 *            patterns emerge (concept drift).
 * Model    : MLPClassifier with partial() updates — the model learns
 *            from each mini-batch without forgetting past knowledge.
 * Business : Fraud patterns change weekly. A static model trained
 *            quarterly loses 20–40 % of its detection power within
 *            30 days. Online learning keeps the model current.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Estimators\Classifiers\MLPClassifier;
use Pml\Metrics\Classification\RocAuc;
use Pml\Metrics\Classification\F1Score;

section('Streaming Fraud Detection — Online MLP');

// ── 1. Transaction stream generator ──────────────────────────────────────────
// Features: amount, hour, velocity_1h, foreign (0/1), new_device (0/1),
//           avg_spend_30d, failed_attempts, merchant_risk_score

mt_srand(17);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

function generateBatch(callable $rng, int $n, float $fraudRate, bool $drifted = false): array
{
    $rows = []; $lbls = [];
    for ($i = 0; $i < $n; $i++) {
        $isFraud = (mt_rand() / mt_getrandmax()) < $fraudRate;
        if ($drifted && $isFraud) {
            // New fraud pattern: small amounts, domestic, new device (card-testing)
            $rows[] = [$rng(1, 25), (int)$rng(10, 22), $rng(8, 20), 0.0, 1.0,
                       $rng(100, 500), (int)$rng(0, 1), $rng(0.5, 0.8)];
        } elseif ($isFraud) {
            // Classic fraud: large amount, foreign, late night
            $rows[] = [$rng(500, 3000), (int)$rng(0, 4), $rng(5, 15), 1.0, 0.0,
                       $rng(30, 120), (int)$rng(1, 4), $rng(0.7, 1.0)];
        } else {
            $rows[] = [$rng(5, 250), (int)$rng(7, 22), $rng(0, 3), 0.0, 0.0,
                       $rng(50, 800), 0.0, $rng(0, 0.3)];
        }
        $lbls[] = $isFraud ? 1.0 : 0.0;
    }
    return [$rows, $lbls];
}

// ── 2. Initial training (warm-start) ─────────────────────────────────────────
section('Phase 1: Initial Training');

[$rows0, $lbls0] = generateBatch($rng, 2000, 0.05);
$initDs = Dataset::fromArray($rows0, $lbls0);

$model = new MLPClassifier(hidden: [64, 32], epochs: 20, batchSize: 64, learningRate: 0.001);
$model->train($initDs);

// Evaluate on holdout
[$rows0t, $lbls0t] = generateBatch($rng, 500, 0.05);
$holdout0 = Dataset::fromArray($rows0t, $lbls0t);
$pred0    = $model->predict($holdout0);
$labels0  = $holdout0->labels();
metric('Initial AUC', (new RocAuc())->score($pred0, $labels0));

// ── 3. Simulate concept drift (new fraud pattern) ─────────────────────────────
section('Phase 2: Concept Drift (new fraud pattern)');
printf("  Fraudsters switch from large-foreign to card-testing (small domestic)\n\n");

// Evaluate BEFORE updating — AUC should drop
[$rowsDrift, $lblsDrift] = generateBatch($rng, 500, 0.05, drifted: true);
$driftDs = Dataset::fromArray($rowsDrift, $lblsDrift);
$predDrift = $model->predict($driftDs);
metric('AUC before update (drift detected!)', (new RocAuc())->score($predDrift, $driftDs->labels()));

// ── 4. Online update — mini-batch streaming ───────────────────────────────────
section('Phase 3: Online Update (stream learning)');

$aucs = [];
for ($week = 1; $week <= 4; $week++) {
    [$rowsNew, $lblsNew] = generateBatch($rng, 300, 0.05, drifted: true);
    $newBatch = Dataset::fromArray($rowsNew, $lblsNew);

    // partial() = one epoch on the new data, preserving past weights
    $model->partial($newBatch);

    // Evaluate
    [$rowsEval, $lblsEval] = generateBatch($rng, 200, 0.05, drifted: true);
    $evalDs = Dataset::fromArray($rowsEval, $lblsEval);
    $auc    = (new RocAuc())->score($model->predict($evalDs), $evalDs->labels());
    $aucs[] = $auc;

    printf("  Week %d update | AUC=%.4f | Transactions processed: %d\n",
           $week, $auc, $week * 300);
}

metric('AUC recovery after adaptation', end($aucs));

// ── 5. Throughput benchmark ───────────────────────────────────────────────────
section('Throughput (Inference Speed)');

[$rowsPerf, $lblsPerf] = generateBatch($rng, 10000, 0.0);
$perfDs = Dataset::fromArray($rowsPerf, $lblsPerf);

$t0 = microtime(true);
$model->predict($perfDs);
$elapsed = (microtime(true) - $t0) * 1000;

printf("  10,000 transactions scored in %.1f ms\n", $elapsed);
printf("  Throughput: %.0f transactions/second\n", 10000 / ($elapsed / 1000));
printf("  Latency per transaction: %.3f ms\n", $elapsed / 10000);

echo "\n✓ Done\n";
