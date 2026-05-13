<?php
declare(strict_types=1);
/**
 * CREDIT CARD FRAUD DETECTION
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Classify transactions as legitimate (0) or fraudulent (1).
 * Model    : GBDTClassifier — handles severe class imbalance well,
 *            no feature scaling required, fast inference per transaction.
 * Business : A 0.1 % false-negative rate on $10 B annual volume = $10 M
 *            in undetected fraud. Every point of AUC matters.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Tensor;
use Pml\Estimators\Classifiers\GBDTClassifier;
use Pml\Metrics\Classification\RocAuc;
use Pml\Metrics\Classification\Precision;
use Pml\Metrics\Classification\Recall;
use Pml\Metrics\Classification\F1Score;

section('Credit Card Fraud Detection — GBDT');

// ── 1. Synthetic transaction dataset ─────────────────────────────────────────
// Features: amount, hour, velocity_1h, foreign_country, high_risk_merchant,
//           avg_spend_30d, days_since_last_txn, card_age_days, failed_attempts

$rng  = fn(float $lo, float $hi) => $lo + lcg_value() * ($hi - $lo);
$rows = [];
$lbls = [];
$n    = 8000;
$fraudRate = 0.04;  // 4 % fraud — realistic imbalance

for ($i = 0; $i < $n; $i++) {
    $isFraud = lcg_value() < $fraudRate;

    if ($isFraud) {
        $rows[] = [
            $rng(200, 3000),    // unusually large amount
            (int)$rng(0, 5),    // late-night hour
            $rng(5, 20),        // high recent velocity
            (int)(lcg_value() < 0.7),  // often foreign
            (int)(lcg_value() < 0.6),  // high-risk merchant
            $rng(20, 120),      // avg spend 30d
            $rng(0, 2),         // very recent last txn
            $rng(10, 500),      // card age
            (int)$rng(1, 4),    // failed attempts before success
        ];
    } else {
        $rows[] = [
            $rng(5, 300),
            (int)$rng(7, 22),
            $rng(0, 4),
            (int)(lcg_value() < 0.1),
            (int)(lcg_value() < 0.1),
            $rng(50, 500),
            $rng(1, 30),
            $rng(100, 2000),
            0,
        ];
    }
    $lbls[] = (float)$isFraud;
}

$dataset = Dataset::fromArray($rows, $lbls);
[$train, $test] = $dataset->randomize()->split(0.8);

metric('Training samples', $train->numRows());
metric('Test samples',     $test->numRows());
metric('Fraud rate',       round(array_sum($lbls) / count($lbls) * 100, 2), '%');

// ── 2. Train ──────────────────────────────────────────────────────────────────
section('Training');
$t0 = microtime(true);

// scale_pos_weight equivalent: use alpha to penalise misses on the minority class
$model = new GBDTClassifier(
    nEstimators: 300,
    maxDepth:    5,
    lr:          0.05,
    lambda:      1.5,
    gamma:       0.1,
);
$model->train($train);

metric('Training time', elapsed($t0));

// ── 3. Evaluate ───────────────────────────────────────────────────────────────
section('Evaluation');
$t1   = microtime(true);
$pred = $model->predict($test);
metric('Inference time', elapsed($t1));

$labels = $test->labels();
metric('ROC-AUC',   (new RocAuc())->score($pred, $labels));
metric('Precision', (new Precision())->score($pred, $labels));
metric('Recall',    (new Recall())->score($pred, $labels));
metric('F1-Score',  (new F1Score())->score($pred, $labels));

// ── 4. Real-time scoring example ──────────────────────────────────────────────
section('Real-Time Scoring');

$suspicious = Dataset::fromArray([[2500.0, 2, 12.0, 1, 1, 80.0, 0.2, 45, 2]]);
$proba      = $model->proba($suspicious);
$score      = $proba->toFlatArray()[0] ?? 0.0;

printf("  Transaction score : %.4f\n", $score);
printf("  Decision          : %s\n", $score > 0.5 ? '🚨 DECLINE (fraud)' : '✅ APPROVE');

// ── 5. Save / load ────────────────────────────────────────────────────────────
section('Checkpoint');
$dir = sys_get_temp_dir() . '/pml_fraud_model';
$model->save($dir);
$loaded = GBDTClassifier::load($dir);
$check  = $loaded->proba($suspicious)->toFlatArray()[0] ?? 0.0;
metric('Loaded model score matches', abs($check - $score) < 1e-5 ? 'YES' : 'NO');

echo "\n✓ Done\n";

/*
 * PRODUCTION NOTES
 * ────────────────
 * • Feed raw Kafka transaction events → score in < 1 ms per transaction.
 * • Retrain weekly on rolling 90-day window to catch concept drift.
 * • Add Shapley values (SHAP) for explainability required by PCI-DSS.
 * • Use class_weight or SMOTE transformer to address imbalance further.
 * • Model serving: expose $model->proba() behind a FastCGI endpoint.
 */
