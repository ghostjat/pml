<?php
declare(strict_types=1);
/**
 * CUSTOMER CHURN PREDICTION — Telecom
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Predict which subscribers will cancel within 30 days.
 * Model    : RandomForestClassifier — robust to noisy features,
 *            naturally handles mixed numeric/categorical data.
 * Business : Acquiring a customer costs 5–7× retaining one.
 *            Targeting the top 20 % at-risk users with a $20 voucher
 *            can reduce monthly churn by 40 % and save millions.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Tensor;
use Pml\Estimators\Classifiers\RandomForestClassifier;
use Pml\Metrics\Classification\RocAuc;
use Pml\Metrics\Classification\F1Score;
use Pml\Metrics\Classification\Accuracy;

section('Customer Churn Prediction — Random Forest');

// ── 1. Generate subscriber dataset ───────────────────────────────────────────
// Features: tenure_months, monthly_charge, total_charges, num_products,
//           support_tickets_6m, avg_session_minutes, payment_delay_days,
//           has_contract (0/1), has_fiber (0/1), num_complaints

mt_srand(42);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

$rows = [];
$lbls = [];

for ($i = 0; $i < 6000; $i++) {
    $tenure    = $rng(1, 72);
    $charge    = $rng(20, 120);
    $contract  = (mt_rand(0, 1));
    $tickets   = (int)$rng(0, 10);
    $complaints = (int)$rng(0, 5);

    // Churn probability based on domain logic
    $churnProb = 0.08
        + (72 - $tenure) / 72 * 0.25      // new customers churn more
        + $tickets / 10 * 0.30             // complaints drive churn
        + $complaints / 5 * 0.20
        - $contract * 0.20;                // contracts retain customers

    $churnProb = max(0.0, min(1.0, $churnProb));

    $rows[] = [
        $tenure,
        $charge,
        $tenure * $charge,
        (int)$rng(1, 5),
        $tickets,
        $rng(5, 180),
        $rng(0, 30),
        (float)$contract,
        (float)(mt_rand(0, 1)),
        $complaints,
    ];
    $lbls[] = ((mt_rand() / mt_getrandmax()) < $churnProb) ? 1.0 : 0.0;
}

$dataset = Dataset::fromArray($rows, $lbls);
[$train, $test] = $dataset->randomize()->split(0.8);

$churnCount = (int)array_sum($lbls);
metric('Total subscribers',  count($lbls));
metric('Churned',            $churnCount, ' (' . round($churnCount / count($lbls) * 100, 1) . '%)');

// ── 2. Train ──────────────────────────────────────────────────────────────────
section('Training');
$t0 = microtime(true);

$model = new RandomForestClassifier(nEstimators: 200, maxDepth: 12);
$model->train($train);

metric('Training time', elapsed($t0));

// ── 3. Evaluate ───────────────────────────────────────────────────────────────
section('Evaluation');
$pred   = $model->predict($test);
$labels = $test->labels();

metric('Accuracy', (new Accuracy())->score($pred, $labels));
metric('ROC-AUC',  (new RocAuc())->score($pred, $labels));
metric('F1-Score', (new F1Score())->score($pred, $labels));

// ── 4. Priority churn list (top at-risk) ─────────────────────────────────────
section('At-Risk Customer Prioritisation');

// Score the whole test set and rank by churn probability
$testArr = $test->samples()->toFlatArray();
$cols    = 10;
$nTest   = $test->numRows();
$scores  = $pred->toFlatArray();

// Take top 10 % as "intervention targets"
arsort($scores);
$top10pct = array_slice(array_keys($scores), 0, (int)($nTest * 0.10), preserve_keys: true);

printf("  Top 10%% highest-risk users : %d subscribers\n", count($top10pct));
printf("  Sending retention vouchers to these users...\n");
printf("  Estimated monthly churn saved: %.0f subscribers\n", count($top10pct) * 0.40);

/*
 * PRODUCTION NOTES
 * ────────────────
 * • Run nightly batch on CRM export → update churn_score field.
 * • Trigger Salesforce / HubSpot workflow for score > 0.65.
 * • Retrain monthly; monitor AUC drift — alert if drops > 3 pts.
 * • Add NPS score, last-login recency, and upsell history as features.
 */
echo "\n✓ Done\n";
