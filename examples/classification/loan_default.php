<?php
declare(strict_types=1);
/**
 * LOAN DEFAULT PREDICTION — Consumer Lending
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Predict whether a loan applicant will default within
 *            12 months of origination.
 * Model    : GBDTClassifier (tree models are scale-invariant; no scaler needed).
 * Business : A 1 % improvement in default detection on a $500 M
 *            portfolio saves $5 M in charge-offs annually.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Estimators\Classifiers\GBDTClassifier;
use Pml\Metrics\Classification\RocAuc;
use Pml\Metrics\Classification\Precision;
use Pml\Metrics\Classification\Recall;

section('Loan Default Prediction — GBDT');

// ── 1. Applicant dataset ─────────────────────────────────────────────────────
// Features: credit_score, annual_income, loan_amount, loan_term_months,
//           dti_ratio, num_open_accounts, delinquencies_2yr,
//           employment_years, home_ownership (0=rent,1=own), purpose_encoded

mt_srand(99);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

$rows = [];
$lbls = [];

for ($i = 0; $i < 7000; $i++) {
    $score  = (int)$rng(300, 850);
    $income = $rng(20000, 200000);
    $loan   = $rng(1000, 50000);
    $dti    = $rng(0.05, 0.60);
    $delinq = (int)$rng(0, 5);

    $defaultProb = 0.06
        + (700 - $score) / 700 * 0.35
        + $dti * 0.25
        + $delinq * 0.06
        - min($income, 100000) / 100000 * 0.15;

    $defaultProb = max(0.01, min(0.95, $defaultProb));

    $rows[] = [
        (float)$score,
        $income,
        $loan,
        (float)(int)$rng(12, 60),
        $dti,
        (float)(int)$rng(2, 20),
        (float)$delinq,
        $rng(0, 25),
        (float)(mt_rand(0, 1)),
        (float)(int)$rng(0, 6),
    ];
    $lbls[] = ((mt_rand() / mt_getrandmax()) < $defaultProb) ? 1.0 : 0.0;
}

$dataset = Dataset::fromArray($rows, $lbls);
[$train, $test] = $dataset->randomize()->split(0.8);

$defaults = (int)array_sum($lbls);
metric('Applications', count($lbls));
metric('Default rate', round($defaults / count($lbls) * 100, 1), '%');

// ── 2. Train GBDT ─────────────────────────────────────────────────────────────
section('Training');
$t0 = microtime(true);

$model = new GBDTClassifier(nEstimators: 200, maxDepth: 4, lr: 0.08, lambda: 2.0);
$model->train($train);

metric('Training time', elapsed($t0));

// ── 3. Evaluate ───────────────────────────────────────────────────────────────
section('Evaluation');
$pred   = $model->predict($test);
$labels = $test->labels();

metric('ROC-AUC',   (new RocAuc())->score($pred, $labels));
metric('Precision', (new Precision())->score($pred, $labels));
metric('Recall',    (new Recall())->score($pred, $labels));

// ── 4. Underwriting decision engine ───────────────────────────────────────────
section('Underwriting Decisions');

$applicants = Dataset::fromArray([
    [780, 95000, 15000, 36, 0.12, 8, 0, 10, 1, 1],  // excellent
    [580, 28000, 12000, 60, 0.52, 15, 3, 1,  0, 4],  // high risk
    [660, 55000, 8000,  24, 0.28, 6, 1, 5,  0, 2],  // borderline
]);

$proba  = $model->proba($applicants);
$flat   = $proba->toFlatArray();
$names  = ['Alice (prime)', 'Bob (subprime)', 'Carol (near-prime)'];

for ($j = 0; $j < 3; $j++) {
    $pDefault = $flat[$j * 2 + 1];  // P(class=1) = column 1 from [N,2] proba
    $decision = match(true) {
        $pDefault < 0.15 => 'APPROVE  — standard rate',
        $pDefault < 0.35 => 'APPROVE  — risk-adjusted rate +2%',
        $pDefault < 0.55 => 'REVIEW   — manual underwriting',
        default          => 'DECLINE',
    };
    printf("  %-22s score=%.3f  →  %s\n", $names[$j], $pDefault, $decision);
}

// ── 5. Save for production API ────────────────────────────────────────────────
$dir = sys_get_temp_dir() . '/pml_loan_model';
$model->save($dir);
metric('Model saved to', $dir);

echo "\n✓ Done\n";
