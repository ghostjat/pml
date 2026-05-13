<?php
declare(strict_types=1);
/**
 * DIABETES RISK PREDICTION — Clinical Decision Support
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Predict Type-2 diabetes onset within 5 years from
 *            routine lab values and patient demographics.
 * Model    : GBDTClassifier — high recall (sensitivity) is critical:
 *            missing a diabetic is worse than a false alarm.
 * Business : Early intervention (diet, exercise, metformin) reduces
 *            progression to full diabetes by 58 % (DPP trial).
 *            Flagging at-risk patients at annual check-ups costs $0
 *            in extra tests and saves ~$10,000 per prevented case.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Pipeline;
use Pml\Estimators\Classifiers\GBDTClassifier;
use Pml\Transformers\StandardScaler;
use Pml\Metrics\Classification\RocAuc;
use Pml\Metrics\Classification\Recall;
use Pml\Metrics\Classification\Precision;
use Pml\Metrics\Classification\F1Score;

section('Diabetes Risk Prediction — GBDT');

// ── 1. Clinical dataset ───────────────────────────────────────────────────────
// Features: glucose_fasting, bmi, age, blood_pressure_systolic,
//           hba1c, insulin_uU_ml, skin_thickness_mm, family_history (0/1),
//           pregnancies (females), cholesterol_total

mt_srand(11);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

$rows = []; $lbls = [];

for ($i = 0; $i < 5000; $i++) {
    $glucose   = $rng(70, 200);
    $bmi       = $rng(18, 50);
    $age       = $rng(21, 80);
    $bp        = $rng(60, 140);
    $hba1c     = $rng(4.5, 10.0);
    $insulin   = $rng(10, 300);
    $skin      = $rng(10, 60);
    $family    = mt_rand(0, 1);
    $preg      = (int)$rng(0, 10);
    $chol      = $rng(140, 280);

    // Epidemiologically-grounded risk function
    $risk = -5.0
        + ($glucose - 100) * 0.06
        + ($bmi - 25)      * 0.08
        + ($age - 35)      * 0.04
        + ($hba1c - 5.5)   * 0.80
        + $family          * 0.60
        + ($chol - 200)    * 0.01
        + $preg            * 0.08;

    $prob = 1.0 / (1.0 + exp(-$risk));

    $rows[] = [$glucose, $bmi, $age, $bp, $hba1c, $insulin, $skin,
               (float)$family, (float)$preg, $chol];
    $lbls[] = ((mt_rand() / mt_getrandmax()) < $prob) ? 1.0 : 0.0;
}

$ds = Dataset::fromArray($rows, $lbls);
[$train, $test] = $ds->randomize()->split(0.8);

$pos = (int)array_sum($lbls);
metric('Total patients', count($lbls));
metric('Diabetic (positive)', $pos . ' (' . round($pos / count($lbls) * 100, 1) . '%)');

// ── 2. Pipeline ───────────────────────────────────────────────────────────────
section('Training');
$t0 = microtime(true);

$pipeline = new Pipeline(
    [new StandardScaler()],
    new GBDTClassifier(nEstimators: 250, maxDepth: 5, lr: 0.05, lambda: 1.5)
);
$pipeline->train($train);

metric('Training time', elapsed($t0));

// ── 3. Evaluate — prioritise Recall (catch every diabetic) ────────────────────
section('Clinical Evaluation');
$pred   = $pipeline->predict($test);
$labels = $test->labels();

metric('ROC-AUC',   (new RocAuc())->score($pred, $labels));
metric('Recall (sensitivity)',   (new Recall())->score($pred, $labels));
metric('Precision (PPV)',        (new Precision())->score($pred, $labels));
metric('F1-Score',               (new F1Score())->score($pred, $labels));

// ── 4. Simulate clinical workflow ─────────────────────────────────────────────
section('Automated Risk Stratification');

$patients = [
    'Patient A (low risk)'  => [85,  22.0, 28, 75,  5.1, 18,  20, 0, 0, 160],
    'Patient B (pre-diab.)' => [118, 29.5, 45, 90,  5.9, 85,  32, 1, 0, 210],
    'Patient C (high risk)' => [160, 38.0, 58, 110, 7.2, 190, 45, 1, 3, 255],
    'Patient D (border)'    => [102, 27.0, 36, 82,  5.5, 42,  26, 0, 1, 195],
];

printf("\n  %-28s | %-8s | %s\n", 'Patient', 'Risk', 'Recommendation');
printf("  %s\n", str_repeat('-', 75));

foreach ($patients as $name => $feats) {
    $sampleDs = Dataset::fromArray([$feats]);
    $score    = $pipeline->predict($sampleDs)->toFlatArray()[0] ?? 0.0;

    [$risk, $action] = match(true) {
        $score > 0.70 => ['HIGH',   '🔴 Refer to endocrinologist + HbA1c retest in 3m'],
        $score > 0.40 => ['MEDIUM', '🟡 Lifestyle intervention + retest in 6 months'],
        default       => ['LOW',    '🟢 Annual screening — standard care'],
    };
    printf("  %-28s | %-8s | %s\n", $name, $risk, $action);
}

echo "\n✓ Done\n";

/*
 * REGULATORY NOTES
 * ────────────────
 * • FDA/CE clinical decision support rules apply; model must be validated
 *   on prospective clinical data before deployment.
 * • Always present risk score to a licensed clinician, not directly to patient.
 * • Log every prediction with patient ID and model version for audit trail.
 */
