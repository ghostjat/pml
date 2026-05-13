<?php
declare(strict_types=1);
/**
 * FEATURE ENGINEERING PIPELINE — Production Tabular ML
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Employee attrition prediction with a full preprocessing
 *            pipeline: imputation → encoding → scaling → selection.
 * Pattern  : Pipeline(transformers[], estimator) — the idiomatic PML
 *            way to chain preprocessing and training reproducibly.
 * Business : Replacing an employee costs 50–200 % of their annual
 *            salary. Proactively identifying flight-risk employees
 *            allows HR to intervene before it's too late.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Pipeline;
use Pml\Estimators\Classifiers\GBDTClassifier;
use Pml\Transformers\Imputer;
use Pml\Transformers\StandardScaler;
use Pml\Transformers\VarianceThreshold;
use Pml\Metrics\Classification\RocAuc;
use Pml\Metrics\Classification\F1Score;
use Pml\Metrics\Classification\Accuracy;

section('Employee Attrition — Full Preprocessing Pipeline');

// ── 1. HR dataset (with realistic messiness) ──────────────────────────────────
// Features: age, years_at_company, salary_k, satisfaction_score(1-5),
//           performance_rating(1-4), work_life_balance(1-4),
//           num_promotions, overtime_hours_week, distance_from_home_km,
//           num_companies_worked, training_hours_last_year

mt_srand(44);
$rng  = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);
$nan  = NAN;

$rows = []; $lbls = [];

for ($i = 0; $i < 4000; $i++) {
    $age         = $rng(22, 60);
    $years       = min($age - 20, $rng(0, 25));
    $salary      = $rng(30, 180);
    $satisfaction= $rng(1, 5);
    $performance = (int)$rng(1, 4);
    $wlb         = $rng(1, 4);
    $promotions  = (int)$rng(0, 4);
    $overtime    = $rng(0, 30);
    $distance    = $rng(1, 80);
    $companies   = (int)$rng(1, 8);
    $training    = $rng(0, 80);

    // Inject 5 % NaN values (realistic data quality issues)
    if (mt_rand(0, 19) === 0) $satisfaction = $nan;
    if (mt_rand(0, 19) === 0) $training     = $nan;

    $attritionProb = 0.10
        + ($satisfaction < 2.5 ? 0.25 : 0)
        + ($overtime > 15      ? 0.15 : 0)
        + ($years < 2          ? 0.20 : 0)
        + ($wlb < 2.0          ? 0.15 : 0)
        - ($promotions > 0     ? 0.10 : 0)
        - ($salary > 100       ? 0.10 : 0)
        + ($companies > 4      ? 0.10 : 0);

    $attritionProb = max(0.02, min(0.90, $attritionProb));

    $rows[] = [$age, $years, $salary, $satisfaction, (float)$performance,
               $wlb, (float)$promotions, $overtime, $distance,
               (float)$companies, $training];
    $lbls[] = ((mt_rand() / mt_getrandmax()) < $attritionProb) ? 1.0 : 0.0;
}

$ds = Dataset::fromArray($rows, $lbls);
[$train, $test] = $ds->randomize()->split(0.8);

$attrCount = (int)array_sum($lbls);
metric('Employees',         count($lbls));
metric('Attrition rate',    round($attrCount / count($lbls) * 100, 1), '%');

// ── 2. Full preprocessing pipeline ───────────────────────────────────────────
section('Pipeline: Imputer → Scaler → VarianceThreshold → GBDT');

// Each transformer mutates the dataset in sequence during train().
// The same fitted transformers are applied automatically during predict().

$pipeline = new Pipeline(
    [
        new Imputer(),              // fill NaN with column median
        new StandardScaler(),       // zero mean, unit variance
        new VarianceThreshold(minVariance: 0.001),  // drop near-constant features
    ],
    new GBDTClassifier(nEstimators: 200, maxDepth: 4, lr: 0.08, lambda: 2.0)
);

$t0 = microtime(true);
$pipeline->train($train);
metric('Total pipeline fit time', elapsed($t0));

// ── 3. Evaluate ───────────────────────────────────────────────────────────────
section('Evaluation');
$pred   = $pipeline->predict($test);
$labels = $test->labels();

metric('ROC-AUC',  (new RocAuc())->score($pred, $labels));
metric('F1-Score', (new F1Score())->score($pred, $labels));
metric('Accuracy', (new Accuracy())->score($pred, $labels));

// ── 4. HR flight-risk report ──────────────────────────────────────────────────
section('Flight-Risk Report');

// Score all employees
$allPred = $pipeline->predict($ds)->toFlatArray();

// Bin into risk tiers
$tiers = ['LOW' => 0, 'MEDIUM' => 0, 'HIGH' => 0, 'CRITICAL' => 0];
foreach ($allPred as $score) {
    $tiers[match(true) {
        $score >= 0.70 => 'CRITICAL',
        $score >= 0.50 => 'HIGH',
        $score >= 0.30 => 'MEDIUM',
        default        => 'LOW',
    }]++;
}

foreach ($tiers as $tier => $count) {
    printf("  %-10s : %4d employees\n", $tier, $count);
}

// ── 5. Targeted intervention ──────────────────────────────────────────────────
section('Intervention Recommendations');

$actions = [
    'CRITICAL' => '🔴 Immediate 1:1 with manager + retention bonus review',
    'HIGH'     => '🟠 Quarterly check-in + career development plan',
    'MEDIUM'   => '🟡 Annual review brought forward + workload review',
    'LOW'      => '🟢 Standard engagement programme',
];

foreach ($actions as $tier => $action) {
    printf("  %-10s → %s\n", $tier, $action);
}

// ── 6. Save production pipeline ───────────────────────────────────────────────
$dir = sys_get_temp_dir() . '/pml_attrition';
$pipeline->save($dir);
metric('Pipeline saved to', $dir);

echo "\n✓ Done\n";
