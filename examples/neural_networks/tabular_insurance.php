<?php
declare(strict_types=1);
/**
 * INSURANCE CLAIM APPROVAL — Deep Learning on Tabular Data
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Predict whether an insurance claim will be approved.
 * Model    : Sequential MLP with batch norm, dropout, and AdamW.
 *            Deep learning on tabular data beats tree models when
 *            there are many continuous features with complex interactions.
 * Business : Auto-approving 80 % of clear-cut claims reduces
 *            adjuster workload, speeds payouts, and improves NPS.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\Gelu;
use Pml\NeuralNetwork\Layers\Dropout;
use Pml\NeuralNetwork\Layers\BatchNormalization;
use Pml\NeuralNetwork\Layers\Sigmoid;
use Pml\NeuralNetwork\Optimizers\AdamW;
use Pml\Losses\BinaryCrossEntropy;
use Pml\Transformers\StandardScaler;
use Pml\Metrics\Classification\RocAuc;
use Pml\Metrics\Classification\F1Score;
use Pml\Metrics\Classification\Accuracy;

section('Insurance Claim Approval — Sequential MLP');

// ── 1. Claims dataset ─────────────────────────────────────────────────────────
// Features: claim_amount, deductible, policy_age_months, num_prior_claims,
//           days_since_incident, repair_estimate, police_report (0/1),
//           witness_count, driver_age, vehicle_age_years, premium_monthly

mt_srand(31);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

$rows = []; $lbls = [];

for ($i = 0; $i < 5000; $i++) {
    $amount    = $rng(200, 50000);
    $deduct    = $rng(250, 5000);
    $policyAge = $rng(1, 120);
    $priorClm  = (int)$rng(0, 4);
    $daysInc   = $rng(0, 30);
    $repEst    = $amount * $rng(0.7, 1.3);
    $police    = (mt_rand(0, 1));
    $witness   = (int)$rng(0, 3);
    $driverAge = $rng(18, 75);
    $vehicleAge= $rng(0, 20);
    $premium   = $rng(50, 500);

    $approveProb = 0.60
        - ($amount > 20000 ? 0.15 : 0)
        - $priorClm * 0.08
        + $police * 0.15
        + $witness * 0.05
        - ($daysInc > 14 ? 0.10 : 0)
        + ($policyAge > 24 ? 0.10 : 0)
        - (abs($repEst - $amount) / max($amount, 1) > 0.5 ? 0.20 : 0);

    $approveProb = max(0.05, min(0.95, $approveProb));

    $rows[] = [$amount, $deduct, $policyAge, (float)$priorClm, $daysInc,
               $repEst, (float)$police, (float)$witness, $driverAge, $vehicleAge, $premium];
    $lbls[] = ((mt_rand() / mt_getrandmax()) < $approveProb) ? 1.0 : 0.0;
}

// Scale features
$rawDs  = Dataset::fromArray($rows, $lbls);
$scaler = new StandardScaler();
$scaler->fit($rawDs);
$scaled = $scaler->transform($rawDs);
[$train, $test] = $scaled->randomize()->split(0.8);

metric('Training claims', $train->numRows());
metric('Test claims',     $test->numRows());

// ── 2. Build MLP ──────────────────────────────────────────────────────────────
section('Building Network');

$net = new Sequential(
    layers: [
        new Dense(11, 128),
        new BatchNormalization(128),
        new Gelu(),
        new Dropout(0.2),
        new Dense(128, 64),
        new Gelu(),
        new Dropout(0.15),
        new Dense(64, 32),
        new Gelu(),
        new Dense(32, 1),
        new Sigmoid(),
    ],
    lossFn:    new BinaryCrossEntropy(),
    optimizer: new AdamW(learningRate: 1e-3, weightDecay: 1e-2),
);

// ── 3. Train ──────────────────────────────────────────────────────────────────
section('Training');
$t0 = microtime(true);

$net->train($train, epochs: 30, batchSize: 64, validation: $test, patience: 5);

metric('Training time', elapsed($t0));

// ── 4. Evaluate ───────────────────────────────────────────────────────────────
section('Evaluation');
$pred   = $net->predict($test)->flatten();  // [N,1] → [N]
$labels = $test->labels();

metric('ROC-AUC',   (new RocAuc())->score($pred, $labels));
metric('F1-Score',  (new F1Score())->score($pred, $labels));
metric('Accuracy',  (new Accuracy())->score($pred, $labels));

// ── 5. Claims decision engine ─────────────────────────────────────────────────
section('Adjuster Decision Engine');

$cases = [
    'Clear-cut fender bender' => [1800, 500, 48, 0, 1, 1900, 1, 1, 35, 3, 120],
    'Suspicious large claim'  => [48000, 1000, 6, 2, 25, 32000, 0, 0, 22, 15, 80],
    'Loyal customer, minor'   => [600, 250, 96, 1, 3, 650, 1, 2, 55, 8, 200],
];

foreach ($cases as $desc => $features) {
    $sampleDs  = Dataset::fromArray([$features]);
    $sampleSc  = $scaler->transform($sampleDs);
    $score     = $net->predict($sampleSc)->toFlatArray()[0] ?? 0.5;
    $decision  = match(true) {
        $score >= 0.75 => '✅ AUTO-APPROVE',
        $score >= 0.45 => '📋 MANUAL REVIEW',
        default        => '❌ FLAG / INVESTIGATE',
    };
    printf("  %-30s score=%.3f  %s\n", $desc, $score, $decision);
}

echo "\n✓ Done\n";
