<?php
declare(strict_types=1);
/**
 * VIDEO ACTION RECOGNITION — Workplace Safety Monitoring
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Classify worker actions from short video clips (1 sec):
 *            0 = Normal work (walking, tool use, desk work)
 *            1 = Fall detected (sudden drop + stationary)
 *            2 = Ergonomic violation (repetitive bending / overhead reach)
 *            3 = Restricted zone entry (worker in danger area)
 *
 * Method   : Temporal feature extraction from synthetic pose-proxy
 *            sequences → Sequential MLP classifier.
 *
 *            Each clip = 10 frames of 12 body-joint coordinates
 *            (x, y, confidence) = 36 features/frame → 360 features/clip.
 *            The MLP learns spatio-temporal patterns over the clip.
 *
 * Business : Workplace injuries cost $170 B/year in the US alone.
 *            Real-time AI monitoring cuts lost-time injuries by 35 %
 *            in industrial facilities (McKinsey, 2023).
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Tensor;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Layers\Dropout;
use Pml\NeuralNetwork\Layers\BatchNormalization;
use Pml\NeuralNetwork\Layers\Softmax;
use Pml\NeuralNetwork\Optimizers\AdamW;
use Pml\Losses\CategoricalCrossEntropy;
use Pml\Metrics\Classification\Accuracy;
use Pml\Metrics\Classification\F1Score;

section('Video Action Recognition — Workplace Safety');

const N_FRAMES  = 10;   // frames per clip at 10 FPS = 1-second clip
const N_JOINTS  = 12;   // body keypoints: head, shoulders, elbows, wrists, hips, knees, ankles
const N_COORDS  = 3;    // x, y, confidence per joint
const CLIP_DIM  = N_FRAMES * N_JOINTS * N_COORDS;  // 360 features per clip
const N_ACTIONS = 4;

mt_srand(314);
$rng    = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);
$randn  = function() use ($rng): float {
    return sqrt(-2 * log(max(1e-10, $rng(0, 1)))) * cos(2 * M_PI * $rng(0, 1));
};

// ── 1. Synthetic pose-sequence generator ─────────────────────────────────────
/**
 * Returns a flat [N_FRAMES × N_JOINTS × N_COORDS] array of pose keypoints.
 *
 * Joint layout (indices):
 *   0=head, 1=neck, 2=r_shoulder, 3=l_shoulder, 4=r_elbow, 5=l_elbow,
 *   6=r_wrist, 7=l_wrist, 8=r_hip, 9=l_hip, 10=r_knee, 11=l_knee
 *
 * Each joint: [x, y, confidence]  (x,y normalised 0–1, conf 0.7–1.0)
 */
function makePoseClip(int $action, callable $rng, callable $randn): array
{
    // Base standing pose (normalised screen coords)
    $base = [
        [0.50, 0.10],  // head
        [0.50, 0.18],  // neck
        [0.58, 0.22],  // r_shoulder
        [0.42, 0.22],  // l_shoulder
        [0.62, 0.35],  // r_elbow
        [0.38, 0.35],  // l_elbow
        [0.64, 0.48],  // r_wrist
        [0.36, 0.48],  // l_wrist
        [0.54, 0.52],  // r_hip
        [0.46, 0.52],  // l_hip
        [0.54, 0.72],  // r_knee
        [0.46, 0.72],  // l_knee
    ];

    $clip = [];

    for ($f = 0; $f < N_FRAMES; $f++) {
        $t = $f / N_FRAMES;

        foreach ($base as $j => [$bx, $by]) {
            $x = $bx; $y = $by;

            switch ($action) {
                case 0: // Normal work: slow drift, slight arm motion
                    $x += $randn() * 0.02;
                    $y += $randn() * 0.02;
                    // Arm swinging
                    if ($j >= 4 && $j <= 7) {
                        $x += sin($t * M_PI * 4) * 0.04;
                    }
                    break;

                case 1: // Fall: rapid downward y shift + body collapses horizontal
                    $yShift = min(0.4, $t * 0.7);   // person descends
                    $y += $yShift;
                    // Joints spread out horizontally as person falls
                    $x += ($j % 2 === 0 ? 1 : -1) * $t * 0.15;
                    // After t=0.5, person is on floor (joints cluster at bottom)
                    if ($t > 0.5) { $y = 0.85 + $randn() * 0.03; }
                    break;

                case 2: // Ergonomic violation: repetitive bending (torso bows forward)
                    $bendAngle = abs(sin($t * M_PI * 3)) * 0.3;
                    if ($j <= 3) { // upper body bends forward
                        $y += $bendAngle;
                        $x += ($j < 2 ? 1 : -1) * $bendAngle * 0.5;
                    }
                    // Wrists reach down (overhead on odd cycles)
                    if ($j >= 6 && $j <= 7) {
                        $y += sin($t * M_PI * 3) > 0 ? 0.25 : -0.15;
                    }
                    break;

                case 3: // Restricted zone entry: moves to corner of frame quickly
                    $zoneX = 0.10; $zoneY = 0.80;
                    $x = $bx + ($zoneX - $bx) * min(1.0, $t * 2);
                    $y = $by + ($zoneY - $by) * min(1.0, $t * 2);
                    break;
            }

            $conf = max(0.5, min(1.0, 0.88 + $randn() * 0.06));
            $clip[] = min(1.0, max(0.0, $x));
            $clip[] = min(1.0, max(0.0, $y));
            $clip[] = $conf;
        }
    }
    return $clip;
}

// ── 2. Dataset generation ────────────────────────────────────────────────────
section('Generating Synthetic Pose-Clip Dataset');

$nPerClass = 600;
$total     = $nPerClass * N_ACTIONS;
$rows = []; $lbls = [];

for ($cls = 0; $cls < N_ACTIONS; $cls++) {
    for ($i = 0; $i < $nPerClass; $i++) {
        $rows[] = makePoseClip($cls, $rng, $randn);
        $lbls[] = (float)$cls;
    }
}

$idx = range(0, $total - 1); shuffle($idx);
$rows = array_map(fn($i) => $rows[$i], $idx);
$lbls = array_map(fn($i) => $lbls[$i], $idx);

$split   = (int)($total * 0.8);
$trainDs = Dataset::fromArray(array_slice($rows, 0, $split), array_slice($lbls, 0, $split));
$testDs  = Dataset::fromArray(array_slice($rows, $split),    array_slice($lbls, $split));

metric('Clip length',      N_FRAMES . ' frames × ' . N_JOINTS . ' joints × ' . N_COORDS . ' coords');
metric('Feature dim',      CLIP_DIM);
metric('Training clips',   $trainDs->numRows());
metric('Test clips',       $testDs->numRows());

// ── 3. One-hot encode ─────────────────────────────────────────────────────────
function oneHotDs(Dataset $ds, int $k): Dataset {
    $idxT = Tensor::fromArray(array_map('floatval', $ds->labels()->toFlatArray()));
    return new Dataset($ds->samples(), Tensor::onehot($idxT, $k));
}

$trainEnc = oneHotDs($trainDs, N_ACTIONS);
$testEnc  = oneHotDs($testDs,  N_ACTIONS);

// ── 4. Temporal MLP ────────────────────────────────────────────────────────────
section('Building Temporal MLP');

// Architecture: treats the 360-dim clip as a flat temporal feature vector.
// Two hidden layers with batch norm + dropout, then 4-class head.

$net = new Sequential(
    layers: [
        new Dense(CLIP_DIM, 256),
        new BatchNormalization(256),
        new ReLU(),
        new Dropout(0.25),
        new Dense(256, 128),
        new BatchNormalization(128),
        new ReLU(),
        new Dropout(0.15),
        new Dense(128, 64),
        new ReLU(),
        new Dense(64, N_ACTIONS),
        new Softmax(),
    ],
    lossFn:    new CategoricalCrossEntropy(),
    optimizer: new AdamW(learningRate: 5e-4, weightDecay: 1e-3),
);

printf("  Input  : %d features (10 frames × 12 joints × 3 coords)\n", CLIP_DIM);
printf("  Layers : Dense(256) → BN → ReLU → Dense(128) → BN → ReLU → Dense(64) → Dense(%d)\n", N_ACTIONS);

// ── 5. Train ──────────────────────────────────────────────────────────────────
section('Training');
$t0 = microtime(true);

$net->train(
    $trainEnc,
    epochs:    40,
    batchSize: 64,
    validation: $testEnc,
    patience:  6,
);

metric('Training time', elapsed($t0));

// ── 6. Evaluate ───────────────────────────────────────────────────────────────
section('Evaluation');

$probas = $net->predict($testDs);
$pred   = $probas->argmaxAxis(1);
$labels = $testDs->labels();
metric('Accuracy', (new Accuracy())->score($pred, $labels));

$actions    = ['Normal', 'Fall', 'Ergo Violation', 'Zone Entry'];
$predFlat   = $pred->toFlatArray();
$labFlat    = $labels->toFlatArray();

$corr = array_fill(0, N_ACTIONS, 0);
$tot  = array_fill(0, N_ACTIONS, 0);
foreach ($labFlat as $j => $lab) {
    $tot[(int)round($lab)]++;
    if ((int)round($predFlat[$j]) === (int)round($lab)) {
        $corr[(int)round($lab)]++;
    }
}
printf("\n");
foreach ($actions as $k => $name) {
    $pct = $tot[$k] > 0 ? 100 * $corr[$k] / $tot[$k] : 0;
    printf("  %-18s : %3d/%3d  (%.1f%%)\n", $name, $corr[$k], $tot[$k], $pct);
}

// ── 7. Live safety monitoring ─────────────────────────────────────────────────
section('Live Safety Monitor — Simulated Incidents');

$incidents = [
    ['desc' => 'Worker walking to station',    'action' => 0],
    ['desc' => 'Worker slips on wet floor',    'action' => 1],
    ['desc' => 'Repetitive overhead reach',    'action' => 2],
    ['desc' => 'Enters machinery danger zone', 'action' => 3],
    ['desc' => 'Picking items from shelf',     'action' => 0],
    ['desc' => 'Sudden fall from ladder',      'action' => 1],
];

$alerts = [
    0 => '✅ Normal  — no action',
    1 => '🚨 FALL ALERT — dispatch first aid immediately',
    2 => '⚠️  ERGO ALERT — task rotation required',
    3 => '🔴 ZONE ALERT — stop machinery + evacuate',
];

printf("\n  %-35s | GT            | Predicted     | Action\n", 'Incident');
printf("  %s\n", str_repeat('-', 100));

foreach ($incidents as $inc) {
    $clip    = makePoseClip($inc['action'], $rng, $randn);
    $ds      = Dataset::fromArray([$clip]);
    $predIdx = (int)$net->predict($ds)->argmaxAxis(1)->toFlatArray()[0];
    $correct = $predIdx === $inc['action'] ? '' : ' ❌';
    printf("  %-35s | %-14s| %-14s| %s%s\n",
           $inc['desc'],
           $actions[$inc['action']],
           $actions[$predIdx],
           $alerts[$predIdx],
           $correct);
}

echo "\n✓ Done\n";
