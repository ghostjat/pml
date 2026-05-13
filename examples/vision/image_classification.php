<?php
declare(strict_types=1);
/**
 * CNN IMAGE CLASSIFICATION — Medical X-Ray Screening
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Classify 32×32 grayscale "X-ray" images into 3 classes:
 *            0 = Normal, 1 = Nodule (circular anomaly),
 *            2 = Fracture (linear crack pattern).
 * Model    : Convolutional Neural Network
 *            Conv2D(1→16) → ReLU → Conv2D(16→32, stride=2) → ReLU
 *            → Conv2D(32→64, stride=2) → ReLU → GlobalAvgPool2D
 *            → Dense(64→3) → Softmax
 * Business : AI-assisted radiology cuts radiologist reading time by
 *            40 % and catches 12 % more early-stage anomalies.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Tensor;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Reshape;
use Pml\NeuralNetwork\Layers\Conv2D;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Layers\Dropout;
use Pml\NeuralNetwork\Layers\GlobalAveragePooling2D;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\Softmax;
use Pml\NeuralNetwork\Optimizers\Adam;
use Pml\Losses\CategoricalCrossEntropy;
use Pml\Metrics\Classification\Accuracy;

section('CNN Image Classification — Medical X-Ray Screening');

// ── Constants ─────────────────────────────────────────────────────────────────
const IMG_H = 32;
const IMG_W = 32;
const N_CLASSES = 3;

mt_srand(42);

// ── 1. Synthetic image generator ─────────────────────────────────────────────
/**
 * Returns a flat [H*W] pixel array (float32, 0-1) for a given class.
 * Each class has distinct spatial structure a CNN can learn.
 */
function makeXrayImage(int $class): array
{
    $px = [];
    for ($r = 0; $r < IMG_H; $r++) {
        for ($c = 0; $c < IMG_W; $c++) {
            // Background noise (lung tissue texture)
            $v = 0.15 + (mt_rand(0, 1000) / 1000.0) * 0.10;

            if ($class === 0) {
                // Normal: soft gradient (uniform tissue, no anomaly)
                $v += 0.45 * (1.0 - abs($r - IMG_H / 2) / (IMG_H / 2))
                           * (1.0 - abs($c - IMG_W / 2) / (IMG_W / 2));

            } elseif ($class === 1) {
                // Nodule: bright circle at random center (tumour-like blob)
                $cr = mt_rand((int)(IMG_H * 0.3), (int)(IMG_H * 0.7));
                $cc = mt_rand((int)(IMG_W * 0.3), (int)(IMG_W * 0.7));
                $rad = mt_rand(4, 7);
                $dist = sqrt(($r - $cr) ** 2 + ($c - $cc) ** 2);
                if ($dist < $rad) {
                    $v += 0.7 * max(0, 1.0 - $dist / $rad);
                }

            } else {
                // Fracture: diagonal high-density line (bone crack)
                $offset = mt_rand(-4, 4);
                $dist = abs(($r - $c) + $offset) / sqrt(2);
                if ($dist < 1.5) {
                    $v += 0.8;
                }
            }

            $px[] = min(1.0, $v);
        }
    }
    return $px;
}

// ── 2. Dataset generation ────────────────────────────────────────────────────
section('Generating Synthetic X-Ray Dataset');

$nPerClass = 400;
$total     = $nPerClass * N_CLASSES;

$rows = []; $lbls = [];
for ($cls = 0; $cls < N_CLASSES; $cls++) {
    for ($i = 0; $i < $nPerClass; $i++) {
        $rows[] = makeXrayImage($cls);
        $lbls[] = (float)$cls;
    }
}

// Shuffle
$idx = range(0, $total - 1); shuffle($idx);
$rows = array_map(fn($i) => $rows[$i], $idx);
$lbls = array_map(fn($i) => $lbls[$i], $idx);

$split   = (int)($total * 0.8);
$trainDs = Dataset::fromArray(array_slice($rows, 0, $split), array_slice($lbls, 0, $split));
$testDs  = Dataset::fromArray(array_slice($rows, $split),    array_slice($lbls, $split));

metric('Image size',      IMG_H . '×' . IMG_W . ' px (grayscale)');
metric('Training images', $trainDs->numRows());
metric('Test images',     $testDs->numRows());

// ── 3. One-hot encode labels ──────────────────────────────────────────────────
function makeOneHot(Dataset $ds, int $nClasses): Dataset
{
    $flat  = $ds->labels()->toFlatArray();
    $idxT  = Tensor::fromArray(array_map('floatval', $flat));
    $ohArr = Tensor::onehot($idxT, $nClasses);
    return new Dataset($ds->samples(), $ohArr);
}

$trainEncoded = makeOneHot($trainDs, N_CLASSES);
$testEncoded  = makeOneHot($testDs,  N_CLASSES);

// ── 4. CNN Architecture ───────────────────────────────────────────────────────
section('Building CNN');

// Input:  [B, 1024]  (32×32 flattened)
// Reshape:[B, 1, 32, 32]
// Conv1:  [B, 16, 32, 32]  (3×3, pad=1, same)
// Conv2:  [B, 32, 16, 16]  (3×3, stride=2, pad=1)
// Conv3:  [B, 64,  8,  8]  (3×3, stride=2, pad=1)
// GAP:    [B, 64]
// Dense:  [B,  3]
// Softmax:[B,  3]

$cnn = new Sequential(
    layers: [
        new Reshape([1, IMG_H, IMG_W]),
        new Conv2D(inChannels: 1,  outChannels: 16, kernelSize: 3, stride: 1, padding: 1),
        new ReLU(),
        new Conv2D(inChannels: 16, outChannels: 32, kernelSize: 3, stride: 2, padding: 1),
        new ReLU(),
        new Dropout(0.1),
        new Conv2D(inChannels: 32, outChannels: 64, kernelSize: 3, stride: 2, padding: 1),
        new ReLU(),
        new GlobalAveragePooling2D(),
        new Dense(64, 3),
        new Softmax(),
    ],
    lossFn:    new CategoricalCrossEntropy(),
    optimizer: new Adam(learningRate: 1e-3),
);

printf("  Architecture  : Reshape→Conv2D(16)→Conv2D(32)→Conv2D(64)→GAP→Dense(3)\n");
printf("  Input shape   : [B, %d]  →  [B, 1, %d, %d] after Reshape\n", IMG_H * IMG_W, IMG_H, IMG_W);
printf("  Feature maps  : 16 → 32 → 64 channels\n");
printf("  Parameters    : ~%.1f k (compact vs dense baseline ~%.1f k)\n",
       (1*16*9 + 16*32*9 + 32*64*9 + 64*3) / 1000,
       (IMG_H * IMG_W * 256 + 256 * 3) / 1000);

// ── 5. Train ──────────────────────────────────────────────────────────────────
section('Training');
$t0 = microtime(true);

$cnn->train(
    $trainEncoded,
    epochs:    25,
    batchSize: 32,
    validation: $testEncoded,
    patience:  5,
);

metric('Training time', elapsed($t0));

// ── 6. Evaluate ───────────────────────────────────────────────────────────────
section('Evaluation');

// predict() returns softmax probabilities [B, K] — argmax to get class indices [B]
$probas = $cnn->predict($testDs);
$pred   = $probas->argmaxAxis(1);
$labels = $testDs->labels();
metric('Test accuracy', (new Accuracy())->score($pred, $labels));

// Per-class breakdown
$classNames = ['Normal', 'Nodule', 'Fracture'];
$predFlat   = $pred->toFlatArray();
$labFlat    = $labels->toFlatArray();

$correct = array_fill(0, N_CLASSES, 0);
$total   = array_fill(0, N_CLASSES, 0);
foreach ($labFlat as $j => $lab) {
    $total[(int)round($lab)]++;
    if ((int)round($predFlat[$j]) === (int)round($lab)) {
        $correct[(int)round($lab)]++;
    }
}
printf("\n");
foreach ($classNames as $k => $name) {
    printf("  %-10s : %d/%d correct  (%.1f%%)\n",
           $name, $correct[$k], $total[$k],
           $total[$k] > 0 ? 100 * $correct[$k] / $total[$k] : 0);
}

// ── 7. Live inference ─────────────────────────────────────────────────────────
section('Live Inference — 3 Scan Examples');

for ($cls = 0; $cls < N_CLASSES; $cls++) {
    $pixels  = makeXrayImage($cls);
    $imgDs   = Dataset::fromArray([$pixels]);
    $predIdx = (int)$cnn->predict($imgDs)->argmaxAxis(1)->toFlatArray()[0];
    $correct = $predIdx === $cls ? '✅' : '❌';
    printf("  Ground truth: %-10s → Predicted: %-10s  %s\n",
           $classNames[$cls], $classNames[$predIdx], $correct);
}

echo "\n✓ Done\n";
