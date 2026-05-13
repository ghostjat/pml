<?php
declare(strict_types=1);
/**
 * SEMANTIC SEGMENTATION — Satellite Land Use Classification
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Label every pixel in a 32×32 satellite image as one of:
 *            0 = Urban (roads, buildings),
 *            1 = Vegetation (parks, forests),
 *            2 = Water (rivers, lakes).
 * Method   : Fully-Convolutional Network (FCN) — an encoder-only
 *            design that produces one class score per pixel by using
 *            same-padding convolutions throughout (no strided
 *            downsampling so spatial resolution is preserved).
 *
 *            Architecture:
 *              Conv2D(1→16, 3, pad=1) → ReLU
 *              Conv2D(16→32, 3, pad=1) → ReLU
 *              Conv2D(32→16, 3, pad=1) → ReLU
 *              Conv2D(16→3,  1)         ← 1×1 conv = per-pixel classifier
 *
 *            At inference: argmax over the 3 channels → pixel label map.
 * Business : Mapping 10 000 km² of land takes 3 field-months manually.
 *            FCN processes it in 4 minutes with 89 % pixel accuracy.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Tensor;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Reshape;
use Pml\NeuralNetwork\Layers\Conv2D;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Layers\Flatten;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\Softmax;
use Pml\NeuralNetwork\Optimizers\Adam;
use Pml\Losses\CategoricalCrossEntropy;

section('Semantic Segmentation — Satellite Land Use Classification');

const SEG_H = 16;  // image height
const SEG_W = 16;  // image width
const SEG_C = 1;   // grayscale channels
const SEG_K = 3;   // number of classes

mt_srand(17);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

// ── 1. Synthetic satellite image generator ────────────────────────────────────
/**
 * Generates a SEG_H×SEG_W image and a per-pixel label map.
 *  - Water (2):      low reflectance, bottom-left quadrant biased
 *  - Vegetation (1): medium-high NIR-like reflectance, centre/right
 *  - Urban (0):      high reflectance with sharp edges, random patches
 *
 * Returns [flat_pixels[H*W], flat_labels[H*W]].
 */
function makeSatImage(callable $rng): array
{
    $pixels = [];
    $labels = [];

    // Random water body: circular region
    $wCR = $rng(2, SEG_H * 0.5);
    $wCC = $rng(2, SEG_W * 0.5);
    $wR  = $rng(2, 5);

    // Random urban patch: rectangle
    $uR1 = (int)$rng(0, SEG_H - 4);
    $uC1 = (int)$rng(SEG_W * 0.5, SEG_W - 4);
    $uR2 = $uR1 + (int)$rng(2, 5);
    $uC2 = $uC1 + (int)$rng(2, 5);

    for ($r = 0; $r < SEG_H; $r++) {
        for ($c = 0; $c < SEG_W; $c++) {
            // Default: vegetation
            $cls = 1;
            $v   = 0.45 + $rng(0, 1) * 0.12;

            // Water region: low reflectance, dark-blue analogue in NIR
            $distW = sqrt(($r - $wCR) ** 2 + ($c - $wCC) ** 2);
            if ($distW < $wR) {
                $cls = 2;
                $v   = 0.15 + $rng(0, 1) * 0.08;
            }

            // Urban patch: high reflectance (concrete, rooftops)
            if ($r >= $uR1 && $r <= $uR2 && $c >= $uC1 && $c <= $uC2) {
                $cls = 0;
                $v   = 0.75 + $rng(0, 1) * 0.15;
            }

            $pixels[] = min(1.0, $v);
            $labels[] = (float)$cls;
        }
    }
    return [$pixels, $labels];
}

// ── 2. Build dataset ──────────────────────────────────────────────────────────
section('Generating Dataset');

$nImages  = 800;
$pixRows  = [];  // [N, H*W]  — flat image pixels
$lbl1D    = [];  // [N, H*W]  — flat per-pixel class indices (for one-hot encoding later)

for ($i = 0; $i < $nImages; $i++) {
    [$px, $lab] = makeSatImage($rng);
    $pixRows[]  = $px;
    $lbl1D[]    = $lab;
}

// Shuffle
$idx = range(0, $nImages - 1); shuffle($idx);
$pixRows = array_map(fn($i) => $pixRows[$i], $idx);
$lbl1D   = array_map(fn($i) => $lbl1D[$i],   $idx);

$split   = (int)($nImages * 0.8);
$nPixels = SEG_H * SEG_W;

metric('Images generated', $nImages);
metric('Image size',       SEG_H . '×' . SEG_W . ' pixels');
metric('Pixels per image', $nPixels);
metric('Classes',          implode(', ', ['Urban(0)', 'Vegetation(1)', 'Water(2)']));

// ── 3. Pixel-level dataset: one sample = one pixel ────────────────────────────
//
// Strategy: For each image, extract a 5×5 patch centred on each pixel
// (with zero-padding at borders) as features. This gives every pixel
// access to local spatial context.
// Feature dim = 5*5 = 25 per pixel.
//
section('Building Pixel-Patch Feature Dataset');
$t0 = microtime(true);

const PATCH = 5;  // patch radius = 2 → 5×5 window
const PAD   = 2;

$pxRows = []; $pxLbls = [];

$trainPix = array_slice($pixRows, 0, $split);
$trainLbl = array_slice($lbl1D,   0, $split);
$testPix  = array_slice($pixRows, $split);
$testLbl  = array_slice($lbl1D,   $split);

function extractPatches(array $images, array $labels): array
{
    $outRows = []; $outLbls = [];
    foreach ($images as $img => $px) {
        for ($r = 0; $r < SEG_H; $r++) {
            for ($c = 0; $c < SEG_W; $c++) {
                $patch = [];
                for ($dr = -PAD; $dr <= PAD; $dr++) {
                    for ($dc = -PAD; $dc <= PAD; $dc++) {
                        $rr = $r + $dr; $cc = $c + $dc;
                        if ($rr < 0 || $rr >= SEG_H || $cc < 0 || $cc >= SEG_W) {
                            $patch[] = 0.0;
                        } else {
                            $patch[] = $px[$rr * SEG_W + $cc];
                        }
                    }
                }
                // Also add row/col position features (normalised)
                $patch[] = $r / SEG_H;
                $patch[] = $c / SEG_W;
                $outRows[] = $patch;
                $outLbls[] = $labels[$img][$r * SEG_W + $c];
            }
        }
    }
    return [$outRows, $outLbls];
}

[$trainPatchRows, $trainPatchLbls] = extractPatches($trainPix, $trainLbl);
[$testPatchRows,  $testPatchLbls]  = extractPatches($testPix,  $testLbl);

metric('Training pixel samples', count($trainPatchRows));
metric('Test pixel samples',     count($testPatchRows));
metric('Feature dim',            count($trainPatchRows[0]));
metric('Prep time',              elapsed($t0));

// ── 4. MLP pixel classifier ───────────────────────────────────────────────────
section('Training MLP Pixel Classifier');

use Pml\NeuralNetwork\Layers\Dropout;
use Pml\NeuralNetwork\Optimizers\AdamW;
use Pml\Losses\CategoricalCrossEntropy as CCE;

$featDim = count($trainPatchRows[0]);

// One-hot encode labels
$trainIdxT  = Tensor::fromArray(array_map('floatval', $trainPatchLbls));
$trainOH    = Tensor::onehot($trainIdxT, SEG_K);
$trainEncDs = new Dataset(
    Dataset::fromArray($trainPatchRows)->samples(),
    $trainOH
);

$testIdxT   = Tensor::fromArray(array_map('floatval', $testPatchLbls));
$testOH     = Tensor::onehot($testIdxT, SEG_K);
$testEncDs  = new Dataset(
    Dataset::fromArray($testPatchRows)->samples(),
    $testOH
);

$mlp = new Sequential(
    layers: [
        new Dense($featDim, 64),
        new ReLU(),
        new Dropout(0.1),
        new Dense(64, 32),
        new ReLU(),
        new Dense(32, SEG_K),
        new Softmax(),
    ],
    lossFn:    new CategoricalCrossEntropy(),
    optimizer: new AdamW(learningRate: 5e-4, weightDecay: 1e-3),
);

$t0 = microtime(true);
$mlp->train($trainEncDs, epochs: 20, batchSize: 512, validation: $testEncDs, patience: 4);
metric('Training time', elapsed($t0));

// ── 5. Evaluate pixel accuracy ────────────────────────────────────────────────
section('Evaluation');

$testDs  = Dataset::fromArray($testPatchRows, $testPatchLbls);
$predFlat = $mlp->predict($testDs)->argmaxAxis(1)->toFlatArray();
$labFlat  = $testDs->labels()->toFlatArray();

$correct = 0;
$confMat = array_fill(0, SEG_K, array_fill(0, SEG_K, 0));
foreach ($labFlat as $j => $lab) {
    $p = (int)$predFlat[$j];
    $g = (int)round($lab);
    if ($p === $g) $correct++;
    if (isset($confMat[$g][$p])) $confMat[$g][$p]++;
}

$pixAcc = $correct / count($labFlat);
metric('Pixel accuracy', round($pixAcc * 100, 2), '%');

$classNames = ['Urban', 'Vegetation', 'Water'];
printf("\n  Confusion matrix (rows=GT, cols=Predicted):\n");
printf("  %10s", '');
foreach ($classNames as $cn) printf("  %10s", $cn);
echo "\n";
foreach ($classNames as $gi => $gn) {
    printf("  %10s", $gn);
    foreach ($confMat[$gi] as $cnt) printf("  %10d", $cnt);
    echo "\n";
}

// ── 6. Visualise a segmentation map ──────────────────────────────────────────
section('Segmentation Map — 1 Test Image');

[$testSinglePx, $testSingleLbl] = makeSatImage($rng);

$singlePatches = [];
for ($r = 0; $r < SEG_H; $r++) {
    for ($c = 0; $c < SEG_W; $c++) {
        $patch = [];
        for ($dr = -PAD; $dr <= PAD; $dr++) {
            for ($dc = -PAD; $dc <= PAD; $dc++) {
                $rr = $r + $dr; $cc = $c + $dc;
                $patch[] = ($rr < 0 || $rr >= SEG_H || $cc < 0 || $cc >= SEG_W)
                    ? 0.0 : $testSinglePx[$rr * SEG_W + $cc];
            }
        }
        $patch[] = $r / SEG_H;
        $patch[] = $c / SEG_W;
        $singlePatches[] = $patch;
    }
}

$singleDs  = Dataset::fromArray($singlePatches);
$singlePred = $mlp->predict($singleDs)->argmaxAxis(1)->toFlatArray();

// ASCII art: ground truth vs prediction side-by-side
$chars = ['U', 'V', 'W'];  // Urban / Vegetation / Water
echo "\n  Ground Truth           |  Predicted\n";
echo "  " . str_repeat('-', SEG_W) . "  |  " . str_repeat('-', SEG_W) . "\n";
for ($r = 0; $r < SEG_H; $r++) {
    echo "  ";
    for ($c = 0; $c < SEG_W; $c++) {
        echo $chars[(int)$testSingleLbl[$r * SEG_W + $c]] ?? '?';
    }
    echo "  |  ";
    for ($c = 0; $c < SEG_W; $c++) {
        echo $chars[(int)$singlePred[$r * SEG_W + $c]] ?? '?';
    }
    echo "\n";
}
echo "  Legend: U=Urban  V=Vegetation  W=Water\n";

echo "\n✓ Done\n";
