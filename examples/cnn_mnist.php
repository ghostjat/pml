<?php

declare(strict_types=1);

/**
 * ════════════════════════════════════════════════════════════════════════════
 *  examples/cnn_mnist.php — LeNet-style CNN on synthetic 16×16 images
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Demonstrates a convolutional neural network using:
 *   Conv2D  →  ReLU  →  MaxPool2D  →  Conv2D  →  ReLU  →  MaxPool2D
 *   →  Flatten  →  Linear(128, 10)  →  softmax cross-entropy
 *
 * ── Architecture ─────────────────────────────────────────────────────────
 *
 *   Input      [B,  1, 16, 16]   synthetic 16×16 grayscale images
 *   Conv1      [B,  4, 16, 16]   Conv2D(1→4, k=3, pad=1)
 *   ReLU1      [B,  4, 16, 16]
 *   Pool1      [B,  4,  8,  8]   MaxPool2D(2)
 *   Conv2      [B,  8,  8,  8]   Conv2D(4→8, k=3, pad=1)
 *   ReLU2      [B,  8,  8,  8]
 *   Pool2      [B,  8,  4,  4]   MaxPool2D(2)
 *   Flatten    [B, 128]          8 × 4 × 4 = 128
 *   Linear     [B,  10]          W_fc [10, 128]  +  b_fc [10]
 *   CE Loss
 *
 * ── Synthetic Data ────────────────────────────────────────────────────────
 *
 *   10 classes.  Each class k has a bright 4×4 block at a class-specific
 *   (row, col) position within the 16×16 image, plus Gaussian noise.
 *   The spatial pattern is discriminative enough for a conv net to learn
 *   near-perfect accuracy within a few epochs.
 *
 * ── Why 16×16 (not 28×28)? ────────────────────────────────────────────────
 *
 *   im2col is implemented in PHP (data rearrangement only, no arithmetic).
 *   On 28×28 images the PHP loop count per batch grows to ~4M iterations;
 *   16×16 reduces this to ~300 K per batch, making the demo run in under
 *   a minute without a JIT compiler.  The architectural principles —
 *   im2col → sgemm, col2im, MaxPool argmax routing — are identical to
 *   full-resolution MNIST.
 *
 * Usage:
 *   php examples/cnn_mnist.php
 * ════════════════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../vendor/autoload.php';

use Pml\Tensor;
use Pml\BlasEngine;
use Pml\Layers\{Conv2D, MaxPool2D};
use Pml\Training\AdamW;

// ─── Hyper-parameters ─────────────────────────────────────────────────────

const IMG_H      = 16;
const IMG_W      = 16;
const N_CLASSES  = 10;
const N_TRAIN    = 500;
const N_TEST     = 100;
const BATCH      = 20;
const N_EPOCHS   = 15;
const LR_CNN     = 1e-3;
const WD_CNN     = 1e-4;

mt_srand(42);

// ─── 1. Synthetic data ────────────────────────────────────────────────────
//
//  Class k: bright 4×4 block at (row_k, col_k) = ((k / 4)*4, (k % 4)*4).
//  Remaining pixels: 0 + N(0, 0.25).  Block pixels: 1.5 + N(0, 0.25).
//

/**
 * Box-Muller N(0,1).
 */
function randn_cnn(): float
{
    static $spare    = null;
    static $hasSpare = false;
    if ($hasSpare) { $hasSpare = false; return $spare; }
    do { $u = mt_rand() / mt_getrandmax(); } while ($u === 0.0);
    $v      = mt_rand() / mt_getrandmax();
    $m      = sqrt(-2.0 * log($u));
    $spare  = $m * sin(2.0 * M_PI * $v);
    $hasSpare = true;
    return $m * cos(2.0 * M_PI * $v);
}

/**
 * Generate $n samples of class $label: flat float[IMG_H*IMG_W].
 */
function genSamples(int $label, int $n): array
{
    // 4×4 block top-left corner for this label
    $blockRow = intdiv($label, 4) * 4;   // rows 0, 4, 8 for labels 0-3, 4-7, 8-11
    $blockCol = ($label % 4) * 4;        // cols 0, 4, 8, 12

    $samples = [];
    for ($s = 0; $s < $n; $s++) {
        $img = array_fill(0, IMG_H * IMG_W, 0.0);
        for ($i = 0; $i < IMG_H; $i++) {
            for ($j = 0; $j < IMG_W; $j++) {
                $noise = 0.25 * randn_cnn();
                $inBlock = $i >= $blockRow && $i < $blockRow + 4
                        && $j >= $blockCol && $j < $blockCol + 4;
                $img[$i * IMG_W + $j] = ($inBlock ? 1.5 : 0.0) + $noise;
            }
        }
        $samples[] = $img;
    }
    return $samples;
}

$trainX = [];
$trainY = [];
$testX  = [];
$testY  = [];

$nPerClass      = intdiv(N_TRAIN, N_CLASSES);
$nPerClassTest  = intdiv(N_TEST,  N_CLASSES);

for ($cls = 0; $cls < N_CLASSES; $cls++) {
    $allSamples = genSamples($cls, $nPerClass + $nPerClassTest);
    for ($i = 0; $i < $nPerClass; $i++) {
        $trainX[] = $allSamples[$i];
        $trainY[] = $cls;
    }
    for ($i = $nPerClass; $i < $nPerClass + $nPerClassTest; $i++) {
        $testX[] = $allSamples[$i];
        $testY[] = $cls;
    }
}

// ─── 2. Tensor packing ────────────────────────────────────────────────────

/**
 * Pack a PHP float[][] (each row = flat [H*W]) into a Tensor [B, 1, H, W].
 */
function packImages(array $images): Tensor
{
    $B   = count($images);
    $out = new Tensor([$B, 1, IMG_H, IMG_W]);
    $off = 0;
    foreach ($images as $img) {
        foreach ($img as $px) {
            $out->buffer[$off++] = (float) $px;
        }
    }
    return $out;
}

$testTensor  = packImages($testX);

// ─── 3. Build LeNet ───────────────────────────────────────────────────────

$conv1 = new Conv2D(1,  4, 3, stride: 1, padding: 1);   // [B,4,16,16]
$pool1 = new MaxPool2D(2);                               // [B,4,8,8]
$conv2 = new Conv2D(4,  8, 3, stride: 1, padding: 1);   // [B,8,8,8]
$pool2 = new MaxPool2D(2);                               // [B,8,4,4]

// Linear head: [B, 128] → [B, 10]
$featDim  = 8 * 4 * 4;   // 128
$W_fc     = Tensor::randn([N_CLASSES, $featDim], 0.0, sqrt(2.0 / $featDim));
$b_fc     = Tensor::zeros([N_CLASSES]);
$W_fc->requiresGrad = true;
$b_fc->requiresGrad = true;

$optimizer = new AdamW(
    [...$conv1->parameters(), ...$conv2->parameters(), $W_fc, $b_fc],
    lr: LR_CNN,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weightDecay: WD_CNN,
);

// ─── 4. Forward helpers ───────────────────────────────────────────────────

/**
 * Element-wise ReLU.  Returns [activated_tensor, mask_tensor].
 * The mask stores 1.0 where input > 0 for the backward pass.
 */
function reluForward(Tensor $x): array
{
    $out  = new Tensor($x->shape);
    $mask = new Tensor($x->shape);
    for ($i = 0; $i < $x->size; $i++) {
        $v = (float) $x->buffer[$i];
        if ($v > 0.0) {
            $out->buffer[$i]  = $v;
            $mask->buffer[$i] = 1.0;
        }
        // else: buffer stays 0.0 (Tensor is zero-initialised)
    }
    return [$out, $mask];
}

/**
 * ReLU backward: d_in[i] = d_out[i] * mask[i].
 */
function reluBackward(Tensor $dout, Tensor $mask): Tensor
{
    $din = new Tensor($dout->shape);
    for ($i = 0; $i < $dout->size; $i++) {
        $din->buffer[$i] = (float) $dout->buffer[$i] * (float) $mask->buffer[$i];
    }
    return $din;
}

/**
 * Fused softmax cross-entropy.
 *
 * @param Tensor  $logits   [B, C]  raw scores from the linear head.
 * @param int[]   $targets  [B]     integer class labels.
 * @return array{Tensor, float}  [dLogits [B, C], mean_loss]
 */
function softmaxCE(Tensor $logits, array $targets): array
{
    [$B, $C] = $logits->shape;
    $dLogits = new Tensor([$B, $C]);
    $totalLoss = 0.0;

    for ($b = 0; $b < $B; $b++) {
        $off = $b * $C;

        // Numerically stable softmax
        $maxVal = -INF;
        for ($c = 0; $c < $C; $c++) {
            $v = (float) $logits->buffer[$off + $c];
            if ($v > $maxVal) $maxVal = $v;
        }
        $sum = 0.0;
        $expVals = [];
        for ($c = 0; $c < $C; $c++) {
            $e = exp((float) $logits->buffer[$off + $c] - $maxVal);
            $expVals[] = $e;
            $sum += $e;
        }

        $t = $targets[$b];
        $totalLoss -= log(max(1e-12, $expVals[$t] / $sum));

        // Gradient: softmax - one_hot(t), divided by B
        for ($c = 0; $c < $C; $c++) {
            $dLogits->buffer[$off + $c] = ($expVals[$c] / $sum - ($c === $t ? 1.0 : 0.0)) / $B;
        }
    }

    return [$dLogits, $totalLoss / $B];
}

/**
 * Full forward pass.  Returns [logits, all_caches].
 */
function lenetForward(
    Tensor $x,
    Conv2D $conv1, MaxPool2D $pool1,
    Conv2D $conv2, MaxPool2D $pool2,
    Tensor $W_fc,  Tensor $b_fc,
    int    $featDim,
): array {
    $blas = BlasEngine::get()->ffi;
    $B    = $x->shape[0];
    $C    = N_CLASSES;

    // ── Conv1 → ReLU1 → Pool1 ─────────────────────────────────────────────
    [$c1out, $c1cache]   = $conv1->forward($x);
    [$r1out, $r1mask]    = reluForward($c1out);
    [$p1out, $p1cache]   = $pool1->forward($r1out);

    // ── Conv2 → ReLU2 → Pool2 ─────────────────────────────────────────────
    [$c2out, $c2cache]   = $conv2->forward($p1out);
    [$r2out, $r2mask]    = reluForward($c2out);
    [$p2out, $p2cache]   = $pool2->forward($r2out);

    // ── Flatten: treat pool2 output as [B, featDim] ───────────────────────
    // p2out has shape [B, 8, 4, 4]; total B*128 elements contiguous.
    // We build a new [B, featDim] view by copying the buffer.
    $flat = new Tensor([$B, $featDim]);
    for ($i = 0; $i < $B * $featDim; $i++) {
        $flat->buffer[$i] = $p2out->buffer[$i];
    }

    // ── Linear: logits [B, C] = flat [B, featDim] @ W_fc^T [featDim, C] ──
    //   sgemm(RowMajor, NoTrans, Trans, B, C, featDim)
    $logits = new Tensor([$B, $C]);
    $blas->cblas_sgemm(
        101, 111, 112,
        $B, $C, $featDim,
        1.0, $flat->buffer,   $featDim,
             $W_fc->buffer,   $featDim,
        0.0, $logits->buffer, $C
    );
    // Add bias: logits[b, c] += b_fc[c]  (O(B·C), not a hot path)
    for ($bi = 0; $bi < $B; $bi++) {
        for ($c = 0; $c < $C; $c++) {
            $logits->buffer[$bi * $C + $c] =
                (float) $logits->buffer[$bi * $C + $c] + (float) $b_fc->buffer[$c];
        }
    }

    $caches = compact(
        'c1cache', 'r1mask', 'p1cache',
        'c2cache', 'r2mask', 'p2cache',
        'flat'
    );

    return [$logits, $caches];
}

/**
 * Full backward pass.  Updates weight->grad for all layers.
 *
 * @param Tensor $dLogits [B, C]  gradient from softmax CE.
 * @param array  $caches          from lenetForward().
 */
function lenetBackward(
    Tensor $dLogits,
    array  $caches,
    Conv2D $conv1, MaxPool2D $pool1,
    Conv2D $conv2, MaxPool2D $pool2,
    Tensor $W_fc,  Tensor $b_fc,
    int    $featDim,
): void {
    $blas = BlasEngine::get()->ffi;
    $B    = $dLogits->shape[0];
    $C    = N_CLASSES;

    $flat    = $caches['flat'];
    $r1mask  = $caches['r1mask'];
    $r2mask  = $caches['r2mask'];
    $c1cache = $caches['c1cache'];
    $c2cache = $caches['c2cache'];
    $p1cache = $caches['p1cache'];
    $p2cache = $caches['p2cache'];

    // ── Linear head backward ──────────────────────────────────────────────

    $W_fc->initGrad();
    $b_fc->initGrad();

    // dW_fc [C, featDim] += dLogits^T [C, B] @ flat [B, featDim]
    //   sgemm(RowMajor, Trans, NoTrans, C, featDim, B)
    $blas->cblas_sgemm(
        101, 112, 111,
        $C, $featDim, $B,
        1.0, $dLogits->buffer, $C,
             $flat->buffer,    $featDim,
        1.0, $W_fc->grad,      $featDim
    );

    // db_fc [C] += sum over batch of dLogits[b, :]
    //   sgemv(RowMajor, Trans, M=B, N=C, 1.0, dLogits [B×C], C, ones_B, 1, 1.0, db, 1)
    $onesB = Tensor::ones([$B]);
    $blas->cblas_sgemv(
        101, 112, $B, $C,
        1.0, $dLogits->buffer, $C, $onesB->buffer, 1,
        1.0, $b_fc->grad, 1
    );

    // d_flat [B, featDim] = dLogits [B, C] @ W_fc [C, featDim]
    //   sgemm(RowMajor, NoTrans, NoTrans, B, featDim, C)
    $dFlat = new Tensor([$B, $featDim]);
    $blas->cblas_sgemm(
        101, 111, 111,
        $B, $featDim, $C,
        1.0, $dLogits->buffer, $C,
             $W_fc->buffer,    $featDim,
        0.0, $dFlat->buffer,   $featDim
    );

    // ── Unflatten: d_flat → dPool2 [B, 8, 4, 4] ──────────────────────────
    $p2shape = $p2cache['B'] === $B
        ? [$B, 8, 4, 4]
        : throw new \RuntimeException('Batch size mismatch in backward');
    $dPool2 = new Tensor($p2shape);
    for ($i = 0; $i < $B * $featDim; $i++) {
        $dPool2->buffer[$i] = $dFlat->buffer[$i];
    }

    // ── Pool2 backward ────────────────────────────────────────────────────
    $dRelu2 = $pool2->backward($dPool2, $p2cache);

    // ── ReLU2 backward ────────────────────────────────────────────────────
    $dConv2 = reluBackward($dRelu2, $r2mask);

    // ── Conv2 backward ────────────────────────────────────────────────────
    $dPool1 = $conv2->backward($dConv2, $c2cache);

    // ── Pool1 backward ────────────────────────────────────────────────────
    $dRelu1 = $pool1->backward($dPool1, $p1cache);

    // ── ReLU1 backward ────────────────────────────────────────────────────
    $dConv1 = reluBackward($dRelu1, $r1mask);

    // ── Conv1 backward ────────────────────────────────────────────────────
    $conv1->backward($dConv1, $c1cache);
    // dinput is discarded (no gradient needed for input pixels)
}

// ─── 5. Accuracy helper ───────────────────────────────────────────────────

function evalAccuracy(
    Tensor $xTensor, array $yLabels,
    Conv2D $conv1, MaxPool2D $pool1,
    Conv2D $conv2, MaxPool2D $pool2,
    Tensor $W_fc, Tensor $b_fc,
    int $featDim,
): float {
    [$logits,] = lenetForward(
        $xTensor, $conv1, $pool1, $conv2, $pool2, $W_fc, $b_fc, $featDim
    );
    $B       = $logits->shape[0];
    $C       = N_CLASSES;
    $correct = 0;
    for ($b = 0; $b < $B; $b++) {
        $off  = $b * $C;
        $best = 0;
        $bestV = (float) $logits->buffer[$off];
        for ($c = 1; $c < $C; $c++) {
            $v = (float) $logits->buffer[$off + $c];
            if ($v > $bestV) { $bestV = $v; $best = $c; }
        }
        if ($best === $yLabels[$b]) $correct++;
    }
    return $correct / $B;
}

// ─── 6. Print header ──────────────────────────────────────────────────────

echo "\n";
echo "════════════════════════════════════════════════════════════\n";
echo "  LeNet-style CNN — Synthetic 16×16 Image Classifier\n";
echo sprintf(
    "  Train: %d   Test: %d   Classes: %d   Epochs: %d\n",
    N_TRAIN, N_TEST, N_CLASSES, N_EPOCHS
);
echo "  Architecture: Conv(1→4) → ReLU → Pool → Conv(4→8) → ReLU → Pool → Linear(128→10)\n";
echo "════════════════════════════════════════════════════════════\n\n";
echo sprintf("  %-6s  %-12s  %-12s\n", 'Epoch', 'Train Loss', 'Test Acc');
echo "  " . str_repeat('─', 34) . "\n";

// ─── 7. Training loop ─────────────────────────────────────────────────────

$nBatches = (int) ceil(N_TRAIN / BATCH);

for ($epoch = 1; $epoch <= N_EPOCHS; $epoch++) {
    // Shuffle
    $idx = range(0, N_TRAIN - 1);
    shuffle($idx);

    $epochLoss   = 0.0;
    $batchesSeen = 0;

    for ($b = 0; $b < $nBatches; $b++) {
        $start = $b * BATCH;
        $end   = min($start + BATCH, N_TRAIN);

        $batchImgs   = [];
        $batchLabels = [];
        for ($i = $start; $i < $end; $i++) {
            $batchImgs[]   = $trainX[$idx[$i]];
            $batchLabels[] = $trainY[$idx[$i]];
        }
        $bX = packImages($batchImgs);

        // ── Zero gradients ────────────────────────────────────────────────
        $conv1->zeroGrad();
        $conv2->zeroGrad();
        $W_fc->zeroGrad();
        $b_fc->zeroGrad();

        // ── Forward ───────────────────────────────────────────────────────
        [$logits, $caches] = lenetForward(
            $bX, $conv1, $pool1, $conv2, $pool2, $W_fc, $b_fc, $featDim
        );

        // ── Loss + gradient ───────────────────────────────────────────────
        [$dLogits, $loss] = softmaxCE($logits, $batchLabels);
        $epochLoss += $loss;
        $batchesSeen++;

        // ── Backward ──────────────────────────────────────────────────────
        lenetBackward(
            $dLogits, $caches,
            $conv1, $pool1, $conv2, $pool2, $W_fc, $b_fc, $featDim
        );

        // ── Optimizer step ────────────────────────────────────────────────
        $optimizer->step();
    }

    $avgLoss = $epochLoss / $batchesSeen;
    $testAcc = evalAccuracy(
        $testTensor, $testY,
        $conv1, $pool1, $conv2, $pool2, $W_fc, $b_fc, $featDim
    );

    echo sprintf("  %-6d  %-12.4f  %-12.4f\n", $epoch, $avgLoss, $testAcc);
}

echo "  " . str_repeat('─', 34) . "\n\n";

// ─── 8. Confusion matrix ──────────────────────────────────────────────────

echo "── Confusion Matrix (test set) ──\n\n";

[$testLogits,] = lenetForward(
    $testTensor, $conv1, $pool1, $conv2, $pool2, $W_fc, $b_fc, $featDim
);

$C          = N_CLASSES;
$confMatrix = array_fill(0, $C, array_fill(0, $C, 0));

for ($i = 0; $i < N_TEST; $i++) {
    $off  = $i * $C;
    $pred = 0;
    $bestV = (float) $testLogits->buffer[$off];
    for ($c = 1; $c < $C; $c++) {
        $v = (float) $testLogits->buffer[$off + $c];
        if ($v > $bestV) { $bestV = $v; $pred = $c; }
    }
    $confMatrix[$testY[$i]][$pred]++;
}

// Header
echo "       Pred→  ";
for ($c = 0; $c < $C; $c++) echo sprintf("%4d", $c);
echo "\n  " . str_repeat('─', 4 + $C * 4) . "\n";

for ($r = 0; $r < $C; $r++) {
    echo sprintf("  Act:%-3d  ", $r);
    for ($c = 0; $c < $C; $c++) {
        echo sprintf("%4d", $confMatrix[$r][$c]);
    }
    echo "\n";
}

// Per-class F1
echo "\n  Class  Precision   Recall      F1\n";
echo "  " . str_repeat('─', 40) . "\n";

$macroF1 = 0.0;
for ($c = 0; $c < $C; $c++) {
    $tp = $confMatrix[$c][$c];
    $fp = 0; $fn = 0;
    for ($r = 0; $r < $C; $r++) {
        if ($r !== $c) $fp += $confMatrix[$r][$c];
        if ($r !== $c) $fn += $confMatrix[$c][$r];
    }
    $prec  = $tp + $fp > 0 ? $tp / ($tp + $fp)  : 0.0;
    $rec   = $tp + $fn > 0 ? $tp / ($tp + $fn)  : 0.0;
    $f1    = $prec + $rec > 0 ? 2 * $prec * $rec / ($prec + $rec) : 0.0;
    $macroF1 += $f1;
    echo sprintf("  %-5d  %9.3f   %9.3f   %.3f\n", $c, $prec, $rec, $f1);
}

$correct = 0;
for ($c = 0; $c < $C; $c++) $correct += $confMatrix[$c][$c];
$total   = N_TEST;

echo "\n  Macro F1         : " . sprintf("%.3f", $macroF1 / $C) . "\n";
echo "  Overall accuracy : " . sprintf("%.2f%%", 100.0 * $correct / $total)
    . " ({$correct}/{$total})\n";

echo "\n════════════════════════════════════════════════════════════\n";
echo "  Done.\n";
echo "════════════════════════════════════════════════════════════\n\n";
