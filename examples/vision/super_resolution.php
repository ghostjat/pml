<?php
declare(strict_types=1);
/**
 * SUPER-RESOLUTION + IMAGE RESIZING UTILITIES
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Enhance low-resolution medical scans:
 *              • Bilinear resize  (up / down) — general utility
 *              • Nearest-neighbour resize     — fast, pixelated
 *              • Residual SRCNN              — learned 4× upscale
 *
 *            Task: 8×8 LR scan → 32×32 HR reconstruction.
 *
 * Method   :
 *   1. Utilities  — pure-PHP bilinear & nearest-neighbour resize.
 *   2. Residual SRCNN — three Conv2D layers trained with residual MSE.
 *
 *            Key insight: train the CNN to predict the HIGH-FREQUENCY
 *            RESIDUAL (HR − bicubic_LR) rather than HR directly.
 *            This lets the network focus on recovering missing edge
 *            detail rather than reconstructing the whole image, which
 *            is both easier to learn and more stable under MSE.
 *
 *   Architecture:
 *     Reshape([1,32,32])
 *     → Conv2D( 1→16, k=5, p=2) → ReLU   feature extraction
 *     → Conv2D(16→8,  k=3, p=1) → ReLU   edge mapping
 *     → Conv2D( 8→1,  k=3, p=1)           residual reconstruction
 *     → Flatten → Sigmoid → [N, 1024]    (offset residual, ∈ [0,1])
 *
 *            Final HR = bicubic_upsampled + (network_output − 0.5)
 *
 * Metrics  : PSNR = −10 log₁₀(MSE)  [dB].  Higher = better.
 *            Each +3 dB halves the mean squared error.
 *
 * Business : Pathology scanners capture 4× lower resolution to cut
 *            scan time from 6 min → 90 sec.  SRCNN recovers
 *            diagnostic-quality detail in real time with a <2 ms
 *            per-frame GPU inference budget.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Reshape;
use Pml\NeuralNetwork\Layers\Conv2D;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Layers\Flatten;
use Pml\NeuralNetwork\Layers\Sigmoid;
use Pml\NeuralNetwork\Optimizers\Adam;
use Pml\Losses\MeanSquaredError;

section('Super-Resolution + Image Resizing');

const SR_H   = 32;
const SR_W   = 32;
const SR_PIX = SR_H * SR_W;
const SCALE  = 4;
const LR_H   = SR_H / SCALE;
const LR_W   = SR_W / SCALE;
const RES_OFF = 0.5;    // residual offset so output is centred in Sigmoid range

mt_srand(31);
$rng   = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);
$randn = fn() => sqrt(-2.0 * log(max(1e-10, mt_rand() / mt_getrandmax())))
                 * cos(2.0 * M_PI * (mt_rand() / mt_getrandmax()));

// ── 1. Resize utilities (pure PHP) ────────────────────────────────────────────

/**
 * Bilinear interpolation resize.
 * Each output pixel is the weighted average of its 4 nearest source neighbours.
 */
function bilinearResize(array $src, int $srcH, int $srcW,
                         int $dstH, int $dstW): array
{
    $dst = [];
    for ($r = 0; $r < $dstH; $r++) {
        for ($c = 0; $c < $dstW; $c++) {
            $sr  = $r * ($srcH - 1) / max(1, $dstH - 1);
            $sc  = $c * ($srcW - 1) / max(1, $dstW - 1);
            $r0  = (int)floor($sr);   $r1 = min($r0 + 1, $srcH - 1);
            $c0  = (int)floor($sc);   $c1 = min($c0 + 1, $srcW - 1);
            $dr  = $sr - $r0;         $dc = $sc - $c0;
            $dst[] = (1-$dr)*(1-$dc) * $src[$r0*$srcW+$c0]
                   + (1-$dr)*$dc     * $src[$r0*$srcW+$c1]
                   +    $dr *(1-$dc) * $src[$r1*$srcW+$c0]
                   +    $dr *$dc     * $src[$r1*$srcW+$c1];
        }
    }
    return $dst;
}

/**
 * Nearest-neighbour resize — fast but introduces blocking artefacts.
 */
function nearestResize(array $src, int $srcH, int $srcW,
                        int $dstH, int $dstW): array
{
    $dst = [];
    for ($r = 0; $r < $dstH; $r++) {
        for ($c = 0; $c < $dstW; $c++) {
            $sr    = (int)round($r * ($srcH - 1) / max(1, $dstH - 1));
            $sc    = (int)round($c * ($srcW - 1) / max(1, $dstW - 1));
            $dst[] = $src[min($sr, $srcH-1) * $srcW + min($sc, $srcW-1)];
        }
    }
    return $dst;
}

/**
 * Anti-aliased downscale via average pooling.
 * Requires $factor to divide $srcH and $srcW exactly.
 */
function avgDownscale(array $src, int $srcH, int $srcW, int $factor): array
{
    $dstH = intdiv($srcH, $factor);
    $dstW = intdiv($srcW, $factor);
    $area = (float)($factor * $factor);
    $dst  = array_fill(0, $dstH * $dstW, 0.0);
    for ($r = 0; $r < $srcH; $r++) {
        for ($c = 0; $c < $srcW; $c++) {
            $dst[intdiv($r, $factor) * $dstW + intdiv($c, $factor)]
                += $src[$r * $srcW + $c] / $area;
        }
    }
    return $dst;
}

/** PSNR in dB; pixel range [0,1]. */
function psnr(array $pred, array $gt): float
{
    $mse = 0.0; $n = count($pred);
    for ($i = 0; $i < $n; $i++) $mse += ($pred[$i] - $gt[$i]) ** 2;
    $mse /= $n;
    return $mse > 1e-12 ? -10.0 * log10($mse) : 99.0;
}

function asciiRow(array $px, int $w, int $r, string $chars): string
{
    $out = ''; $n = strlen($chars);
    for ($c = 0; $c < $w; $c++) {
        $v    = max(0.0, min(0.9999, $px[$r * $w + $c]));
        $out .= $chars[(int)($v * $n)];
    }
    return $out;
}

// ── 2. Demonstrate resize utilities on a 16×16 test pattern ─────────────────
section('Resize Utilities — Bilinear vs Nearest-Neighbour');

$demoH = 16; $demoW = 16;
$demoPx = [];
for ($r = 0; $r < $demoH; $r++) {
    for ($c = 0; $c < $demoW; $c++) {
        $dist    = sqrt(($r - 8.0)**2 + ($c - 8.0)**2);
        $blob    = exp(-$dist**2 / 18.0);
        $stripe  = ($r % 4 < 2) ? 0.3 : 0.0;
        $demoPx[] = min(1.0, $blob * 0.8 + $stripe + ($c / $demoW) * 0.08);
    }
}

$lrDemo = avgDownscale($demoPx, $demoH, $demoW, 4);
$bilUp  = bilinearResize($lrDemo, 4, 4, $demoH, $demoW);
$nnUp   = nearestResize($lrDemo,  4, 4, $demoH, $demoW);

printf("  %-20s  PSNR vs 16×16\n", 'Method');
printf("  %s\n", str_repeat('-', 38));
printf("  %-20s  %.1f dB\n", 'Bilinear ↑4×', psnr($bilUp,  $demoPx));
printf("  %-20s  %.1f dB\n", 'Nearest  ↑4×', psnr($nnUp,   $demoPx));

$shade = ' .:=+*#%@';
printf("\n  Original (16×16)     | Bilinear ↑4×         | Nearest ↑4×\n");
printf("  %s\n", str_repeat('-', 68));
for ($r = 0; $r < $demoH; $r++) {
    printf("  %s  |  %s  |  %s\n",
        asciiRow($demoPx, $demoW, $r, $shade),
        asciiRow($bilUp,  $demoW, $r, $shade),
        asciiRow($nnUp,   $demoW, $r, $shade));
}
printf("  Bilinear is smooth; nearest shows %d×%d blocks (each input pixel → 4×4 block).\n",
       SCALE, SCALE);

// ── 3. Synthetic HR image generator ──────────────────────────────────────────
/**
 * Generates a 32×32 scan with one of three tissue types:
 *   0 = Smooth lobe    : Gaussian with sharp border (organ boundary)
 *   1 = Sharp nodule   : Circle + hard edge (high-contrast lesion)
 *   2 = Fine structure : Cross-hatch pattern (bone trabeculae / vessels)
 * Low pixel noise (σ=0.01) to avoid swamping structural PSNR.
 */
function makeScan(callable $randn): array
{
    $type = mt_rand(0, 2);
    // Randomise centre & size — fixed per call so no loop-level mt_rand()
    $cx  = mt_rand(10, 22);
    $cy  = mt_rand(10, 22);
    $rad = mt_rand(5, 8);

    $px = [];
    for ($r = 0; $r < SR_H; $r++) {
        for ($c = 0; $c < SR_W; $c++) {
            $bg = 0.25 + $randn() * 0.01;

            $v = match ($type) {
                0 => $bg + 0.62 * exp(-0.5 * (($r-$cy)**2 + ($c-$cx)**2) / (7.5**2)),
                1 => $bg + (sqrt(($r-$cy)**2 + ($c-$cx)**2) < $rad ? 0.65 : 0.0),
                2 => $bg + (($r % 5 < 2 || $c % 5 < 2) ? 0.55 : 0.0),
                default => $bg,
            };
            $px[] = min(1.0, max(0.0, $v));
        }
    }
    return $px;
}

// ── 4. Build residual-SRCNN dataset ──────────────────────────────────────────
section('Building Residual-SRCNN Dataset');
$t0 = microtime(true);

$nImages   = 1000;
$hrImages  = [];   // [N, 1024] original HR — kept for final PSNR eval
$bilImages = [];   // [N, 1024] bicubic-upsampled LR (network input)
$residuals = [];   // [N, 1024] HR − bicubic + RES_OFF  (network target)

for ($i = 0; $i < $nImages; $i++) {
    $hr  = makeScan($randn);
    $lr  = avgDownscale($hr, SR_H, SR_W, SCALE);
    $bil = bilinearResize($lr, LR_H, LR_W, SR_H, SR_W);
    // Residual: shift by +0.5 so range straddles 0.5 (good for Sigmoid)
    $res = array_map(fn($h, $b) => min(1.0, max(0.0, $h - $b + RES_OFF)), $hr, $bil);
    $hrImages[]  = $hr;
    $bilImages[] = $bil;
    $residuals[] = $res;
}

$split   = (int)($nImages * 0.8);
$trainDs = Dataset::fromArray(
    array_slice($bilImages, 0, $split),   // input:  blurry bicubic
    array_slice($residuals, 0, $split)    // target: HR − bicubic + 0.5
);
$testDs  = Dataset::fromArray(
    array_slice($bilImages, $split),
    array_slice($residuals, $split)
);

// Bicubic baseline PSNR (comparing bicubic to HR, no CNN)
$bicubicPsnr = 0.0;
for ($i = $split; $i < $nImages; $i++) {
    $bicubicPsnr += psnr($bilImages[$i], $hrImages[$i]);
}
$bicubicPsnr /= ($nImages - $split);

metric('Training pairs', $trainDs->numRows());
metric('Test pairs',     $testDs->numRows());
metric('HR size',        SR_H . '×' . SR_W . '  (' . SR_PIX . ' px)');
metric('LR size',        LR_H . '×' . LR_W . '  (4× downscale)');
metric('Bicubic PSNR',   round($bicubicPsnr, 2) . ' dB  (baseline to beat)');
metric('Dataset time',   elapsed($t0));

// ── 5. Residual SRCNN architecture ───────────────────────────────────────────
section('Building Residual SRCNN');

$srcnn = new Sequential(
    layers: [
        new Reshape([1, SR_H, SR_W]),                              // [B,1,32,32]
        new Conv2D(inChannels:  1, outChannels: 16, kernelSize: 5,
                   stride: 1, padding: 2),                         // [B,16,32,32]
        new ReLU(),
        new Conv2D(inChannels: 16, outChannels:  8, kernelSize: 3,
                   stride: 1, padding: 1),                         // [B, 8,32,32]
        new ReLU(),
        new Conv2D(inChannels:  8, outChannels:  1, kernelSize: 3,
                   stride: 1, padding: 1),                         // [B, 1,32,32]
        new Flatten(),                                              // [B, 1024]
        new Sigmoid(),                                             // residual ∈ [0,1]
    ],
    lossFn:    new MeanSquaredError(),
    optimizer: new Adam(learningRate: 2e-3),
);

printf("  Input  : bicubic LR [B,%d] → Reshape → [B,1,%d,%d]\n", SR_PIX, SR_H, SR_W);
printf("  Stage 1: Conv2D(1→16, k=5, p=2) + ReLU   [feature extraction]\n");
printf("  Stage 2: Conv2D(16→8, k=3, p=1) + ReLU   [edge mapping]\n");
printf("  Stage 3: Conv2D(8→1,  k=3, p=1) + Flatten + Sigmoid  [residual output]\n");
printf("  Target : HR − bicubic + 0.5  (offset residual ∈ [0,1])\n");
printf("  Post-process : predicted_HR = bicubic + output − 0.5\n");

// ── 6. Train ──────────────────────────────────────────────────────────────────
section('Training Residual SRCNN');
$t0 = microtime(true);
$srcnn->train($trainDs, epochs: 40, batchSize: 32, validation: $testDs, patience: 6);
metric('Training time', elapsed($t0));

// ── 7. Evaluate PSNR ─────────────────────────────────────────────────────────
section('Evaluation — PSNR Improvement');

$srcnnPsnr  = 0.0;
$nearestPsnr = 0.0;
$nTest       = $nImages - $split;

for ($i = $split; $i < $nImages; $i++) {
    $ds      = Dataset::fromArray([$bilImages[$i]]);
    $resPred = $srcnn->predict($ds)->toFlatArray();
    // Reconstruct: bicubic + predicted_residual − RES_OFF
    $hrPred  = array_map(fn($r, $b) => min(1.0, max(0.0, $r - RES_OFF + $b)),
                          $resPred, $bilImages[$i]);
    $srcnnPsnr += psnr($hrPred, $hrImages[$i]);

    // Nearest-neighbour baseline
    $lr  = avgDownscale($hrImages[$i], SR_H, SR_W, SCALE);
    $nn  = nearestResize($lr, LR_H, LR_W, SR_H, SR_W);
    $nearestPsnr += psnr($nn, $hrImages[$i]);
}
$srcnnPsnr   /= $nTest;
$nearestPsnr /= $nTest;

printf("  %-22s  %6.2f dB\n", 'Nearest-neighbour ↑4×', $nearestPsnr);
printf("  %-22s  %6.2f dB  (baseline)\n", 'Bicubic ↑4×', $bicubicPsnr);
printf("  %-22s  %6.2f dB  (+%.1f dB vs bicubic)\n",
       'Residual SRCNN ↑4×', $srcnnPsnr, $srcnnPsnr - $bicubicPsnr);

// ── 8. Visual comparison ──────────────────────────────────────────────────────
section('Visual Comparison — 1 Test Scan');

mt_srand(77);   // reproducible demo image
$hrGt  = makeScan($randn);
$lr    = avgDownscale($hrGt, SR_H, SR_W, SCALE);
$bilUp = bilinearResize($lr, LR_H, LR_W, SR_H, SR_W);
$nnUp  = nearestResize($lr,  LR_H, LR_W, SR_H, SR_W);

$ds      = Dataset::fromArray([$bilUp]);
$resPred = $srcnn->predict($ds)->toFlatArray();
$srcnnHr = array_map(fn($r, $b) => min(1.0, max(0.0, $r - RES_OFF + $b)),
                      $resPred, $bilUp);

$pBil   = psnr($bilUp,  $hrGt);
$pNN    = psnr($nnUp,   $hrGt);
$pSR    = psnr($srcnnHr, $hrGt);

printf("\n  PSNR — Nearest: %.1f dB  |  Bilinear: %.1f dB  |  Residual SRCNN: %.1f dB\n\n",
       $pNN, $pBil, $pSR);

$shade2 = ' .,:;|+=*?#@%&';
printf("  HR (ground truth)               |  Bilinear ↑4×                  |  Residual SRCNN\n");
printf("  %s\n", str_repeat('-', 101));
for ($r = 0; $r < SR_H; $r++) {
    printf("  %s  |  %s  |  %s\n",
        asciiRow($hrGt,   SR_W, $r, $shade2),
        asciiRow($bilUp,  SR_W, $r, $shade2),
        asciiRow($srcnnHr,SR_W, $r, $shade2));
}

// ── 9. 8×8 raw sensor view ───────────────────────────────────────────────────
section('8×8 Raw Sensor Input');
printf("  (Each cell = 1 LR pixel; 4×4 HR pixels are averaged into it)\n\n");
for ($r = 0; $r < LR_H; $r++) {
    printf("  ");
    for ($c = 0; $c < LR_W; $c++) {
        $v = max(0.0, min(0.9999, $lr[$r * LR_W + $c]));
        printf("%s ", $shade2[(int)($v * strlen($shade2))]);
    }
    echo "\n";
}

// ── 10. Resolution downgrade quality ladder ───────────────────────────────────
section('Resolution Downgrade Quality Ladder — PSNR after bilinear upsample');

printf("  %-12s | %-8s | PSNR vs %dx%d\n", 'Target size', 'Factor', SR_H, SR_W);
printf("  %s\n", str_repeat('-', 42));
foreach ([2, 4, 8] as $f) {
    if (SR_H % $f !== 0) continue;
    $h2 = intdiv(SR_H, $f); $w2 = intdiv(SR_W, $f);
    $lr2 = avgDownscale($hrGt, SR_H, SR_W, $f);
    $up  = bilinearResize($lr2, $h2, $w2, SR_H, SR_W);
    printf("  %-12s | %-8s | %.1f dB\n", $h2.'x'.$w2, $f.'x', psnr($up, $hrGt));
}

echo "\n✓ Done\n";
