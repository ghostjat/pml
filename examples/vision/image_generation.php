<?php
declare(strict_types=1);
/**
 * AI IMAGE STUDIO — Text-to-Image Synthesis
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Generate 20×20 grayscale images from text prompts.
 *            5 visual styles: sunset, aurora, halo, grid, diagonal.
 *
 * Method   : Conditional Generator MLP
 *              Input : 5-dim one-hot prompt embedding
 *              Hidden: Dense(5→128) → ReLU → Dense(128→256) → ReLU
 *              Output: Dense(256→400) → Sigmoid → 400 pixel values
 *            Loss   : MSE — generator learns the per-style prototype.
 *
 *            Text-to-image pipeline:
 *              1. Tokenize prompt → keyword match → style index
 *              2. One-hot encode → 5-dim
 *              3. Forward through generator → 400 pixels (20×20)
 *              4. Render as ASCII art
 *
 * Business : E-commerce product rendering: "warm sunset gradient",
 *            "radiant spotlight", "clean grid pattern" → instant
 *            background mockup at 60 fps on commodity hardware.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Tensor;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Layers\Dropout;
use Pml\NeuralNetwork\Layers\Sigmoid;
use Pml\NeuralNetwork\Optimizers\Adam;
use Pml\Losses\MeanSquaredError;

section('AI Image Studio — Text-to-Image Synthesis');

const GEN_H      = 20;
const GEN_W      = 20;
const GEN_PIXELS = GEN_H * GEN_W;   // 400
const N_STYLES   = 5;

mt_srand(99);
$rng   = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);
$randn = fn() => sqrt(-2.0 * log(max(1e-10, mt_rand() / mt_getrandmax())))
                 * cos(2.0 * M_PI * (mt_rand() / mt_getrandmax()));

// ── 1. Style prototype generators ────────────────────────────────────────────
/**
 * Five distinct visual styles — each produces a consistent spatial structure
 * so the generator MLP can learn a crisp per-style prototype.
 *
 *   0 = sunset   : vertical gradient, bright at top fading to dark
 *   1 = aurora   : sinusoidal horizontal wave bands
 *   2 = halo     : radial Gaussian glow from image centre
 *   3 = grid     : bright squares on dark background (checkerboard-like)
 *   4 = diagonal : diagonal gradient, bright top-left → dark bottom-right
 */
$styleNames = ['sunset', 'aurora', 'halo', 'grid', 'diagonal'];

function makeStyleImage(int $style, callable $randn): array
{
    $px = [];
    $cx = GEN_W / 2.0;
    $cy = GEN_H / 2.0;

    for ($r = 0; $r < GEN_H; $r++) {
        for ($c = 0; $c < GEN_W; $c++) {
            $noise = $randn() * 0.025;   // small texture noise

            $v = match ($style) {
                0 => 1.0 - ($r / (GEN_H - 1)),                // sunset: top=1, bottom=0
                1 => 0.5 + 0.45 * sin(($r / GEN_H) * M_PI * 4 // aurora: 4 wave cycles
                         + ($c / GEN_W) * M_PI),
                2 => exp(-0.5 * ((($r-$cy)**2 + ($c-$cx)**2)  // halo: σ=7
                         / (7.0 ** 2))),
                3 => ((int)($r / 4) + (int)($c / 4)) % 2      // grid: 4px cells
                         === 0 ? 0.85 : 0.15,
                4 => 1.0 - ($r + $c) / (GEN_H + GEN_W - 2.0), // diagonal: TL→BR
                default => 0.5,
            };

            $px[] = min(1.0, max(0.0, $v + $noise));
        }
    }
    return $px;
}

// ── 2. Prompt keyword dictionary ──────────────────────────────────────────────
$keywords = [
    0 => ['sunset', 'warm', 'sky', 'dusk', 'twilight', 'horizon', 'warm gradient'],
    1 => ['aurora', 'wave', 'ripple', 'northern lights', 'shimmer', 'flow', 'bands'],
    2 => ['halo', 'glow', 'spotlight', 'radiant', 'vignette', 'bright center', 'bloom'],
    3 => ['grid', 'pattern', 'mesh', 'geometric', 'tiles', 'checkerboard', 'squares'],
    4 => ['diagonal', 'slant', 'angle', 'oblique', 'fade', 'tilt', 'cross gradient'],
];

function resolvePrompt(string $prompt, array $keywords): int
{
    $prompt = strtolower(trim($prompt));
    foreach ($keywords as $style => $terms) {
        foreach ($terms as $kw) {
            if (str_contains($prompt, $kw)) return $style;
        }
    }
    return -1;  // unknown
}

// ── 3. Generate training dataset ──────────────────────────────────────────────
section('Generating Training Dataset');

$nPerStyle = 500;
$inputRows  = [];   // [N, N_STYLES] one-hot
$targetRows = [];   // [N, GEN_PIXELS] pixel arrays

for ($style = 0; $style < N_STYLES; $style++) {
    $oneHot = array_fill(0, N_STYLES, 0.0);
    $oneHot[$style] = 1.0;
    for ($i = 0; $i < $nPerStyle; $i++) {
        $inputRows[]  = $oneHot;
        $targetRows[] = makeStyleImage($style, $randn);
    }
}

// Shuffle
$n   = count($inputRows);
$idx = range(0, $n - 1);
shuffle($idx);
$inputRows  = array_map(fn($i) => $inputRows[$i],  $idx);
$targetRows = array_map(fn($i) => $targetRows[$i], $idx);

$split   = (int)($n * 0.85);
$trainDs = Dataset::fromArray(array_slice($inputRows, 0, $split),  array_slice($targetRows, 0, $split));
$testDs  = Dataset::fromArray(array_slice($inputRows, $split),     array_slice($targetRows, $split));

metric('Styles',           implode(', ', $styleNames));
metric('Training samples', $trainDs->numRows());
metric('Test samples',     $testDs->numRows());
metric('Input dim',        N_STYLES . ' (one-hot style embedding)');
metric('Output dim',       GEN_PIXELS . ' pixel values (20×20)');

// ── 4. Conditional Generator MLP ─────────────────────────────────────────────
section('Building Conditional Generator');

$generator = new Sequential(
    layers: [
        new Dense(N_STYLES, 128),
        new ReLU(),
        new Dense(128, 256),
        new ReLU(),
        new Dropout(0.05),
        new Dense(256, GEN_PIXELS),
        new Sigmoid(),
    ],
    lossFn:    new MeanSquaredError(),
    optimizer: new Adam(learningRate: 2e-3),
);

printf("  Input  : %d-dim one-hot prompt embedding\n", N_STYLES);
printf("  Hidden : Dense(128) → ReLU → Dense(256) → ReLU\n");
printf("  Output : Dense(%d) → Sigmoid  (pixel values 0–1)\n", GEN_PIXELS);

// ── 5. Train ──────────────────────────────────────────────────────────────────
section('Training Generator');
$t0 = microtime(true);

$generator->train($trainDs, epochs: 80, batchSize: 64, validation: $testDs, patience: 10);
metric('Training time', elapsed($t0));

// ── 6. Evaluate reconstruction quality ───────────────────────────────────────
section('Reconstruction Quality (MSE per style)');

foreach ($styleNames as $si => $name) {
    $oneHot = array_fill(0, N_STYLES, 0.0);
    $oneHot[$si] = 1.0;

    // Build 20-sample test batch for this style
    $batchIn  = []; $batchTgt = [];
    for ($i = 0; $i < 20; $i++) {
        $batchIn[]  = $oneHot;
        $batchTgt[] = makeStyleImage($si, $randn);
    }
    $batchDs  = Dataset::fromArray($batchIn, $batchTgt);
    $predTens = $generator->predict($batchDs);   // [20, 400]
    $labTens  = $batchDs->labels();              // [20, 400]
    $mse      = $predTens->sub($labTens)->square()->mean();
    $psnr     = $mse > 0 ? -10.0 * log10($mse) : 99.0;
    printf("  %-10s : MSE = %.5f   PSNR = %.1f dB\n", $name, $mse, $psnr);
}

// ── 7. ASCII art gallery ──────────────────────────────────────────────────────
section('Style Gallery — Ground Truth vs Generated');

$shadeChars = ' .,:;-=+*?#@%&';  // 14 levels

function renderAscii(array $px, int $w, int $h, string $chars): string
{
    $n   = strlen($chars);
    $out = '';
    for ($r = 0; $r < $h; $r++) {
        for ($c = 0; $c < $w; $c++) {
            $v    = max(0.0, min(0.9999, $px[$r * $w + $c]));
            $out .= $chars[(int)($v * $n)];
        }
        $out .= "\n";
    }
    return $out;
}

foreach ($styleNames as $si => $name) {
    $oneHot = array_fill(0, N_STYLES, 0.0);
    $oneHot[$si] = 1.0;

    // Ground truth prototype (zero noise for clarity)
    $gtPx   = makeStyleImage($si, fn() => 0.0);

    // Generated image
    $genDs  = Dataset::fromArray([$oneHot]);
    $genPx  = $generator->predict($genDs)->toFlatArray();   // [400]

    // Side-by-side display
    $gtLines  = explode("\n", rtrim(renderAscii($gtPx,  GEN_W, GEN_H, $shadeChars)));
    $genLines = explode("\n", rtrim(renderAscii($genPx, GEN_W, GEN_H, $shadeChars)));

    printf("\n  Style: %-10s  Ground Truth %s  Generated\n",
           strtoupper($name), str_repeat(' ', GEN_W - 10));
    printf("  %s  %s\n", str_repeat('-', GEN_W), str_repeat('-', GEN_W));

    for ($r = 0; $r < GEN_H; $r++) {
        printf("  %s  %s\n", $gtLines[$r] ?? '', $genLines[$r] ?? '');
    }
}

// ── 8. Text prompt interface ──────────────────────────────────────────────────
section('Text Prompt Interface');

$prompts = [
    'warm twilight sky',
    'glowing spotlight in dark room',
    'tile checkerboard pattern',
    'northern lights aurora',
    'diagonal fade gradient',
    'luminous bloom effect',
    'geometric mesh design',
    'dusk over the horizon',
];

printf("  %-35s | %-12s | %s\n", 'Prompt', 'Style', 'Thumbnail (first row)');
printf("  %s\n", str_repeat('-', 80));

foreach ($prompts as $prompt) {
    $si = resolvePrompt($prompt, $keywords);
    if ($si < 0) {
        printf("  %-35s | %-12s | (unknown prompt)\n", $prompt, '?');
        continue;
    }
    $oneHot = array_fill(0, N_STYLES, 0.0);
    $oneHot[$si] = 1.0;

    $genDs = Dataset::fromArray([$oneHot]);
    $px    = $generator->predict($genDs)->toFlatArray();

    // Show first row as thumbnail
    $thumb = '';
    for ($c = 0; $c < GEN_W; $c++) {
        $v      = max(0.0, min(0.9999, $px[$c]));
        $thumb .= $shadeChars[(int)($v * strlen($shadeChars))];
    }
    printf("  %-35s | %-12s | %s\n", $prompt, $styleNames[$si], $thumb);
}

echo "\n✓ Done\n";
