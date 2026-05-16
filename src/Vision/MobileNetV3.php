<?php

declare(strict_types=1);

namespace Pml\Vision;

use Pml\Lib\VisionEngine;
use Pml\NeuralNetwork\Layers\BatchNorm2D;
use Pml\NeuralNetwork\Layers\Conv2D;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\DepthwiseConv2D;
use Pml\NeuralNetwork\Layers\GlobalAveragePooling2D;
use Pml\NeuralNetwork\Layers\HardSwish;
use Pml\NeuralNetwork\Layers\InvertedResidual;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Layers\Softmax;
use Pml\NeuralNetwork\Optimizers\Adam;
use Pml\Tensor;

/**
 * MobileNetV3 — efficient CNN backbone (Howard et al., 2019).
 *
 * Variants:
 *   'small' — 2.9M params, 56 MFlops  @ 224×224
 *   'large' — 5.4M params, 219 MFlops @ 224×224
 *
 * Architecture:
 *   Stem Conv → 11/16 InvertedResidual blocks (with SE + HardSwish) → Classifier head
 *
 * Usage:
 *   $net = new MobileNetV3('small', numClasses: 10, inputSize: 224);
 *   $net->train($trainImages, $trainLabels, epochs: 30);
 *   $probs = $net->classify($image);
 *   $feat  = $net->extract($image);   // 576-D or 960-D feature vector
 *
 * Zero PHP arithmetic — all ops via TensorEngine / VisionEngine C.
 */
final class MobileNetV3
{
    private const SMALL_CFG = [
        // [inC, expandC, outC, kernel, stride, act, useSE]
        [ 16,  16,  16, 3, 2, 'relu', true ],
        [ 16,  72,  24, 3, 2, 'relu', false],
        [ 24,  88,  24, 3, 1, 'relu', false],
        [ 24,  96,  40, 5, 2, 'hs',   true ],
        [ 40, 240,  40, 5, 1, 'hs',   true ],
        [ 40, 240,  40, 5, 1, 'hs',   true ],
        [ 40, 120,  48, 5, 1, 'hs',   true ],
        [ 48, 144,  48, 5, 1, 'hs',   true ],
        [ 48, 288,  96, 5, 2, 'hs',   true ],
        [ 96, 576,  96, 5, 1, 'hs',   true ],
        [ 96, 576,  96, 5, 1, 'hs',   true ],
    ];
    private const LARGE_CFG = [
        [ 16,  16,  16, 3, 1, 'relu', false],
        [ 16,  64,  24, 3, 2, 'relu', false],
        [ 24,  72,  24, 3, 1, 'relu', false],
        [ 24,  72,  40, 5, 2, 'relu', true ],
        [ 40, 120,  40, 5, 1, 'relu', true ],
        [ 40, 120,  40, 5, 1, 'relu', true ],
        [ 40, 240,  80, 3, 2, 'hs',   false],
        [ 80, 200,  80, 3, 1, 'hs',   false],
        [ 80, 184,  80, 3, 1, 'hs',   false],
        [ 80, 184,  80, 3, 1, 'hs',   false],
        [ 80, 480, 112, 3, 1, 'hs',   true ],
        [112, 672, 112, 3, 1, 'hs',   true ],
        [112, 672, 160, 5, 2, 'hs',   true ],
        [160, 960, 160, 5, 1, 'hs',   true ],
        [160, 960, 160, 5, 1, 'hs',   true ],
    ];

    private array     $blocks  = [];   // InvertedResidual[]
    private Conv2D    $stemConv;
    private BatchNorm2D $stemBn;
    private HardSwish $stemAct;
    private Conv2D    $headConv;
    private BatchNorm2D $headBn;
    private HardSwish $headAct;
    private GlobalAveragePooling2D $gap;
    private Dense     $classifier;
    private Softmax   $softmax;
    private readonly int $featDim;     // 576 (small) or 960 (large)

    public function __construct(
        private readonly string $variant   = 'small',
        private readonly int    $numClasses = 1000,
        private readonly int    $inputSize  = 224,
    ) {
        $cfg = $variant === 'large' ? self::LARGE_CFG : self::SMALL_CFG;

        // Stem: Conv2D(3→16, k=3, s=2) → BN → HardSwish
        $this->stemConv = new Conv2D(3, 16, kernelSize: 3, stride: 2, padding: 1);
        $this->stemBn   = new BatchNorm2D(16);
        $this->stemAct  = new HardSwish();

        // Inverted residual blocks
        foreach ($cfg as $c) {
            $this->blocks[] = new InvertedResidual(
                inC: $c[0], expandC: $c[1], outC: $c[2],
                kernelSize: $c[3], stride: $c[4],
                activation: $c[5], useSE: $c[6]
            );
        }

        // Head conv
        $lastC         = $variant === 'large' ? 160 : 96;
        $this->featDim = $variant === 'large' ? 960 : 576;
        $this->headConv = new Conv2D($lastC, $this->featDim, kernelSize: 1);
        $this->headBn   = new BatchNorm2D($this->featDim);
        $this->headAct  = new HardSwish();

        $this->gap        = new GlobalAveragePooling2D();
        $this->classifier = new Dense($this->featDim, $numClasses);
        $this->softmax    = new Softmax();
    }

    /* ── Forward pass (returns class probabilities [N, numClasses]) ───────── */
    public function forward(Tensor $x): Tensor
    {
        $x = $this->stemAct->forward(
            $this->stemBn->forward(
                $this->stemConv->forward($x)
            )
        );
        foreach ($this->blocks as $block) {
            $x = $block->forward($x);
        }
        $x = $this->headAct->forward(
            $this->headBn->forward(
                $this->headConv->forward($x)
            )
        );
        $x = $this->gap->forward($x);         // [N, featDim]
        $x = $this->classifier->forward($x);  // [N, numClasses]
        return $this->softmax->forward($x);
    }

    /* ── Feature extraction (returns [N, featDim] before classifier) ─────── */
    public function extract(Tensor $x): Tensor
    {
        $x = $this->stemAct->forward(
            $this->stemBn->forward($this->stemConv->forward($x))
        );
        foreach ($this->blocks as $block) {
            $x = $block->forward($x);
        }
        $x = $this->headAct->forward(
            $this->headBn->forward($this->headConv->forward($x))
        );
        return $this->gap->forward($x);
    }

    /* ── Preprocess: Image → float32 CHW tensor [1, 3, H, W] ─────────────── */
    public function preprocess(Image $img): Tensor
    {
        $ffi = VisionEngine::get()->ffi();
        $raw = $img->ptr();

        // Resize → float32 /255 → CHW
        $resized = $ffi->vision_resize($raw, $this->inputSize, $this->inputSize, 1);
        $floated = $ffi->vision_to_float32($resized, 1.0 / 255.0);
        $ffi->vision_image_free($resized);
        $chw = $ffi->vision_hwc_to_chw($floated);
        $ffi->vision_image_free($floated);

        // Convert CHW VisionImage → owned Tensor (one copy, then free image)
        $rawF = $ffi->vision_image_to_tensor($chw);   // float* CHW buffer
        $ffi->vision_image_free($chw);
        $t = Tensor::fromFloatCopy(\FFI::cast('float*', $rawF), [1, 3, $this->inputSize, $this->inputSize]);
        $ffi->vision_free_raw($rawF);
        return $t;
    }

    /* ── Inference convenience: Image → class probabilities ──────────────── */
    public function classify(Image $img): array
    {
        $x     = $this->preprocess($img);
        $probs = $this->forward($x);
        return $probs->toArray();
    }

    /* ── Get all trainable parameters ─────────────────────────────────────── */
    public function parameters(): array
    {
        $p = array_merge(
            $this->stemConv->getParameters(),
            $this->stemBn->getParameters(),
        );
        foreach ($this->blocks as $i => $b) {
            foreach ($b->getParameters() as $k => $v) $p["block{$i}.{$k}"] = $v;
        }
        return array_merge($p,
            $this->headConv->getParameters(),
            $this->headBn->getParameters(),
            $this->classifier->getParameters(),
        );
    }

    public function featDim(): int { return $this->featDim; }
    public function variant(): string { return $this->variant; }
}
