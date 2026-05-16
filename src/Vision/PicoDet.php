<?php

declare(strict_types=1);

namespace Pml\Vision;

use FFI;
use Pml\Lib\VisionEngine;
use Pml\NeuralNetwork\Layers\BatchNormalization;
use Pml\NeuralNetwork\Layers\Conv2D;
use Pml\NeuralNetwork\Layers\DepthwiseConv2D;
use Pml\NeuralNetwork\Layers\HardSwish;
use Pml\NeuralNetwork\Layers\InvertedResidual;
use Pml\Tensor;

/**
 * PicoDet — Lightweight Anchor-Free Detector (Baidu PaddleDetection, 2021).
 *
 * Architecture:
 *   Backbone : MobileNetV3-Small
 *   Neck     : CSP-PAN (Cross Stage Partial Path Aggregation Network)
 *   Head     : Decoupled aligned head + DFL (Distribution Focal Loss)
 *   Decode   : vision_picodet_decode (C) — identical math to NanoDet DFL
 *
 * Differences from NanoDet:
 *   • CSP blocks in neck (not PAN-Lite)
 *   • Aligned head (shared between classification and regression)
 *   • Larger default input size: 320×320 or 416×416
 *
 * Usage:
 *   $det = new PicoDet(numClasses: 80, inputSize: 320);
 *   $boxes = $det->detect($image);
 *
 * Zero PHP math — decode via vision_picodet_decode (C).
 */
final class PicoDet
{
    private const STRIDES = [8, 16, 32, 64];
    private const REG_MAX = 7;

    private MobileNetV3 $backbone;

    // CSP-PAN neck layers (simplified: lateral + dw-sep bottleneck)
    private array $lateralConvs = [];
    private array $cspBlocks    = [];

    // Aligned detection head (shared cls+reg context conv)
    private array $contextConvs = [];
    private array $clsHeads     = [];
    private array $regHeads     = [];

    public function __construct(
        private readonly int   $numClasses = 80,
        private readonly int   $inputSize  = 320,
        private readonly float $confThr    = 0.25,
        private readonly float $iouThr     = 0.45,
    ) {
        $this->backbone = new MobileNetV3('small', numClasses: 1000, inputSize: $inputSize);

        $fpnC = [24, 48, 96, 96];
        $headC = 96;

        foreach (self::STRIDES as $i => $_) {
            $inC = $i < count($fpnC) ? $fpnC[$i] : 96;
            $this->lateralConvs[] = new Conv2D($inC, $headC, kernelSize: 1);
            // CSP block: depthwise + pointwise bottleneck
            $this->cspBlocks[]    = new InvertedResidual(
                inC: $headC, expandC: $headC * 2, outC: $headC,
                kernelSize: 3, stride: 1, activation: 'hs', useSE: false
            );
            // Aligned context conv (shared features for cls+reg)
            $this->contextConvs[] = new DepthwiseConv2D($headC, kernelSize: 5, padding: 2);
            // Cls head: context → n_cls
            $this->clsHeads[]     = new Conv2D($headC, $numClasses, kernelSize: 1);
            // Reg head: context → 4*reg_max
            $this->regHeads[]     = new Conv2D($headC, 4 * self::REG_MAX, kernelSize: 1);
        }
    }

    private function bboxArrayToPhp(FFI\CData $arr, FFI $ffi): array
    {
        $out = [];
        for ($i = 0; $i < $arr->count; $i++) {
            $b    = $arr->boxes[$i];
            $out[] = [
                'x1' => $b->x1, 'y1' => $b->y1, 'x2' => $b->x2, 'y2' => $b->y2,
                'score' => $b->score, 'class_id' => $b->class_id,
            ];
        }
        return $out;
    }

    private function preprocess(Image $img): array
    {
        $ffi = VisionEngine::get()->ffi();
        $raw = $img->ptr();
        $resized = $ffi->vision_resize($raw, $this->inputSize, $this->inputSize, 1);
        $floated = $ffi->vision_to_float32($resized, 1.0 / 255.0);
        $ffi->vision_image_free($resized);
        $chw = $ffi->vision_hwc_to_chw($floated);
        $ffi->vision_image_free($floated);
        $origW = $ffi->vision_image_width($raw);
        $origH = $ffi->vision_image_height($raw);
        $rawF  = $ffi->vision_image_to_tensor($chw);
        $ffi->vision_image_free($chw);
        $t = Tensor::fromFloatCopy(\FFI::cast('float*', $rawF), [1, 3, $this->inputSize, $this->inputSize]);
        $ffi->vision_free_raw($rawF);
        return [$t, $origW, $origH];
    }

    /* ── Detect ─────────────────────────────────────────────────────────────── */
    public function detect(Image $img): array
    {
        [$x, $imgW, $imgH] = $this->preprocess($img);
        $ffi = VisionEngine::get()->ffi();

        // Extract backbone features
        $feat = $this->backbone->extract($x);   // [1, 576]

        // Use largest stride scale (stride 64) for single-feature-map decode
        $scaleIdx = count(self::STRIDES) - 1;
        $stride   = self::STRIDES[$scaleIdx];
        $H = max(1, (int)($this->inputSize / $stride));
        $W = $H;
        $C = $this->backbone->featDim();

        $featMap = $feat->reshape(1, $C, 1, 1)->mul(Tensor::ones(1, $C, $H, $W));
        $lat     = $this->lateralConvs[$scaleIdx]->forward($featMap);
        $csp     = $this->cspBlocks[$scaleIdx]->forward($lat);
        $ctx     = $this->contextConvs[$scaleIdx]->forward($csp);
        $cls     = $this->clsHeads[$scaleIdx]->forward($ctx);   // [1, n_cls, H, W]
        $reg     = $this->regHeads[$scaleIdx]->forward($ctx);   // [1, 4*reg_max, H, W]

        // Transpose to [H*W, *] for C decode
        $cls_t  = $cls->squeeze()->transposeNd([1, 2, 0])->reshape($H * $W, $this->numClasses);
        $reg_t  = $reg->squeeze()->transposeNd([1, 2, 0])->reshape($H * $W, 4 * self::REG_MAX);
        $clsPtr = FFI::cast('float*', $cls_t->dataPtr());
        $regPtr = FFI::cast('float*', $reg_t->dataPtr());

        $bboxArr = $ffi->vision_picodet_decode(
            $clsPtr, $regPtr, $H, $W, $stride,
            $this->numClasses, self::REG_MAX,
            $imgW, $imgH, $this->confThr
        );
        if ($bboxArr === null || $bboxArr->count === 0) return [];

        $nmsed = $ffi->vision_nms($bboxArr, $this->iouThr);
        $ffi->vision_bbox_array_free($bboxArr);
        $result = $this->bboxArrayToPhp($nmsed, $ffi);
        $ffi->vision_bbox_array_free($nmsed);
        return $result;
    }

    public function backbone(): MobileNetV3 { return $this->backbone; }
}
