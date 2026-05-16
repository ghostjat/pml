<?php

declare(strict_types=1);

namespace Pml\Vision;

use FFI;
use Pml\Lib\VisionEngine;
use Pml\NeuralNetwork\Layers\BatchNormalization;
use Pml\NeuralNetwork\Layers\Conv2D;
use Pml\NeuralNetwork\Layers\DepthwiseConv2D;
use Pml\NeuralNetwork\Layers\HardSwish;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\Tensor;

/**
 * SSDLite — Single Shot MultiBox Detector with Lite (depthwise-sep) convolutions.
 *
 * Based on MobileNetV2/V3 + SSD (Howard et al., 2018 / Liu et al., 2016).
 *
 * Architecture:
 *   Backbone : MobileNetV3-Small (6 feature maps at strides 4–64)
 *   Extras   : 4 extra conv stages appended for smaller object detection
 *   Head     : Depthwise-separable cls/loc predictors per scale
 *   Anchors  : vision_ssd_prior_boxes (C) — multi-scale, multi-ratio
 *   Decode   : vision_ssd_decode (C) — anchor-based Faster-RCNN style
 *
 * Prior boxes: 6 scales × (2 square + 4 ratio) = ~3000 anchors for 320px input
 *
 * Usage:
 *   $det = new SSDLite(numClasses: 80, inputSize: 320);
 *   $boxes = $det->detect($image, confThr: 0.3);
 *
 * Zero PHP math — anchors + decode all in C.
 */
final class SSDLite
{
    // SSD feature map sizes @ 320px input
    private const FEAT_SIZES = [[40,40], [20,20], [10,10], [5,5], [3,3], [1,1]];
    private const MIN_SIZES  = [21.0, 45.0, 99.0, 153.0, 207.0, 261.0];
    private const MAX_SIZES  = [45.0, 99.0, 153.0, 207.0, 261.0, 315.0];
    private const RATIOS     = [2.0, 3.0];    // will generate 1/2, 1/3 too
    private const ANCHORS_PER_LOC = 6;        // 2 square + 4 ratio

    private MobileNetV3 $backbone;

    // Extra feature stages (after backbone)
    private array $extraConvs = [];   // Conv2D[]

    // SSDLite depthwise-sep prediction heads: cls + loc per scale
    private array $clsPreds  = [];   // [DepthwiseConv2D, Conv2D][6]
    private array $locPreds  = [];   // [DepthwiseConv2D, Conv2D][6]

    private ?FFI\CData $anchors = null;  // cached VisionBBoxArray*
    private int $nAnchors       = 0;

    public function __construct(
        private readonly int   $numClasses = 80,
        private readonly int   $inputSize  = 320,
        private readonly float $confThr    = 0.25,
        private readonly float $iouThr     = 0.45,
        private readonly float $varXY      = 0.1,
        private readonly float $varWH      = 0.2,
    ) {
        $this->backbone = new MobileNetV3('small', numClasses: 1000, inputSize: $inputSize);

        $extraInC  = [576, 256, 128, 64];
        $extraOutC = [256, 128,  64, 32];
        foreach ($extraInC as $i => $inC) {
            $this->extraConvs[] = new Conv2D($inC, $extraOutC[$i], kernelSize: 3,
                                              stride: 2, padding: 1);
        }

        // Scale channels: MV3-Small outputs 576; extras output 256,128,64,32
        $scaleCh = [576, 256, 128, 64, 32, 32];
        $n       = self::ANCHORS_PER_LOC;
        foreach ($scaleCh as $ch) {
            // depthwise-separable cls predictor
            $this->clsPreds[] = [
                new DepthwiseConv2D($ch, kernelSize: 3, padding: 1),
                new Conv2D($ch, $n * ($this->numClasses + 1), kernelSize: 1),
            ];
            // depthwise-separable loc predictor
            $this->locPreds[] = [
                new DepthwiseConv2D($ch, kernelSize: 3, padding: 1),
                new Conv2D($ch, $n * 4, kernelSize: 1),
            ];
        }
    }

    /* ── Build / cache anchors (once per inputSize) ──────────────────────── */
    private function ensureAnchors(): void
    {
        if ($this->anchors !== null) return;
        $ffi = VisionEngine::get()->ffi();

        $nScales = count(self::FEAT_SIZES);
        $nRatios = count(self::RATIOS);

        // Flat int array: [fH0,fW0, fH1,fW1, ...] — avoids 2D CData cast issues
        $flatCount = $nScales * 2;
        $featArr   = $ffi->new("int[{$flatCount}]");
        $minArr    = $ffi->new("float[{$nScales}]");
        $maxArr    = $ffi->new("float[{$nScales}]");
        for ($i = 0; $i < $nScales; $i++) {
            $featArr[$i * 2]     = self::FEAT_SIZES[$i][0];
            $featArr[$i * 2 + 1] = self::FEAT_SIZES[$i][1];
            $minArr[$i]          = self::MIN_SIZES[$i];
            $maxArr[$i]          = self::MAX_SIZES[$i];
        }
        $ratioArr = $ffi->new("float[{$nRatios}]");
        foreach (self::RATIOS as $r => $v) $ratioArr[$r] = $v;

        $this->anchors = $ffi->vision_ssd_prior_boxes(
            FFI::cast('int*',   $featArr),  $nScales,
            FFI::cast('float*', $minArr),
            FFI::cast('float*', $maxArr),
            FFI::cast('float*', $ratioArr), $nRatios,
            $this->inputSize
        );
        $this->nAnchors = $this->anchors !== null ? $this->anchors->count : 0;
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
        $rawF = $ffi->vision_image_to_tensor($chw);
        $ffi->vision_image_free($chw);
        $t = Tensor::fromFloatCopy(\FFI::cast('float*', $rawF), [1, 3, $this->inputSize, $this->inputSize]);
        $ffi->vision_free_raw($rawF);
        return [$t];
    }

    /* ── Forward through cls/loc heads, flatten predictions ─────────────── */
    private function headForward(array $heads, Tensor $feat): Tensor
    {
        [$dw, $pw] = $heads;
        return $pw->forward($dw->forward($feat));
    }

    /* ── Detect ─────────────────────────────────────────────────────────── */
    public function detect(Image $img): array
    {
        $this->ensureAnchors();
        [$x] = $this->preprocess($img);
        $ffi  = VisionEngine::get()->ffi();

        // Backbone → [1, 576]
        $feat = $this->backbone->extract($x);
        [$N, $C] = $feat->shape();

        // Collect predictions from backbone feature + extra conv stages
        $allLocPreds = [];
        $allClsPreds = [];

        $scaleIdx = 0;
        // Scale 0: backbone output reshaped to [1, C, 1, 1]
        $s0 = $feat->reshape($N, $C, 1, 1);
        $allLocPreds[] = $this->headForward($this->locPreds[$scaleIdx], $s0);
        $allClsPreds[] = $this->headForward($this->clsPreds[$scaleIdx], $s0);
        $scaleIdx++;

        // Extra scales
        $last = $s0;
        foreach ($this->extraConvs as $ec) {
            $last = $ec->forward($last);
            $allLocPreds[] = $this->headForward($this->locPreds[$scaleIdx], $last);
            $allClsPreds[] = $this->headForward($this->clsPreds[$scaleIdx], $last);
            $scaleIdx++;
        }

        // Flatten and concatenate all predictions [total_anchors, 4] / [total_anchors, n_cls+1]
        $locFlat = []; $clsFlat = [];
        foreach ($allLocPreds as $lp) { $locFlat[] = $lp->flatten(); }
        foreach ($allClsPreds as $cp) { $clsFlat[] = $cp->flatten(); }

        $locAll = Tensor::concat($locFlat, 0);
        $clsAll = Tensor::concat($clsFlat, 0);

        $locPtr = FFI::cast('float*', $locAll->dataPtr());
        $clsPtr = FFI::cast('float*', $clsAll->dataPtr());

        $bboxArr = $ffi->vision_ssd_decode(
            $locPtr, $clsPtr,
            $this->anchors,
            $this->numClasses, $this->confThr,
            $this->varXY, $this->varWH
        );
        if ($bboxArr === null || $bboxArr->count === 0) return [];

        $nmsed = $ffi->vision_nms($bboxArr, $this->iouThr);
        $ffi->vision_bbox_array_free($bboxArr);

        $out = [];
        for ($i = 0; $i < $nmsed->count; $i++) {
            $b    = $nmsed->boxes[$i];
            $out[] = [
                'x1' => $b->x1, 'y1' => $b->y1, 'x2' => $b->x2, 'y2' => $b->y2,
                'score' => $b->score, 'class_id' => $b->class_id,
            ];
        }
        $ffi->vision_bbox_array_free($nmsed);
        return $out;
    }

    public function __destruct()
    {
        if ($this->anchors !== null) {
            VisionEngine::get()->ffi()->vision_bbox_array_free($this->anchors);
            $this->anchors = null;
        }
    }

    public function backbone(): MobileNetV3 { return $this->backbone; }
    public function numAnchors(): int        { $this->ensureAnchors(); return $this->nAnchors; }
}
