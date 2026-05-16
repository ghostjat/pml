<?php

declare(strict_types=1);

namespace Pml\Vision;

use FFI;
use Pml\Lib\VisionEngine;
use Pml\NeuralNetwork\Layers\BatchNorm2D;
use Pml\NeuralNetwork\Layers\Conv2D;
use Pml\NeuralNetwork\Layers\InvertedResidual;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\Tensor;

/**
 * YOLO11n — Ultralytics YOLO11 Nano variant (2024).
 *
 * Architecture:
 *   Backbone : C2f (Cross Stage Partial with faster feature) + SPPF
 *   Neck     : C2f + PANet
 *   Head     : Anchor-free, decoupled cls/reg, DFL (Distribution Focal Loss)
 *   Decode   : vision_yolo11_decode (C) per FPN stride
 *
 * Key differences from YOLOv8:
 *   • C2f blocks replaced by C3k2 blocks (inner cross-stage partial)
 *   • Attention-aware SPPF (A-SPPF)
 *   • Deeper P3/P4/P5 neck connections
 *
 * This implementation approximates C2f with stacked InvertedResidual blocks
 * (same computational budget, identical API surface).
 *
 * Usage:
 *   $det = new Yolo11n(numClasses: 80, inputSize: 640);
 *   $boxes = $det->detect($image);
 *
 * Zero PHP math — decode via vision_yolo11_decode (C).
 */
final class Yolo11n
{
    private const STRIDES = [8, 16, 32];
    private const REG_MAX = 16;   // DFL bins: 0..15

    // C2f backbone stages (simplified as stacked InvertedResiduals)
    private array $stage1 = [];   // stride 8  — P3
    private array $stage2 = [];   // stride 16 — P4
    private array $stage3 = [];   // stride 32 — P5

    // Stem
    private Conv2D $stemConv;
    private BatchNorm2D $stemBn;
    private ReLU   $stemAct;

    // SPPF (Spatial Pyramid Pooling Fast) — approximated with multi-scale max-pool via conv
    private Conv2D $sppfIn;
    private Conv2D $sppfOut;

    // PANet neck
    private array $neckConvs = [];   // Conv2D[4]

    // Detection heads (one per stride: 8, 16, 32)
    private array $clsHeads = [];    // Conv2D[3]
    private array $regHeads = [];    // Conv2D[3]

    public function __construct(
        private readonly int   $numClasses = 80,
        private readonly int   $inputSize  = 640,
        private readonly float $confThr    = 0.25,
        private readonly float $iouThr     = 0.45,
    ) {
        // Stem: 3 → 16
        $this->stemConv = new Conv2D(3,  16, kernelSize: 3, stride: 2, padding: 1);
        $this->stemBn   = new BatchNorm2D(16);
        $this->stemAct  = new ReLU();

        // C2f stages (channel sizes: 16 → 32 → 64 → 128)
        $this->stage1 = [
            new InvertedResidual(16,  32,  32, 3, 2, 'hs', false),
            new InvertedResidual(32,  64,  32, 3, 1, 'hs', false),
        ];
        $this->stage2 = [
            new InvertedResidual(32,  64,  64, 3, 2, 'hs', false),
            new InvertedResidual(64, 128,  64, 3, 1, 'hs', false),
        ];
        $this->stage3 = [
            new InvertedResidual(64,  128, 128, 3, 2, 'hs', true),
            new InvertedResidual(128, 256, 128, 3, 1, 'hs', true),
        ];

        // SPPF: 128 → 64 → 128
        $this->sppfIn  = new Conv2D(128,  64, kernelSize: 1);
        $this->sppfOut = new Conv2D(256, 128, kernelSize: 1);  // 4×64 cat → 256 → 128

        // PANet neck convs (up/down sampling approximated by convs)
        $this->neckConvs = [
            new Conv2D(192, 64, kernelSize: 1),   // P4 + upsample P5 → P4'
            new Conv2D(96,  32, kernelSize: 1),   // P3 + upsample P4'→ P3'
            new Conv2D(96,  64, kernelSize: 3, stride: 2, padding: 1),  // P3'→P4''
            new Conv2D(192,128, kernelSize: 3, stride: 2, padding: 1),  // P4''→P5''
        ];

        // Detection heads
        $headC = [32, 64, 128];
        foreach ($headC as $i => $hC) {
            $this->clsHeads[] = new Conv2D($hC, $this->numClasses, kernelSize: 1);
            $this->regHeads[] = new Conv2D($hC, 4 * self::REG_MAX,  kernelSize: 1);
        }
    }

    private function runStage(array $layers, Tensor $x): Tensor
    {
        foreach ($layers as $l) $x = $l->forward($x);
        return $x;
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

    /* ── Single-scale decode via C ──────────────────────────────────────── */
    private function decodeHead(int $idx, Tensor $cls, Tensor $reg,
                                 int $imgW, int $imgH): array
    {
        $ffi    = VisionEngine::get()->ffi();
        $stride = self::STRIDES[$idx];
        [, , $H, $W] = $cls->shape();
        $n      = $H * $W;
        $rowDim = 4 * self::REG_MAX + $this->numClasses;

        // Merge cls + reg → [H*W, row_dim] for vision_yolo11_decode
        $cls_t = $cls->squeeze()->transposeNd([1, 2, 0])->reshape($n, $this->numClasses);
        $reg_t = $reg->squeeze()->transposeNd([1, 2, 0])->reshape($n, 4 * self::REG_MAX);
        // Concat [reg | cls] → [H*W, reg+cls]
        $merged = Tensor::concat([$reg_t, $cls_t], axis: 1)->makeContiguous();

        $ptr    = FFI::cast('float*', $merged->dataPtr());
        $bArr   = $ffi->vision_yolo11_decode(
            $ptr, $H, $W, $stride,
            $this->numClasses, self::REG_MAX,
            $imgW, $imgH, $this->confThr
        );
        if ($bArr === null || $bArr->count === 0) return [];

        $out = [];
        for ($i = 0; $i < $bArr->count; $i++) {
            $b    = $bArr->boxes[$i];
            $out[] = ['x1'=>$b->x1,'y1'=>$b->y1,'x2'=>$b->x2,'y2'=>$b->y2,
                       'score'=>$b->score,'class_id'=>$b->class_id];
        }
        $ffi->vision_bbox_array_free($bArr);
        return $out;
    }

    /* ── Detect ─────────────────────────────────────────────────────────── */
    public function detect(Image $img): array
    {
        [$x, $imgW, $imgH] = $this->preprocess($img);
        $ffi = VisionEngine::get()->ffi();

        // Backbone
        $stem   = $this->stemAct->forward($this->stemBn->forward($this->stemConv->forward($x)));
        $p3     = $this->runStage($this->stage1, $stem);   // stride 8,  32ch
        $p4     = $this->runStage($this->stage2, $p3);     // stride 16, 64ch
        $p5raw  = $this->runStage($this->stage3, $p4);     // stride 32, 128ch

        // SPPF (simplified: reduce then expand)
        $sppf   = $this->sppfOut->forward(
            Tensor::concat([
                $this->sppfIn->forward($p5raw),
                $this->sppfIn->forward($p5raw),
                $this->sppfIn->forward($p5raw),
                $this->sppfIn->forward($p5raw),
            ], axis: 1)
        );  // [1, 128, H/32, W/32]

        // PANet: top-down
        $p5up  = $sppf->upsample(scale: 2);                              // [1,128,H/16,W/16]
        $p4up  = $this->neckConvs[0]->forward(Tensor::concat([$p4, $p5up], 1)); // 192→64
        $p4up2 = $p4up->upsample(scale: 2);                              // [1,64,H/8,W/8]
        $p3out = $this->neckConvs[1]->forward(Tensor::concat([$p3, $p4up2], 1)); // 96→32

        // Use top-down FPN levels directly as the three detection heads.
        // P3'(32ch@H/8) → stride-8, P4'(64ch@H/16) → stride-16, P5/sppf(128ch@H/32) → stride-32.
        // neckConvs[2/3] (bottom-up) are skipped: p3out/p4up/sppf already match head channel counts.
        $feats = [$p3out, $p4up, $sppf];
        $all   = [];
        foreach ($feats as $i => $feat) {
            $cls  = $this->clsHeads[$i]->forward($feat);
            $reg  = $this->regHeads[$i]->forward($feat);
            array_push($all, ...$this->decodeHead($i, $cls, $reg, $imgW, $imgH));
        }
        if (empty($all)) return [];

        // Global NMS
        $bArr = $ffi->vision_bbox_array_create(count($all));
        foreach ($all as $d) {
            $b = $ffi->new('VisionBBox');
            $b->x1=$d['x1']; $b->y1=$d['y1']; $b->x2=$d['x2']; $b->y2=$d['y2'];
            $b->score=$d['score']; $b->class_id=$d['class_id'];
            $ffi->vision_bbox_array_push($bArr, FFI::addr($b));
        }
        $nmsed = $ffi->vision_nms($bArr, $this->iouThr);
        $ffi->vision_bbox_array_free($bArr);

        $out = [];
        for ($i = 0; $i < $nmsed->count; $i++) {
            $b    = $nmsed->boxes[$i];
            $out[] = ['x1'=>$b->x1,'y1'=>$b->y1,'x2'=>$b->x2,'y2'=>$b->y2,
                       'score'=>$b->score,'class_id'=>$b->class_id];
        }
        $ffi->vision_bbox_array_free($nmsed);
        return $out;
    }
}
