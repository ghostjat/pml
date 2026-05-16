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
 * FastSAM — Fast Segment Anything Model (Zhao et al., 2023).
 *
 * FastSAM replaces SAM's ViT backbone with a YOLOv8-style CNN, enabling
 * real-time instance segmentation at 50× the speed of SAM.
 *
 * Architecture:
 *   Backbone : YOLO11n-style C2f backbone (3 strides)
 *   Neck     : PANet feature pyramid
 *   Det head : Anchor-free box + objectness (like YOLO11n)
 *   Seg head : Mask prototype bank (32 protos) + per-detection coefficients
 *   Decode   : vision_yolo11_decode (boxes) + vision_fastsam_assemble_masks (masks)
 *
 * Outputs per detection:
 *   • Bounding box (x1, y1, x2, y2)
 *   • Confidence score
 *   • Class id (0 = generic object for promptless segmentation)
 *   • Binary mask (VisionImage, same size as input)
 *
 * Usage:
 *   $sam = new FastSAM(inputSize: 640);
 *   $segs = $sam->segment($image, confThr: 0.35);
 *   foreach ($segs as $s) {
 *       echo "class={$s['class_id']} score={$s['score']}\n";
 *       $s['mask']->imwrite("mask_{$i}.png");
 *   }
 *
 * Zero PHP math — all decode + mask assembly in C.
 */
final class FastSAM
{
    private const STRIDES   = [8, 16, 32];
    private const REG_MAX   = 16;
    private const N_PROTO   = 32;    // prototype mask channels
    private const PROTO_SZ  = 160;   // prototype spatial size (1/4 of 640 input)

    // Backbone (shared with YOLO11n)
    private Conv2D $stemConv;
    private BatchNorm2D $stemBn;
    private ReLU   $stemAct;
    private array  $stage1 = [];
    private array  $stage2 = [];
    private array  $stage3 = [];
    private Conv2D $sppfIn;
    private Conv2D $sppfOut;

    // PANet neck (same as YOLO11n)
    private array $neckConvs = [];

    // Detection heads (box + cls per stride)
    private array $clsHeads = [];
    private array $regHeads = [];

    // Segmentation: mask coefficient heads per stride
    private array $coefHeads = [];   // Conv2D[3] → n_proto coefficients

    // Prototype mask generation head (P3 scale)
    private Conv2D $protoConv1;
    private Conv2D $protoConv2;
    private Conv2D $protoConv3;

    public function __construct(
        private readonly int   $numClasses = 1,    // 1 for promptless "everything"
        private readonly int   $inputSize  = 640,
        private readonly float $confThr    = 0.35,
        private readonly float $iouThr     = 0.45,
        private readonly float $maskThr    = 0.5,
    ) {
        // Backbone (same as Yolo11n)
        $this->stemConv = new Conv2D(3,  16, kernelSize: 3, stride: 2, padding: 1);
        $this->stemBn   = new BatchNorm2D(16);
        $this->stemAct  = new ReLU();

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

        $this->sppfIn  = new Conv2D(128,  64, kernelSize: 1);
        $this->sppfOut = new Conv2D(256, 128, kernelSize: 1);

        // PANet neck (same topology as YOLO11n)
        $this->neckConvs = [
            new Conv2D(192, 64,  kernelSize: 1),
            new Conv2D(96,  32,  kernelSize: 1),
            new Conv2D(96,  64,  kernelSize: 3, stride: 2, padding: 1),
            new Conv2D(192, 128, kernelSize: 3, stride: 2, padding: 1),
        ];

        // Detection heads
        $headC = [32, 64, 128];
        foreach ($headC as $i => $hC) {
            $this->clsHeads[]  = new Conv2D($hC, $this->numClasses, kernelSize: 1);
            $this->regHeads[]  = new Conv2D($hC, 4 * self::REG_MAX,  kernelSize: 1);
            $this->coefHeads[] = new Conv2D($hC, self::N_PROTO,       kernelSize: 1);
        }

        // Prototype head: P3 → 32 prototype masks at PROTO_SZ × PROTO_SZ
        $this->protoConv1 = new Conv2D(32,  64,             kernelSize: 3, padding: 1);
        $this->protoConv2 = new Conv2D(64,  64,             kernelSize: 3, padding: 1);
        $this->protoConv3 = new Conv2D(64,  self::N_PROTO,  kernelSize: 1);
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

    /* ── Segment: Image → array of {class_id, score, bbox, mask} ───────── */
    public function segment(Image $img): array
    {
        [$x, $imgW, $imgH] = $this->preprocess($img);
        $ffi = VisionEngine::get()->ffi();

        // Backbone
        $stem  = $this->stemAct->forward($this->stemBn->forward($this->stemConv->forward($x)));
        $p3    = $this->runStage($this->stage1, $stem);
        $p4    = $this->runStage($this->stage2, $p3);
        $p5raw = $this->runStage($this->stage3, $p4);
        $sppf  = $this->sppfOut->forward(
            Tensor::concat(array_fill(0, 4, $this->sppfIn->forward($p5raw)), axis: 1)
        );

        // PANet
        $p5up  = $sppf->upsample(scale: 2);
        $p4up  = $this->neckConvs[0]->forward(Tensor::concat([$p4, $p5up], 1));
        $p4up2 = $p4up->upsample(scale: 2);
        $p3out = $this->neckConvs[1]->forward(Tensor::concat([$p3, $p4up2], 1));
        // Use top-down FPN levels directly (same fix as Yolo11n — bottom-up concat is
        // spatially invalid since p3out@H/8 ≠ p4up@H/16).
        // p3out(32ch@H/8), p4up(64ch@H/16), sppf(128ch@H/32) match head channel counts.

        // Prototype masks from P3 (highest resolution)
        $proto = $this->protoConv3->forward(
            $this->protoConv2->forward(
                $this->protoConv1->forward($p3out)
            )
        );  // [1, N_PROTO, pH, pW]
        [, $nP, $pH, $pW] = $proto->shape();

        // Detection + coefficient heads
        $feats   = [$p3out, $p4up, $sppf];
        $allDets = [];
        $allCoef = [];
        foreach ($feats as $i => $feat) {
            $cls  = $this->clsHeads[$i]->forward($feat);
            $reg  = $this->regHeads[$i]->forward($feat);
            $coef = $this->coefHeads[$i]->forward($feat);  // [1, N_PROTO, H, W]

            [,, $H, $W] = $cls->shape();
            $n = $H * $W;
            $stride = self::STRIDES[$i];

            $cls_t   = $cls->squeeze()->transposeNd([1,2,0])->reshape($n, $this->numClasses);
            $reg_t   = $reg->squeeze()->transposeNd([1,2,0])->reshape($n, 4 * self::REG_MAX);
            $merged  = Tensor::concat([$reg_t, $cls_t], axis: 1)->makeContiguous();
            $ptr     = FFI::cast('float*', $merged->dataPtr());

            $bArr = $ffi->vision_yolo11_decode(
                $ptr, $H, $W, $stride,
                $this->numClasses, self::REG_MAX,
                $imgW, $imgH, $this->confThr
            );
            if ($bArr === null || $bArr->count === 0) continue;

            for ($j = 0; $j < $bArr->count; $j++) {
                $b = $bArr->boxes[$j];
                $allDets[] = [
                    'x1'=>$b->x1,'y1'=>$b->y1,'x2'=>$b->x2,'y2'=>$b->y2,
                    'score'=>$b->score,'class_id'=>$b->class_id,
                ];
                // Extract coefficient vector for this detection location
                // (spatial position in coef map corresponding to detection centre)
                $cx = (int)((($b->x1 + $b->x2) * 0.5) / $stride);
                $cy = (int)((($b->y1 + $b->y2) * 0.5) / $stride);
                $cx = max(0, min($W - 1, $cx));
                $cy = max(0, min($H - 1, $cy));
                // coef: [1, N_PROTO, H, W] → slice at [0, :, cy, cx]
                $coef_vec = $coef->slice(0, 0, 1)   // [1,N_PROTO,H,W]
                                  ->squeeze()         // [N_PROTO,H,W]
                                  ->slice(1, $cy, 1)  // [N_PROTO,1,W]
                                  ->squeeze()         // [N_PROTO,W]
                                  ->slice(1, $cx, 1)  // [N_PROTO,1]
                                  ->flatten();         // [N_PROTO]
                $allCoef[] = $coef_vec;
            }
            $ffi->vision_bbox_array_free($bArr);
        }

        if (empty($allDets)) return [];

        $nDets = count($allDets);

        // Pack coefficient tensors → [nDets, N_PROTO]
        $coefMat  = Tensor::concat($allCoef, axis: 0)->reshape($nDets, self::N_PROTO)->makeContiguous();
        $coefPtr  = FFI::cast('float*', $coefMat->dataPtr());

        // Proto flat: [N_PROTO, pH, pW] contiguous
        $protoFlat = $proto->squeeze()->makeContiguous();  // [N_PROTO, pH, pW]
        $protoPtr  = FFI::cast('float*', $protoFlat->dataPtr());

        // Build VisionBBoxArray for mask cropping
        $bboxArr = $ffi->vision_bbox_array_create($nDets);
        foreach ($allDets as $d) {
            $b = $ffi->new('VisionBBox');
            $b->x1=$d['x1']; $b->y1=$d['y1']; $b->x2=$d['x2']; $b->y2=$d['y2'];
            $b->score=$d['score']; $b->class_id=$d['class_id'];
            $ffi->vision_bbox_array_push($bboxArr, FFI::addr($b));
        }

        // Assemble masks in C → [n_dets channels, imgH, imgW] uint8
        $masksImg = $ffi->vision_fastsam_assemble_masks(
            $protoPtr, $nP, $pH, $pW,
            $coefPtr,  $nDets,
            $bboxArr,
            $imgW, $imgH, $this->maskThr
        );
        $ffi->vision_bbox_array_free($bboxArr);

        // NMS on detection boxes
        $bArr2 = $ffi->vision_bbox_array_create($nDets);
        foreach ($allDets as $d) {
            $b = $ffi->new('VisionBBox');
            $b->x1=$d['x1']; $b->y1=$d['y1']; $b->x2=$d['x2']; $b->y2=$d['y2'];
            $b->score=$d['score']; $b->class_id=$d['class_id'];
            $ffi->vision_bbox_array_push($bArr2, FFI::addr($b));
        }
        $nmsed = $ffi->vision_nms($bArr2, $this->iouThr);
        $ffi->vision_bbox_array_free($bArr2);

        // Build result: each detection gets its mask as a single-channel VisionImage
        $results = [];
        $maskData = $masksImg !== null ? $ffi->vision_image_data_ptr($masksImg) : null;
        $maskStride = $imgW * $imgH;

        for ($i = 0; $i < min($nmsed->count, $nDets); $i++) {
            $b = $nmsed->boxes[$i];

            // Wrap each mask channel as its own Image
            $maskImg = null;
            if ($maskData !== null && $i < $nDets) {
                $chan   = $ffi->vision_image_create($imgW, $imgH, 1,
                    0 /* UINT8 */, 0 /* HWC */, 2 /* GRAY */);
                if ($chan !== null) {
                    $dst = $ffi->vision_image_data_ptr($chan);
                    // Copy single channel from packed CHW mask image
                    FFI::memcpy($dst, $maskData + $i * $maskStride, (int)$maskStride);
                    $maskImg = Image::wrapPtr($chan);
                }
            }

            $results[] = [
                'x1'       => $b->x1, 'y1'    => $b->y1,
                'x2'       => $b->x2, 'y2'    => $b->y2,
                'score'    => $b->score,
                'class_id' => $b->class_id,
                'mask'     => $maskImg,   // Image|null
            ];
        }

        $ffi->vision_bbox_array_free($nmsed);
        if ($masksImg !== null) $ffi->vision_image_free($masksImg);

        return $results;
    }
}
