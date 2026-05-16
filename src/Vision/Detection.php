<?php

declare(strict_types=1);

namespace Pml\Vision;

use FFI;
use Pml\Lib\VisionEngine;

/**
 * Detection — static wrapper for bounding-box utilities.
 *
 * All computation is in C. PHP only marshals box arrays.
 */
final class Detection
{
    /* ------------------------------------------------------------------ IoU variants */

    /** Intersection-over-Union. */
    public static function iou(array $a, array $b): float
    {
        $ffi = VisionEngine::get()->ffi();
        $ba  = self::toC($ffi, $a);
        $bb  = self::toC($ffi, $b);
        return (float)$ffi->vision_iou(FFI::addr($ba), FFI::addr($bb));
    }

    /** Generalised IoU. */
    public static function giou(array $a, array $b): float
    {
        $ffi = VisionEngine::get()->ffi();
        $ba = self::toC($ffi, $a); $bb = self::toC($ffi, $b);
        return (float)$ffi->vision_giou(FFI::addr($ba), FFI::addr($bb));
    }

    /** Distance IoU. */
    public static function diou(array $a, array $b): float
    {
        $ffi = VisionEngine::get()->ffi();
        $ba = self::toC($ffi, $a); $bb = self::toC($ffi, $b);
        return (float)$ffi->vision_diou(FFI::addr($ba), FFI::addr($bb));
    }

    /* ------------------------------------------------------------------ NMS */

    /**
     * Greedy NMS.
     *
     * @param  array[] $boxes  Each: ['x1','y1','x2','y2','score','class_id']
     * @return array[]
     */
    public static function nms(array $boxes, float $iouThresh = 0.5): array
    {
        $ffi = VisionEngine::get()->ffi();
        $arr = self::buildArray($ffi, $boxes);
        $out = $ffi->vision_nms($arr, $iouThresh);
        $ffi->vision_bbox_array_free($arr);
        if (FFI::isNull($out)) return [];
        $result = self::fromArray($out);
        $ffi->vision_bbox_array_free($out);
        return $result;
    }

    /**
     * Soft-NMS (Gaussian weight decay).
     *
     * @param  array[] $boxes
     * @return array[]
     */
    public static function softNms(array $boxes, float $sigma = 0.5,
                                    float $scoreThresh = 0.001): array
    {
        $ffi = VisionEngine::get()->ffi();
        $arr = self::buildArray($ffi, $boxes);
        $out = $ffi->vision_soft_nms($arr, $sigma, $scoreThresh);
        $ffi->vision_bbox_array_free($arr);
        if (FFI::isNull($out)) return [];
        $result = self::fromArray($out);
        $ffi->vision_bbox_array_free($out);
        return $result;
    }

    /* ------------------------------------------------------------------ Anchors */

    /**
     * Generate anchor boxes for a feature map.
     *
     * @param  float[] $scales  e.g. [0.5, 1.0, 2.0]
     * @param  float[] $ratios  e.g. [0.5, 1.0, 2.0]
     * @return array[]
     */
    public static function generateAnchors(int $featW, int $featH, int $stride,
                                            array $scales, array $ratios): array
    {
        $ffi = VisionEngine::get()->ffi();
        $ns  = count($scales);
        $nr  = count($ratios);
        $sArr = $ffi->new("float[{$ns}]");
        $rArr = $ffi->new("float[{$nr}]");
        for ($i = 0; $i < $ns; $i++) $sArr[$i] = (float)$scales[$i];
        for ($i = 0; $i < $nr; $i++) $rArr[$i] = (float)$ratios[$i];
        $out = $ffi->vision_generate_anchors($featW, $featH, $stride, $sArr, $ns, $rArr, $nr);
        if (FFI::isNull($out)) return [];
        $result = self::fromArray($out);
        $ffi->vision_bbox_array_free($out);
        return $result;
    }

    /* ------------------------------------------------------------------ encode / decode */

    /**
     * Encode target box relative to anchor (Faster-RCNN style).
     * Returns [dx, dy, dw, dh].
     */
    public static function encode(array $anchor, array $target): array
    {
        $ffi = VisionEngine::get()->ffi();
        $ca  = self::toC($ffi, $anchor); $ct = self::toC($ffi, $target);
        $dx  = $ffi->new('float'); $dy = $ffi->new('float');
        $dw  = $ffi->new('float'); $dh = $ffi->new('float');
        $ffi->vision_bbox_encode(
            FFI::addr($ca), FFI::addr($ct),
            FFI::addr($dx), FFI::addr($dy), FFI::addr($dw), FFI::addr($dh)
        );
        return [(float)$dx->cdata, (float)$dy->cdata,
                (float)$dw->cdata, (float)$dh->cdata];
    }

    /**
     * Decode delta (dx,dy,dw,dh) back to a box.
     */
    public static function decode(array $anchor, float $dx, float $dy,
                                   float $dw, float $dh): array
    {
        $ffi = VisionEngine::get()->ffi();
        $ca  = self::toC($ffi, $anchor);
        $out = $ffi->new('VisionBBox');
        $ffi->vision_bbox_decode(FFI::addr($ca), $dx, $dy, $dw, $dh, FFI::addr($out));
        return self::boxToPhp($out);
    }

    /* ------------------------------------------------------------------ helpers */

    private static function toC(FFI $ffi, array $box): FFI\CData
    {
        $b = $ffi->new('VisionBBox');
        $b->x1 = $box['x1'] ?? $box[0] ?? 0.0;
        $b->y1 = $box['y1'] ?? $box[1] ?? 0.0;
        $b->x2 = $box['x2'] ?? $box[2] ?? 0.0;
        $b->y2 = $box['y2'] ?? $box[3] ?? 0.0;
        $b->score    = $box['score']    ?? 1.0;
        $b->class_id = $box['class_id'] ?? 0;
        return $b;
    }

    private static function boxToPhp(FFI\CData $b): array
    {
        return [
            'x1'       => (float)$b->x1,
            'y1'       => (float)$b->y1,
            'x2'       => (float)$b->x2,
            'y2'       => (float)$b->y2,
            'score'    => (float)$b->score,
            'class_id' => (int)$b->class_id,
        ];
    }

    private static function buildArray(FFI $ffi, array $boxes): FFI\CData
    {
        $n   = count($boxes);
        $arr = $ffi->vision_bbox_array_create($n);
        foreach ($boxes as $box) {
            $cb = self::toC($ffi, $box);
            $ffi->vision_bbox_array_push($arr, FFI::addr($cb));
        }
        return $arr;
    }

    private static function fromArray(FFI\CData $arr): array
    {
        $result = [];
        for ($i = 0; $i < $arr->count; $i++) {
            $result[] = self::boxToPhp($arr->boxes[$i]);
        }
        return $result;
    }
}
