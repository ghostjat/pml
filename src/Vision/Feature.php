<?php

declare(strict_types=1);

namespace Pml\Vision;

use FFI;
use Pml\Lib\VisionEngine;

/**
 * Feature — wrapper for feature extraction (HOG, LBP, Harris, FAST).
 *
 * Returns PHP arrays. No PHP-side computation.
 */
final class Feature
{
    /**
     * Compute HOG descriptors.
     *
     * @return float[]
     */
    public static function hog(Image $img, int $cellSize = 8,
                                int $blockSize = 2, int $nbins = 9): array
    {
        $ffi    = VisionEngine::get()->ffi();
        $lenPtr = $ffi->new('int');
        $res    = $ffi->vision_hog($img->ptr(), $cellSize, $blockSize, $nbins,
                                   FFI::addr($lenPtr));
        if (FFI::isNull($res)) throw new \RuntimeException('vision_hog failed');
        $len = (int)$lenPtr->cdata;
        $out = [];
        for ($i = 0; $i < $len; $i++) $out[] = (float)$res->descriptors[$i];
        $ffi->vision_hog_free($res);
        return $out;
    }

    /**
     * Compute LBP histogram descriptors.
     *
     * @return float[]
     */
    public static function lbp(Image $img, int $radius = 1,
                                int $gridX = 4, int $gridY = 4): array
    {
        $ffi    = VisionEngine::get()->ffi();
        $lenPtr = $ffi->new('int');
        $res    = $ffi->vision_lbp($img->ptr(), $radius, $gridX, $gridY,
                                   FFI::addr($lenPtr));
        if (FFI::isNull($res)) throw new \RuntimeException('vision_lbp failed');
        $len = (int)$lenPtr->cdata;
        $out = [];
        for ($i = 0; $i < $len; $i++) $out[] = (float)$res->descriptors[$i];
        $ffi->vision_lbp_free($res);
        return $out;
    }

    /**
     * Harris corner detection.
     *
     * @return array[] Each: ['x'=>float, 'y'=>float, 'score'=>float]
     */
    public static function harrisCorners(Image $img, float $k = 0.04,
                                          float $threshold = 0.01,
                                          int $nmsRadius = 5): array
    {
        $ffi     = VisionEngine::get()->ffi();
        $cntPtr  = $ffi->new('int');
        $res     = $ffi->vision_harris_corners($img->ptr(), $k, $threshold,
                                               $nmsRadius, FFI::addr($cntPtr));
        if (FFI::isNull($res)) return [];
        $n   = (int)$cntPtr->cdata;
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[] = [
                'x'     => (float)$res->x[$i],
                'y'     => (float)$res->y[$i],
                'score' => (float)$res->score[$i],
            ];
        }
        $ffi->vision_corners_free($res);
        return $out;
    }

    /**
     * FAST keypoint detection.
     *
     * @return array[] Each: ['x'=>float, 'y'=>float, 'score'=>float]
     */
    public static function fastCorners(Image $img, int $threshold = 20,
                                        int $nConsecutive = 9): array
    {
        $ffi    = VisionEngine::get()->ffi();
        $cntPtr = $ffi->new('int');
        $res    = $ffi->vision_fast_corners($img->ptr(), $threshold, $nConsecutive,
                                            FFI::addr($cntPtr));
        if (FFI::isNull($res)) return [];
        $n   = (int)$cntPtr->cdata;
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[] = [
                'x'     => (float)$res->x[$i],
                'y'     => (float)$res->y[$i],
                'score' => (float)$res->score[$i],
            ];
        }
        $ffi->vision_corners_free($res);
        return $out;
    }
}
