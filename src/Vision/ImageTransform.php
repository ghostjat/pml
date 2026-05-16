<?php

declare(strict_types=1);

namespace Pml\Vision;

use FFI;
use Pml\Lib\VisionEngine;

/**
 * ImageTransform — static façade for image operations.
 *
 * All heavy work dispatches to C. No PHP pixel loops.
 */
final class ImageTransform
{
    /* ------------------------------------------------------------------ resize */

    public static function resize(Image $img, int $w, int $h,
                                   int $interp = Interp::BILINEAR): Image
    {
        return $img->resize($w, $h, $interp);
    }

    public static function resizeLongEdge(Image $img, int $longEdge,
                                           int $interp = Interp::BILINEAR): Image
    {
        return $img->resizeLongEdge($longEdge, $interp);
    }

    public static function centerCrop(Image $img, int $w, int $h): Image
    {
        return $img->centerCrop($w, $h);
    }

    public static function crop(Image $img, int $x, int $y, int $w, int $h): Image
    {
        return $img->crop($x, $y, $w, $h);
    }

    /* ------------------------------------------------------------------ spatial */

    public static function flipH(Image $img): Image { return $img->flipHorizontal(); }
    public static function flipV(Image $img): Image { return $img->flipVertical(); }
    public static function rotate90(Image $img, int $k = 1): Image { return $img->rotate90($k); }

    public static function rotate(Image $img, float $angleDeg,
                                   int $interp = Interp::BILINEAR,
                                   int $border = Border::CONSTANT,
                                   float $fill = 0.0): Image
    {
        return $img->rotate($angleDeg, $interp, $border, $fill);
    }

    public static function affine(Image $img, array $M6x1,
                                   int $outW, int $outH,
                                   int $interp = Interp::BILINEAR,
                                   int $border = Border::CONSTANT,
                                   float $fill = 0.0): Image
    {
        $ffi = VisionEngine::get()->ffi();
        $mArr = $ffi->new('float[6]');
        for ($i = 0; $i < 6; $i++) $mArr[$i] = (float)$M6x1[$i];
        $ptr = $ffi->vision_affine($img->ptr(), $mArr, $outW, $outH, $interp, $border, $fill);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_affine failed');
        return new Image($ptr); // @phpstan-ignore-line (constructor is private — use reflection)
    }

    /* ------------------------------------------------------------------ color */

    public static function toGrayscale(Image $img): Image { return $img->toGrayscale(); }

    public static function normalize(Image $img, array $mean, array $stdDev): Image
    {
        return $img->normalize($mean, $stdDev);
    }

    public static function adjustBrightness(Image $img, float $d): Image
    {
        return $img->adjustBrightness($d);
    }

    public static function adjustContrast(Image $img, float $f): Image
    {
        return $img->adjustContrast($f);
    }

    public static function adjustGamma(Image $img, float $g): Image
    {
        return $img->adjustGamma($g);
    }

    /* ------------------------------------------------------------------ filter */

    public static function gaussianBlur(Image $img, int $radius, float $sigma): Image
    {
        return $img->gaussianBlur($radius, $sigma);
    }

    public static function canny(Image $img, float $lo, float $hi): Image
    {
        return $img->canny($lo, $hi);
    }

    /* ------------------------------------------------------------------ HOG */

    /**
     * Compute HOG descriptor and return as PHP float array.
     *
     * @return float[]
     */
    public static function hog(Image $img,
                                int $cellSize = 8, int $blockSize = 2,
                                int $nbins = 9): array
    {
        $ffi = VisionEngine::get()->ffi();
        $lenPtr = $ffi->new('int');
        $res = $ffi->vision_hog($img->ptr(), $cellSize, $blockSize, $nbins, FFI::addr($lenPtr));
        if (FFI::isNull($res)) throw new \RuntimeException('vision_hog failed');
        $len  = $lenPtr->cdata;
        $out  = [];
        for ($i = 0; $i < $len; $i++) $out[] = (float)$res->descriptors[$i];
        $ffi->vision_hog_free($res);
        return $out;
    }

    /* ------------------------------------------------------------------ format */

    public static function toFloat32(Image $img, float $scale = 1.0/255.0): Image
    {
        return $img->toFloat32($scale);
    }

    public static function toUint8(Image $img, float $scale = 255.0): Image
    {
        return $img->toUint8($scale);
    }

    public static function toCHW(Image $img): Image { return $img->toCHW(); }
    public static function toHWC(Image $img): Image { return $img->toHWC(); }
}
