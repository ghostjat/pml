<?php

declare(strict_types=1);

namespace Pml\Vision;

use FFI;
use Pml\Lib\VisionEngine;

/**
 * Segmentation — mask utilities and connected components.
 *
 * No pixel math in PHP. All work in C.
 */
final class Segmentation
{
    /**
     * Resize a label mask using nearest-neighbour (preserves integer labels).
     */
    public static function maskResize(Image $mask, int $w, int $h): Image
    {
        $ffi = VisionEngine::get()->ffi();
        $ptr = $ffi->vision_mask_resize($mask->ptr(), $w, $h);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_mask_resize failed');
        return self::wrapPtr($ptr);
    }

    /**
     * Rasterize a polygon into a binary mask image.
     *
     * @param  float[] $pts  Flat [x0,y0, x1,y1, ...] polygon vertices
     * @return Image   uint8 single-channel mask
     */
    public static function polygonToMask(array $pts, int $w, int $h,
                                          int $fillVal = 255): Image
    {
        $ffi = VisionEngine::get()->ffi();
        $n   = count($pts);
        $pArr = $ffi->new("float[{$n}]");
        for ($i = 0; $i < $n; $i++) $pArr[$i] = (float)$pts[$i];
        $ptr = $ffi->vision_polygon_rasterize($pArr, $n / 2, $w, $h, $fillVal);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_polygon_rasterize failed');
        return self::wrapPtr($ptr);
    }

    /**
     * Connected components on a binary uint8 mask.
     *
     * Returns an array with:
     *   'n'       => int             (number of components)
     *   'labels'  => int[]           (flat H×W label map, 0=background)
     *   'boxes'   => array[]         (each: x1,y1,x2,y2,area)
     */
    public static function connectedComponents(Image $mask): array
    {
        $ffi = VisionEngine::get()->ffi();
        $cc  = $ffi->vision_connected_components($mask->ptr());
        if (FFI::isNull($cc)) throw new \RuntimeException('vision_connected_components failed');

        $n = (int)$cc->n_components;
        $W = (int)$cc->width;
        $H = (int)$cc->height;

        $labels = [];
        for ($i = 0; $i < $W * $H; $i++) $labels[] = (int)$cc->labels[$i];

        $boxes = [];
        for ($k = 1; $k <= $n; $k++) {
            $boxes[] = [
                'x1'   => (int)$cc->bbox_x1[$k],
                'y1'   => (int)$cc->bbox_y1[$k],
                'x2'   => (int)$cc->bbox_x2[$k],
                'y2'   => (int)$cc->bbox_y2[$k],
                'area' => (int)$cc->areas[$k],
            ];
        }
        $ffi->vision_cc_free($cc);
        return ['n' => $n, 'labels' => $labels, 'boxes' => $boxes];
    }

    private static function wrapPtr(FFI\CData $ptr): Image
    {
        $ref = new \ReflectionClass(Image::class);
        $obj = $ref->newInstanceWithoutConstructor();
        $rp  = $ref->getProperty('ptr'); $rp->setAccessible(true); $rp->setValue($obj, $ptr);
        $re  = $ref->getProperty('eng'); $re->setAccessible(true);
        $re->setValue($obj, VisionEngine::get());
        return $obj;
    }
}
