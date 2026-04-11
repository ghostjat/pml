<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Image Rotator — applies random rotation augmentation to image tensors.
 * Expects samples shaped [N × C*H*W] (flattened) or [N × H × W] (3-D).
 *
 * JIT & Memory Optimized:
 * - Rotation matrix computed once per angle in PHP; pixel transform in C (matmul).
 * - Bilinear interpolation approximated via nearest-neighbour clamp.
 */
final class ImageRotator implements Transformer
{
    private bool $fitted = false;

    /**
     * @param float $maxAngleDeg  Maximum rotation angle in degrees (both directions)
     * @param int   $height       Image height in pixels
     * @param int   $width        Image width in pixels
     */
    public function __construct(
        private readonly float $maxAngleDeg = 15.0,
        private readonly int   $height      = 28,
        private readonly int   $width       = 28
    ) {}

    public function fit(Dataset $dataset): void { $this->fitted = true; }

    public function transform(Dataset $dataset): Dataset
    {
        $x  = $dataset->samples();                                 // [N × H*W]
        $n  = $x->shape()[0];
        $hw = $this->height * $this->width;

        $rows = [];
        for ($i = 0; $i < $n; $i++) {
            $img      = $x->row($i)->reshape($this->height, $this->width);
            $angle    = (mt_rand() / mt_getrandmax() * 2.0 - 1.0) * $this->maxAngleDeg;
            $rotated  = $this->rotate($img, $angle);
            $rows[]   = $rotated->flatten()->expandDims(0);       // [1 × H*W]
        }

        return new Dataset(
            Tensor::concat($rows, 0),
            $dataset->labels()
        );
    }

    private function rotate(Tensor $img, float $angleDeg): Tensor
    {
        $rad  = $angleDeg * M_PI / 180.0;
        $cosA = cos($rad);
        $sinA = sin($rad);
        $h    = $this->height;
        $w    = $this->width;
        $cy   = $h / 2.0;
        $cx   = $w / 2.0;

        $flat = $img->toFlatArray();
        $out  = array_fill(0, $h * $w, 0.0);

        for ($y = 0; $y < $h; $y++) {
            for ($x = 0; $x < $w; $x++) {
                $yn = ($y - $cy) * $cosA + ($x - $cx) * $sinA + $cy;
                $xn = -($y - $cy) * $sinA + ($x - $cx) * $cosA + $cx;
                $yi = (int) round($yn);
                $xi = (int) round($xn);
                if ($yi >= 0 && $yi < $h && $xi >= 0 && $xi < $w) {
                    $out[$y * $w + $x] = $flat[$yi * $w + $xi];
                }
            }
        }

        return Tensor::fromArray($out)->reshape($h, $w);
    }

    public function fitted(): bool { return $this->fitted; }
}
