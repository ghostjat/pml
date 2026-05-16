<?php

declare(strict_types=1);

namespace Pml\Vision;

use FFI;
use Pml\Lib\VisionEngine;

/**
 * Augmentation — builder pipeline for random image augmentations.
 *
 * All randomisation runs in C via xoshiro128+ RNG. No PHP random.
 *
 * Usage:
 *   $aug = (new Augmentation(seed: 42))
 *       ->randomFlipH(0.5)
 *       ->randomBrightness(0.2)
 *       ->randomCrop(224, 224);
 *   $augmented = $aug($image);
 */
final class Augmentation
{
    private FFI\CData $rng;
    private array $steps = [];

    public function __construct(int $seed = 0)
    {
        $eng = VisionEngine::get();
        $this->rng = $eng->newRng();
        if ($seed !== 0) {
            $eng->ffi()->vision_rng_init(FFI::addr($this->rng), $seed);
        }
    }

    /* ------------------------------------------------------------------ builder methods */

    public function randomFlipH(float $prob = 0.5): self
    {
        $this->steps[] = ['flipH', $prob];
        return $this;
    }

    public function randomFlipV(float $prob = 0.5): self
    {
        $this->steps[] = ['flipV', $prob];
        return $this;
    }

    public function randomCrop(int $w, int $h): self
    {
        $this->steps[] = ['crop', $w, $h];
        return $this;
    }

    public function randomResizeCrop(int $w, int $h,
                                      float $scaleLo = 0.08, float $scaleHi = 1.0,
                                      float $ratioLo = 0.75, float $ratioHi = 1.333,
                                      int $interp = Interp::BILINEAR): self
    {
        $this->steps[] = ['rrc', $w, $h, $scaleLo, $scaleHi, $ratioLo, $ratioHi, $interp];
        return $this;
    }

    public function randomBrightness(float $maxDelta = 0.2): self
    {
        $this->steps[] = ['brightness', $maxDelta];
        return $this;
    }

    public function randomContrast(float $lo = 0.8, float $hi = 1.2): self
    {
        $this->steps[] = ['contrast', $lo, $hi];
        return $this;
    }

    public function randomHue(float $maxDelta = 0.1): self
    {
        $this->steps[] = ['hue', $maxDelta];
        return $this;
    }

    public function randomRotation(float $maxAngle = 15.0,
                                    int $interp = Interp::BILINEAR,
                                    int $border = Border::REFLECT,
                                    float $fill = 0.0): self
    {
        $this->steps[] = ['rotation', $maxAngle, $interp, $border, $fill];
        return $this;
    }

    public function cutout(int $nHoles = 1, int $holeSize = 16, float $fill = 0.0): self
    {
        $this->steps[] = ['cutout', $nHoles, $holeSize, $fill];
        return $this;
    }

    /* ------------------------------------------------------------------ apply */

    /** Apply the augmentation pipeline to one image. Returns a new Image. */
    public function __invoke(Image $input): Image
    {
        $ffi  = VisionEngine::get()->ffi();
        $rngP = FFI::addr($this->rng);
        $current = $input->clone();

        foreach ($this->steps as $step) {
            $op  = $step[0];
            $next = null;
            switch ($op) {
                case 'flipH':
                    $next = $ffi->vision_random_flip_horizontal($current->ptr(), $step[1], $rngP);
                    break;
                case 'flipV':
                    $next = $ffi->vision_random_flip_vertical($current->ptr(), $step[1], $rngP);
                    break;
                case 'crop':
                    $next = $ffi->vision_random_crop($current->ptr(), $step[1], $step[2], $rngP);
                    break;
                case 'rrc':
                    $next = $ffi->vision_random_resize_crop(
                        $current->ptr(), $step[1], $step[2],
                        $step[3], $step[4], $step[5], $step[6],
                        $rngP, $step[7]
                    );
                    break;
                case 'brightness':
                    $next = $ffi->vision_random_brightness($current->ptr(), $step[1], $rngP);
                    break;
                case 'contrast':
                    $next = $ffi->vision_random_contrast($current->ptr(), $step[1], $step[2], $rngP);
                    break;
                case 'hue':
                    $next = $ffi->vision_random_hue($current->ptr(), $step[1], $rngP);
                    break;
                case 'rotation':
                    $next = $ffi->vision_random_rotation(
                        $current->ptr(), $step[1], $rngP, $step[2], $step[3], $step[4]
                    );
                    break;
                case 'cutout':
                    $next = $ffi->vision_cutout($current->ptr(), $step[1], $step[2], $step[3], $rngP);
                    break;
            }
            if ($next !== null && !FFI::isNull($next)) {
                $current = $this->wrapPtr($next);
            }
        }
        return $current;
    }

    /** Mixup two images (returns new image + lambda). */
    public function mixup(Image $a, Image $b, float $alpha = 0.2): array
    {
        $ffi    = VisionEngine::get()->ffi();
        $lambda = $ffi->new('float');
        $ptr    = $ffi->vision_mixup(
            $a->ptr(), $b->ptr(), $alpha, FFI::addr($this->rng), FFI::addr($lambda)
        );
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_mixup failed');
        return [$this->wrapPtr($ptr), (float)$lambda->cdata];
    }

    /** CutMix two images (returns new image + lambda). */
    public function cutmix(Image $a, Image $b, float $alpha = 1.0): array
    {
        $ffi    = VisionEngine::get()->ffi();
        $lambda = $ffi->new('float');
        $ptr    = $ffi->vision_cutmix(
            $a->ptr(), $b->ptr(), $alpha, FFI::addr($this->rng), FFI::addr($lambda)
        );
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_cutmix failed');
        return [$this->wrapPtr($ptr), (float)$lambda->cdata];
    }

    private function wrapPtr(FFI\CData $ptr): Image
    {
        // Access private constructor via reflection for clean encapsulation
        $ref = new \ReflectionClass(Image::class);
        $obj = $ref->newInstanceWithoutConstructor();
        $refProp = $ref->getProperty('ptr');
        $refProp->setAccessible(true);
        $refProp->setValue($obj, $ptr);
        $refEng = $ref->getProperty('eng');
        $refEng->setAccessible(true);
        $refEng->setValue($obj, VisionEngine::get());
        return $obj;
    }
}
