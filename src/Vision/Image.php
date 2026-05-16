<?php

declare(strict_types=1);

namespace Pml\Vision;

use FFI;
use Pml\Lib\VisionEngine;

/**
 * Image — a managed wrapper around a C VisionImage pointer.
 *
 * Every public method that returns an Image allocates a new C object.
 * The caller (or GC via __destruct) always owns and frees it.
 *
 * No pixel math here. All processing is dispatched to C via VisionEngine.
 */
final class Image
{
    private FFI\CData $ptr;    // VisionImage*
    private VisionEngine $eng;

    private function __construct(FFI\CData $ptr)
    {
        $this->ptr = $ptr;
        $this->eng = VisionEngine::get();
    }

    public function __destruct()
    {
        $this->eng->ffi()->vision_image_free($this->ptr);
    }

    /* ------------------------------------------------------------------ factories */

    /**
     * Wrap an existing C VisionImage* pointer in a PHP Image.
     * The returned Image takes ownership and will call vision_image_free on destruct.
     * Used by FastSAM / model classes that create VisionImage* in C FFI.
     */
    public static function wrapPtr(FFI\CData $ptr): self
    {
        return new self($ptr);
    }

    /** Load image from file. Channels: 0=auto, 1=gray, 3=RGB, 4=RGBA. */
    public static function read(string $path, int $channels = 0): self
    {
        $ffi = VisionEngine::get()->ffi();
        $ptr = $ffi->vision_imread($path, $channels);
        if (FFI::isNull($ptr)) {
            throw new \RuntimeException("vision_imread failed: {$path}");
        }
        return new self($ptr);
    }

    /** Decode image from raw bytes (e.g. HTTP upload). */
    public static function decode(string $bytes, int $channels = 0): self
    {
        $ffi = VisionEngine::get()->ffi();
        $buf = $ffi->new('uint8_t[' . strlen($bytes) . ']');
        FFI::memcpy($buf, $bytes, strlen($bytes));
        $ptr = $ffi->vision_imdecode($buf, strlen($bytes), $channels);
        if (FFI::isNull($ptr)) {
            throw new \RuntimeException('vision_imdecode failed');
        }
        return new self($ptr);
    }

    /**
     * Create from a flat float32 PHP array in HWC order.
     * Values should be in [0,1].
     */
    public static function fromArray(array $data, int $width, int $height, int $channels): self
    {
        $ffi = VisionEngine::get()->ffi();
        $img = $ffi->vision_image_create(
            $width, $height, $channels,
            PixelFormat::FLOAT32, Layout::HWC, ColorSpace::RGB
        );
        if (FFI::isNull($img)) {
            throw new \RuntimeException('vision_image_create failed');
        }
        $count  = $width * $height * $channels;
        $cArr   = $ffi->new("float[{$count}]");
        for ($i = 0; $i < $count; $i++) {
            $cArr[$i] = (float)($data[$i] ?? 0.0);
        }
        // Copy row by row respecting stride
        $elemSz = 4; // float32
        for ($row = 0; $row < $height; $row++) {
            $dst = $img->data + $row * $img->stride;
            FFI::memcpy($dst, FFI::addr($cArr[$row * $width * $channels]),
                        $width * $channels * $elemSz);
        }
        return new self($img);
    }

    /* ------------------------------------------------------------------ I/O */

    /** Save image to file. Format determined by extension. */
    public function save(string $path): void
    {
        $ok = $this->eng->ffi()->vision_imwrite($path, $this->ptr);
        if (!$ok) {
            throw new \RuntimeException("vision_imwrite failed: {$path}");
        }
    }

    /** Encode to raw bytes. $ext e.g. '.png', '.jpg'. */
    public function encode(string $ext = '.png'): string
    {
        $ffi = $this->eng->ffi();
        $lenPtr = $ffi->new('size_t');
        $buf = $ffi->vision_imencode($this->ptr, $ext, FFI::addr($lenPtr));
        if (FFI::isNull($buf)) {
            throw new \RuntimeException('vision_imencode failed');
        }
        $result = FFI::string($buf, $lenPtr->cdata);
        $ffi->vision_imencode_free($buf);
        return $result;
    }

    /* ------------------------------------------------------------------ accessors */

    public function width(): int    { return $this->ptr->width; }
    public function height(): int   { return $this->ptr->height; }
    public function channels(): int { return $this->ptr->channels; }
    public function format(): int   { return $this->ptr->format; }
    public function layout(): int   { return $this->ptr->layout; }
    public function colorSpace(): int { return $this->ptr->color_space; }

    public function ptr(): FFI\CData { return $this->ptr; }

    /**
     * Copy pixel data out as a flat PHP float array (HWC, row-major).
     * Useful only for small images / debugging.
     */
    public function toArray(): array
    {
        $ffi = $this->eng->ffi();
        $img = $this->toFloat32(); // ensure float32
        $W = $img->width(); $H = $img->height(); $C = $img->channels();
        $result = [];
        $ptr = $img->ptr();
        for ($row = 0; $row < $H; $row++) {
            $rowPtr = FFI::cast('float*', $ptr->data + $row * $ptr->stride);
            for ($x = 0; $x < $W; $x++) {
                for ($c = 0; $c < $C; $c++) {
                    $result[] = (float)$rowPtr[$x * $C + $c];
                }
            }
        }
        return $result;
    }

    /* ------------------------------------------------------------------ format & layout */

    public function toFloat32(float $scale = 1.0 / 255.0): self
    {
        if ($this->ptr->format === PixelFormat::FLOAT32) {
            return $this->clone();
        }
        $ptr = $this->eng->ffi()->vision_to_float32($this->ptr, $scale);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_to_float32 failed');
        return new self($ptr);
    }

    public function toUint8(float $scale = 255.0): self
    {
        if ($this->ptr->format === PixelFormat::UINT8) {
            return $this->clone();
        }
        $ptr = $this->eng->ffi()->vision_to_uint8($this->ptr, $scale);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_to_uint8 failed');
        return new self($ptr);
    }

    public function toHWC(): self
    {
        if ($this->ptr->layout === Layout::HWC) return $this->clone();
        $ptr = $this->eng->ffi()->vision_chw_to_hwc($this->ptr);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_chw_to_hwc failed');
        return new self($ptr);
    }

    public function toCHW(): self
    {
        if ($this->ptr->layout === Layout::CHW) return $this->clone();
        $ptr = $this->eng->ffi()->vision_hwc_to_chw($this->ptr);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_hwc_to_chw failed');
        return new self($ptr);
    }

    /* ------------------------------------------------------------------ resize & spatial */

    public function resize(int $w, int $h, int $interp = Interp::BILINEAR): self
    {
        $ptr = $this->eng->ffi()->vision_resize($this->ptr, $w, $h, $interp);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_resize failed');
        return new self($ptr);
    }

    public function resizeLongEdge(int $longEdge, int $interp = Interp::BILINEAR): self
    {
        $ptr = $this->eng->ffi()->vision_resize_long_edge($this->ptr, $longEdge, $interp);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_resize_long_edge failed');
        return new self($ptr);
    }

    public function crop(int $x, int $y, int $w, int $h): self
    {
        $ptr = $this->eng->ffi()->vision_crop($this->ptr, $x, $y, $w, $h);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_crop failed');
        return new self($ptr);
    }

    public function centerCrop(int $w, int $h): self
    {
        $ptr = $this->eng->ffi()->vision_center_crop($this->ptr, $w, $h);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_center_crop failed');
        return new self($ptr);
    }

    public function pad(int $top, int $bottom, int $left, int $right,
                        int $border = Border::CONSTANT, float $fill = 0.0): self
    {
        $ptr = $this->eng->ffi()->vision_pad($this->ptr, $top, $bottom, $left, $right, $border, $fill);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_pad failed');
        return new self($ptr);
    }

    public function flipHorizontal(): self
    {
        $ptr = $this->eng->ffi()->vision_flip_horizontal($this->ptr);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_flip_horizontal failed');
        return new self($ptr);
    }

    public function flipVertical(): self
    {
        $ptr = $this->eng->ffi()->vision_flip_vertical($this->ptr);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_flip_vertical failed');
        return new self($ptr);
    }

    public function rotate90(int $k = 1): self
    {
        $ptr = $this->eng->ffi()->vision_rotate90($this->ptr, $k);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_rotate90 failed');
        return new self($ptr);
    }

    public function rotate(float $angleDeg, int $interp = Interp::BILINEAR,
                           int $border = Border::CONSTANT, float $fill = 0.0): self
    {
        $ptr = $this->eng->ffi()->vision_rotate($this->ptr, $angleDeg, $interp, $border, $fill);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_rotate failed');
        return new self($ptr);
    }

    /* ------------------------------------------------------------------ color ops */

    public function toGrayscale(): self
    {
        $ptr = $this->eng->ffi()->vision_to_grayscale($this->ptr);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_to_grayscale failed');
        return new self($ptr);
    }

    public function rgbToBgr(): self
    {
        $ptr = $this->eng->ffi()->vision_rgb_to_bgr($this->ptr);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_rgb_to_bgr failed');
        return new self($ptr);
    }

    public function bgrToRgb(): self
    {
        $ptr = $this->eng->ffi()->vision_bgr_to_rgb($this->ptr);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_bgr_to_rgb failed');
        return new self($ptr);
    }

    public function normalize(array $mean, array $stdDev): self
    {
        $ffi = $this->eng->ffi();
        $C = count($mean);
        $mArr = $ffi->new("float[{$C}]");
        $sArr = $ffi->new("float[{$C}]");
        for ($i = 0; $i < $C; $i++) { $mArr[$i] = $mean[$i]; $sArr[$i] = $stdDev[$i]; }
        $ptr = $ffi->vision_normalize($this->ptr, $mArr, $sArr);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_normalize failed');
        return new self($ptr);
    }

    public function adjustBrightness(float $delta): self
    {
        $ptr = $this->eng->ffi()->vision_adjust_brightness($this->ptr, $delta);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_adjust_brightness failed');
        return new self($ptr);
    }

    public function adjustContrast(float $factor): self
    {
        $ptr = $this->eng->ffi()->vision_adjust_contrast($this->ptr, $factor);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_adjust_contrast failed');
        return new self($ptr);
    }

    public function adjustGamma(float $gamma): self
    {
        $ptr = $this->eng->ffi()->vision_adjust_gamma($this->ptr, $gamma);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_adjust_gamma failed');
        return new self($ptr);
    }

    public function histogramEqualize(): self
    {
        $ptr = $this->eng->ffi()->vision_histogram_equalize($this->ptr);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_histogram_equalize failed');
        return new self($ptr);
    }

    /* ------------------------------------------------------------------ filtering */

    public function gaussianBlur(int $radius, float $sigma): self
    {
        $ptr = $this->eng->ffi()->vision_gaussian_blur($this->ptr, $radius, $sigma);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_gaussian_blur failed');
        return new self($ptr);
    }

    public function boxBlur(int $radius): self
    {
        $ptr = $this->eng->ffi()->vision_box_blur($this->ptr, $radius);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_box_blur failed');
        return new self($ptr);
    }

    public function canny(float $lo, float $hi,
                          int $gRadius = 2, float $gSigma = 1.0): self
    {
        $ptr = $this->eng->ffi()->vision_canny($this->ptr, $lo, $hi, $gRadius, $gSigma);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_canny failed');
        return new self($ptr);
    }

    /* ------------------------------------------------------------------ morphology */

    public function erode(int $radius = 1): self
    {
        $ptr = $this->eng->ffi()->vision_erode($this->ptr, $radius);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_erode failed');
        return new self($ptr);
    }

    public function dilate(int $radius = 1): self
    {
        $ptr = $this->eng->ffi()->vision_dilate($this->ptr, $radius);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_dilate failed');
        return new self($ptr);
    }

    /* ------------------------------------------------------------------ misc */

    public function clone(): self
    {
        $ptr = $this->eng->ffi()->vision_image_clone($this->ptr);
        if (FFI::isNull($ptr)) throw new \RuntimeException('vision_image_clone failed');
        return new self($ptr);
    }

    /**
     * Export as CHW float32 tight buffer for use with TensorEngine.
     * Returns raw bytes suitable for tensor_from_raw_float().
     */
    public function toTensorBytes(): string
    {
        $ffi = $this->eng->ffi();
        $chw = ($this->ptr->layout === Layout::CHW) ? $this->clone() : $this->toCHW();
        $f32 = ($chw->ptr()->format === PixelFormat::FLOAT32)
            ? $chw
            : $chw->toFloat32();

        $ptr = $f32->ptr();
        $W = $ptr->width; $H = $ptr->height; $C = $ptr->channels;
        $planeBytes = $W * $H * 4; // float32
        $result = '';
        for ($c = 0; $c < $C; $c++) {
            $planePtr = FFI::cast('char*', $ptr->data + $c * $ptr->stride);
            $result  .= FFI::string($planePtr, $planeBytes);
        }
        return $result;
    }
}
