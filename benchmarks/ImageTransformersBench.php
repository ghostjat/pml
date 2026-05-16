<?php

declare(strict_types=1);

namespace Pml\Benchmarks;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Transformers\ImageResizer;
use Pml\Transformers\ImageCropper;
use Pml\Transformers\ColorSpaceConverter;

/**
 * Performance profile for the High-Speed Image Transformers.
 * Stress tests the ability to manipulate deep image batches directly in OpenBLAS C-Memory.
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Warmup(2)]
#[Bench\Revs(5)]
#[Bench\Iterations(3)]
final class ImageTransformersBench
{
    private static Dataset $imageBatch;
    private static ImageResizer $resizer;
    private static ImageCropper $cropper;
    private static ColorSpaceConverter $converter;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) return;

        // Simulate a massive batch of High-Res CNN Input (e.g., ResNet)
        // 4 Images, 3 Channels (RGB), 112x112 Pixels = 150,528 total floats
        $samples = Tensor::randomUniform([4, 3, 112, 112], 0.0, 255.0);
        self::$imageBatch = new Dataset($samples);

        // Downscale images by exactly half
        self::$resizer = new ImageResizer(56, 56);
        
        // Extract a center 50x50 crop
        self::$cropper = new ImageCropper(31, 31, 50, 50);
        
        // Standard RGB to Grayscale initialization
        self::$converter = new ColorSpaceConverter();

        self::$initialized = true;
    }

    /**
     * Evaluates hardware-accelerated `tensor_take()` across millions of floats.
     */
    #[Bench\Groups(['images', 'resizer'])]
    #[Bench\Assert('mode(variant.time.avg) < 2ms')]
    public function benchParallelNearestNeighborResizer(): void
    {
        self::$resizer->transform(self::$imageBatch);
    }

    /**
     * Evaluates Zero-Copy pointer slicing and flat C-array packing.
     */
    #[Bench\Groups(['images', 'cropper'])]
    #[Bench\Assert('mode(variant.time.avg) < 1ms')]
    public function benchZeroCopyImageCropper(): void
    {
        self::$cropper->transform(self::$imageBatch);
    }

    /**
     * Evaluates AVX2 In-Place Scalar Accumulation.
     */
    #[Bench\Groups(['images', 'colorspace'])]
    #[Bench\Assert('mode(variant.time.avg) < 1ms')]
    public function benchRGBtoGrayscaleConversion(): void
    {
        self::$converter->transform(self::$imageBatch);
    }
}