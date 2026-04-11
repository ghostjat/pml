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
    private Dataset $imageBatch;
    private ImageResizer $resizer;
    private ImageCropper $cropper;
    private ColorSpaceConverter $converter;

    public function setUp(): void
    {
        // Simulate a massive batch of High-Res CNN Input (e.g., ResNet)
        // 32 Images, 3 Channels (RGB), 224x224 Pixels = 4,816,896 total floats
        $samples = Tensor::randomUniform([32, 3, 224, 224], 0.0, 255.0);
        $this->imageBatch = new Dataset($samples);

        // Downscale images by exactly half
        $this->resizer = new ImageResizer(112, 112);
        
        // Extract a center 100x100 crop
        $this->cropper = new ImageCropper(62, 62, 100, 100);
        
        // Standard RGB to Grayscale initialization
        $this->converter = new ColorSpaceConverter();
    }

    /**
     * Evaluates hardware-accelerated `tensor_take()` across millions of floats.
     */
    #[Bench\Groups(['images', 'resizer'])]
    #[Bench\Assert('mode(variant.time.avg) < 15ms')]
    public function benchParallelNearestNeighborResizer(): void
    {
        $this->resizer->transform($this->imageBatch);
    }

    /**
     * Evaluates Zero-Copy pointer slicing and flat C-array packing.
     */
    #[Bench\Groups(['images', 'cropper'])]
    #[Bench\Assert('mode(variant.time.avg) < 5ms')]
    public function benchZeroCopyImageCropper(): void
    {
        $this->cropper->transform($this->imageBatch);
    }

    /**
     * Evaluates AVX2 In-Place Scalar Accumulation.
     */
    #[Bench\Groups(['images', 'colorspace'])]
    #[Bench\Assert('mode(variant.time.avg) < 10ms')]
    public function benchRGBtoGrayscaleConversion(): void
    {
        $this->converter->transform($this->imageBatch);
    }
}