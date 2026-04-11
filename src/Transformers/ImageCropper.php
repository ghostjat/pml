<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use InvalidArgumentException;

/**
 * Image Cropper.
 * Crops a bounding box from image tensors of shape [Batch, Channels, Height, Width].
 * * JIT & Memory Optimized:
 * - Uses nested `slice()` commands to adjust C-pointers, resulting in an instant Zero-Copy crop.
 * - Safely calls `copy()` to pack the disjointed memory view back into a flat contiguous array for CNNs.
 */
final class ImageCropper implements Transformer
{
    private int $xOffset;
    private int $yOffset;
    private int $width;
    private int $height;

    /**
     * @param int $xOffset Starting X coordinate (Width axis).
     * @param int $yOffset Starting Y coordinate (Height axis).
     * @param int $width The target width of the cropped image.
     * @param int $height The target height of the cropped image.
     */
    public function __construct(int $xOffset, int $yOffset, int $width, int $height)
    {
        if ($width < 1 || $height < 1) {
            throw new InvalidArgumentException("Crop dimensions must be at least 1x1.");
        }
        
        $this->xOffset = max(0, $xOffset);
        $this->yOffset = max(0, $yOffset);
        $this->width = $width;
        $this->height = $height;
    }

    public function fit(Dataset $dataset): void
    {
        // Stateless transformer
    }

    public function transform(Dataset $dataset): Dataset
    {
        $samples = $dataset->samples();
        $shape = $samples->shape();

        if (count($shape) !== 4) {
            throw new InvalidArgumentException("ImageCropper requires image tensors of shape [Batch, Channels, Height, Width].");
        }

        $imgHeight = $shape[2];
        $imgWidth = $shape[3];

        if ($this->yOffset + $this->height > $imgHeight || $this->xOffset + $this->width > $imgWidth) {
            throw new InvalidArgumentException("Crop dimensions exceed the boundaries of the image tensors.");
        }

        // 1. Chained Zero-Copy Cropping
        // Slice Axis 2 (Height), then immediately Slice Axis 3 (Width) of the resulting view.
        $croppedView = $samples->slice(2, $this->yOffset, $this->height)
                               ->slice(3, $this->xOffset, $this->width);

        // 2. Pack to contiguous memory
        // CNN convolutional kernels require physically contiguous blocks in RAM.
        // `copy()` safely flattens the complex offset strides into a fresh continuous C-array.
        $contiguousCrop = $croppedView->copy();

        return new Dataset($contiguousCrop, $dataset->labels());
    }

    public function fitted(): bool
    {
        return true;
    }
}