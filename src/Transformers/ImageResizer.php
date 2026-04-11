<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use InvalidArgumentException;

/**
 * Image Resizer (Nearest Neighbor).
 * Resizes image tensors of shape [Batch, Channels, Height, Width].
 * * JIT & Memory Optimized:
 * - Avoids disastrous nested PHP pixel loops.
 * - Pre-calculates target mapping indices locally, then uses hardware-accelerated `tensor_take()`
 * to massively parallelize the resize in pure C-memory cache.
 */
final class ImageResizer implements Transformer
{
    private int $targetWidth;
    private int $targetHeight;

    public function __construct(int $targetWidth, int $targetHeight)
    {
        if ($targetWidth < 1 || $targetHeight < 1) {
            throw new InvalidArgumentException("Target dimensions must be at least 1x1.");
        }
        
        $this->targetWidth = $targetWidth;
        $this->targetHeight = $targetHeight;
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
            throw new InvalidArgumentException("ImageResizer requires image tensors of shape [Batch, Channels, Height, Width].");
        }

        $oldHeight = $shape[2];
        $oldWidth = $shape[3];

        // 1. Pre-calculate the Nearest-Neighbor index mappings in PHP
        $indicesH = [];
        for ($i = 0; $i < $this->targetHeight; $i++) {
            $indicesH[] = (int) floor($i * $oldHeight / $this->targetHeight);
        }
        
        $indicesW = [];
        for ($i = 0; $i < $this->targetWidth; $i++) {
            $indicesW[] = (int) floor($i * $oldWidth / $this->targetWidth);
        }

        // 2. Push index mapping arrays to contiguous C-Pointers
        $tH = Tensor::fromArray($indicesH);
        $tW = Tensor::fromArray($indicesW);

        // 3. Hardware-Accelerated Resizing
        // `take()` extracts arbitrary slices along an axis based on the index array provided.
        // We take the mapped Height indices, and then the mapped Width indices.
        $resized = $samples->take($tH, 2)->take($tW, 3);

        return new Dataset($resized, $dataset->labels());
    }

    public function fitted(): bool
    {
        return true;
    }
}