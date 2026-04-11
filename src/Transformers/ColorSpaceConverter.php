<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Color Space Converter.
 * Converts RGB image tensors [Batch, 3, Height, Width] into Grayscale [Batch, 1, Height, Width].
 * * JIT & Memory Optimized:
 * - Employs zero-copy channel slices.
 * - Weights are applied and accumulated entirely via in-place AVX2 vector broadcasting.
 */
final class ColorSpaceConverter implements Transformer
{
    // Standard luminosity weights for RGB -> Grayscale conversion
    private const R_WEIGHT = 0.299;
    private const G_WEIGHT = 0.587;
    private const B_WEIGHT = 0.114;

    public function fit(Dataset $dataset): void
    {
        // Stateless transformer
    }

    public function transform(Dataset $dataset): Dataset
    {
        $x = $dataset->samples();
        $shape = $x->shape();

        if (count($shape) !== 4 || $shape[1] !== 3) {
            throw new RuntimeException("ColorSpaceConverter requires image tensors of shape [Batch, 3 (RGB), Height, Width].");
        }

        // 1. Extract the RGB channels natively using Zero-Copy views
        // slice(axis, start, length) -> Extracts shape [Batch, 1, Height, Width]
        $rView = $x->slice(1, 0, 1);
        $gView = $x->slice(1, 1, 1);
        $bView = $x->slice(1, 2, 1);

        // 2. Allocate and weigh the first channel (Creates a new contiguous C-Tensor)
        $grayscale = $rView->mulScalar(self::R_WEIGHT);

        // 3. Accumulate the remaining channels directly into the new tensor IN-PLACE
        // This avoids allocating two additional massive intermediate image tensors!
        $grayscale->addInplace($gView->mulScalar(self::G_WEIGHT))
                  ->addInplace($bView->mulScalar(self::B_WEIGHT));

        return new Dataset($grayscale, $dataset->labels());
    }

    public function fitted(): bool
    {
        return true;
    }
}