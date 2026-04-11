<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Missing Value Imputer.
 * Replaces NaN (Not a Number) values with the column's mean.
 * * JIT & Memory Optimized:
 * - Leverages C-level `isnan()` checks and AVX2 `where` masks.
 * - Fills millions of missing values instantly without pulling data into PHP.
 */
final class Imputer implements Transformer
{
    private ?Tensor $fillValues = null;

    public function fit(Dataset $dataset): void
    {
        $x = $dataset->samples();
        
        // 1. Create a binary mask of valid numbers (1.0 for valid, 0.0 for NaN)
        $validMask = $x->isNan()->logicalNot();

        // 2. Safely zero out NaNs in a temporary tensor to compute accurate sums
        $zeroedX = $x->copy()->nanToNumInplace(0.0, 0.0, 0.0);
        
        // 3. Calculate the sum of valid values per column
        $sums = $zeroedX->sumAxis(0);
        
        // 4. Count the number of valid values per column (clipped to 1.0 to prevent DivByZero)
        $counts = $validMask->sumAxis(0)->clip(1.0, INF);

        // 5. The replacement values are the true means of the non-NaN elements
        $this->fillValues = $sums->divInplace($counts);
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("Imputer has not been fitted.");
        }

        $x = $dataset->samples();

        // 1. Identify where the NaNs are
        $nanMask = $x->isNan();

        // 2. Broadcast the [1, D] fill values to match the [N, D] shape of the dataset
        $expandedFills = Tensor::zeros(...$x->shape())->addInplace($this->fillValues);

        // 3. Conditional Hardware Masking: Where Mask is True, take Fills; Else take X
        // Completely bypasses PHP loops and handles the replacements in C-cache
        $filledX = $nanMask->where($expandedFills, $x);

        return new Dataset($filledX, $dataset->labels());
    }

    public function fitted(): bool
    {
        return $this->fillValues !== null;
    }
}