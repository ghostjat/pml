<?php

declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;

/**
 * Manhattan Distance (L1 Norm).
 * Computes the distance as the sum of absolute differences across all dimensions.
 * * JIT & Memory Optimized:
 * - 100% Vectorized AVX2 `abs()` and `sumAxis()` execution.
 */
final class Manhattan implements Distance
{
    public function compute(Tensor $a, Tensor $b): Tensor
    {
        // Formula: sum( |A - B|, axis=1 )
        // Automatically broadcasts the [D] vector $a against the [N, D] matrix $b
        return $b->sub($a)->abs()->sumAxis(1);
    }
}