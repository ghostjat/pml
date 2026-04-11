<?php

declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;

/**
 * Hamming Distance.
 * Measures the proportion of coordinates that differ between two vectors. Highly optimized for categorical/boolean sets.
 */
final class Hamming implements Distance
{
    public function compute(Tensor $a, Tensor $b): Tensor
    {
        // Vectorized boolean mismatch: sum(A != B, axis=1) / D
        $d = (float) $b->shape()[1];
        
        $mismatchMask = $b->notEqual($a);
        
        return $mismatchMask->sumAxis(1)->mulScalarInplace(1.0 / $d);
    }
}