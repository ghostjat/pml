<?php
declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;

/**
 * Safe Euclidean Distance — Euclidean distance with NaN/Inf protection.
 * NaN and Inf values are replaced with 0 before computing the distance.
 *
 * JIT & Memory Optimized: single nanToNumInplace call in C, then standard dot product.
 */
final class SafeEuclidean implements Distance
{
    public function compute(Tensor $a, Tensor $b): float
    {
        $ca = $a->copy()->nanToNumInplace(0.0, 0.0, 0.0);
        $cb = $b->copy()->nanToNumInplace(0.0, 0.0, 0.0);
        $diff = $ca->sub($cb);
        return sqrt($diff->dot($diff));
    }
}
