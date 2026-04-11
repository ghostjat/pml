<?php
declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;

/**
 * Euclidean (L2) Distance — sqrt( sum((a_i - b_i)^2) ).
 * JIT & Memory Optimized: diff → square → sum → sqrt all in C.
 */
final class Euclidean implements Distance
{
    public function compute(Tensor $a, Tensor $b): float
    {
        $diff = $a->sub($b);
        return sqrt($diff->dot($diff));
    }
}
