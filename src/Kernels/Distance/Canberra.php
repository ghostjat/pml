<?php
declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;

/**
 * Canberra Distance — sum( |a_i - b_i| / (|a_i| + |b_i|) ).
 * Sensitive to values near zero; useful for non-negative sparse data.
 *
 * JIT & Memory Optimized: numerator and denominator computed in C;
 * division uses clip to avoid 0/0.
 */
final class Canberra implements Distance
{
    public function compute(Tensor $a, Tensor $b): float
    {
        $num   = $a->sub($b)->abs();                               // |a - b|
        $denom = $a->abs()->add($b->abs())->clip(1e-10, INF);     // |a| + |b|
        return $num->div($denom)->sum();
    }
}
