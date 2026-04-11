<?php
declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;

/**
 * Diagonal (Weighted Euclidean) Distance — sqrt( sum( w_i * (a_i - b_i)^2 ) ).
 * Each feature is scaled by its own weight vector, enabling feature importance weighting.
 *
 * JIT & Memory Optimized: weight multiplication is a single in-place C mul.
 */
final class Diagonal implements Distance
{
    private Tensor $weights;

    /**
     * @param float[] $weights  Per-feature weight vector (length must equal feature dim)
     */
    public function __construct(array $weights)
    {
        $this->weights = Tensor::fromArray($weights);
    }

    public function compute(Tensor $a, Tensor $b): float
    {
        $diff = $a->sub($b);
        return sqrt($diff->square()->mul($this->weights)->sum());
    }
}
