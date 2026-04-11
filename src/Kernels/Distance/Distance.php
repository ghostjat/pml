<?php

declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;

/**
 * Interface for Distance metrics.
 */
interface Distance
{
    /**
     * Compute distance between vector A and matrix B.
     * @return Tensor Distance scores.
     */
    public function compute(Tensor $a, Tensor $b): Tensor;
}