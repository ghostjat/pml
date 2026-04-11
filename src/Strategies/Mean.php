<?php
declare(strict_types=1);

namespace Pml\Strategies;

use Pml\Tensor;

/**
 * Imputes with the arithmetic mean of observed values.
 * C-level mean reduction — O(1) FFI boundary crossing.
 */
final class Mean implements Strategy
{
    private float $mean = 0.0;

    public function fit(Tensor $values): void
    {
        $this->mean = $values->mean();
    }

    public function guess(): float
    {
        return $this->mean;
    }
}
