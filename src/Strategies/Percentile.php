<?php
declare(strict_types=1);

namespace Pml\Strategies;

use Pml\Tensor;
use Pml\Helpers\Stats;

/**
 * Imputes with the p-th percentile of observed values (default: median = 50th).
 */
final class Percentile implements Strategy
{
    private float $value = 0.0;

    public function __construct(private readonly float $p = 50.0) {}

    public function fit(Tensor $values): void
    {
        $this->value = Stats::percentile($values, $this->p);
    }

    public function guess(): float
    {
        return $this->value;
    }
}
