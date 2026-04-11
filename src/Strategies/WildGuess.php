<?php
declare(strict_types=1);

namespace Pml\Strategies;

use Pml\Tensor;

/**
 * Imputes with a uniform random sample from the [min, max] range of observed values.
 * Useful as a baseline or for additive noise injection.
 */
final class WildGuess implements Strategy
{
    private float $min = 0.0;
    private float $max = 1.0;

    public function fit(Tensor $values): void
    {
        $this->min = $values->min();
        $this->max = $values->max();
    }

    public function guess(): float
    {
        return $this->min + mt_rand() / mt_getrandmax() * ($this->max - $this->min);
    }
}
