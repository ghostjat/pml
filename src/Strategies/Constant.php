<?php
declare(strict_types=1);

namespace Pml\Strategies;

use Pml\Tensor;

/**
 * Always returns a user-defined constant — no fitting needed.
 */
final class Constant implements Strategy
{
    public function __construct(private readonly float $value = 0.0) {}

    public function fit(Tensor $values): void {}

    public function guess(): float
    {
        return $this->value;
    }
}
