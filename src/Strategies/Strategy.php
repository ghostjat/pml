<?php
declare(strict_types=1);

namespace Pml\Strategies;

use Pml\Tensor;

/**
 * Imputation strategy interface.
 * Strategies learn a fill value from training data, then produce it on demand.
 */
interface Strategy
{
    /**
     * Fit the strategy to a 1-D tensor of observed values.
     */
    public function fit(Tensor $values): void;

    /**
     * Return the imputed fill value as a float.
     */
    public function guess(): float;
}
