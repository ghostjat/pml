<?php

declare(strict_types=1);

namespace Pml\Metrics\Regression;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Calculates the Mean Absolute Error (MAE).
 */
final class MeanAbsoluteError implements Metric
{
    public function score(Tensor $predictions, Tensor $labels): float
    {
        return $predictions->sub($labels)->abs()->mean();
    }
}