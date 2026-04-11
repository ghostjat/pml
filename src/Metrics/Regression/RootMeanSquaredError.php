<?php

declare(strict_types=1);

namespace Pml\Metrics\Regression;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Calculates the Root Mean Squared Error (RMSE).
 */
final class RootMeanSquaredError implements Metric
{
    public function score(Tensor $predictions, Tensor $labels): float
    {
        return sqrt($predictions->sub($labels)->square()->mean());
    }
}