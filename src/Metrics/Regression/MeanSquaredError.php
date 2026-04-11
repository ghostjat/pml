<?php

declare(strict_types=1);

namespace Pml\Metrics\Regression;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Calculates the Mean Squared Error (MSE).
 */
final class MeanSquaredError implements Metric
{
    public function score(Tensor $predictions, Tensor $labels): float
    {
        // Executes entirely in AVX2 C-memory:
        // 1. Subtract labels from predictions (creates 1 temporary tensor)
        // 2. Square it
        // 3. OpenMP parallel reduction to get the mean
        return $predictions->sub($labels)->square()->mean();
    }
}