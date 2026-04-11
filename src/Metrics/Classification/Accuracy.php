<?php

declare(strict_types=1);

namespace Pml\Metrics\Classification;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Calculates the accuracy for classification tasks.
 */
final class Accuracy implements Metric
{
    public function score(Tensor $predictions, Tensor $labels): float
    {
        // 1. Round predictions to nearest integer (e.g., 0.8 -> 1.0)
        // 2. C-level logical check (creates binary 1.0 / 0.0 mask)
        // 3. The mean of the mask is the exact accuracy percentage
        return $predictions->round()->equal($labels)->mean();
    }
}