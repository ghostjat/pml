<?php
declare(strict_types=1);

namespace Pml\Metrics\Regression;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Median Absolute Error — robust to outliers, unlike MAE/RMSE.
 * Negated so higher = better (consistent with scoring convention).
 *
 * JIT & Memory Optimized: absolute differences computed in C; median via C-level partial sort.
 */
final class MedianAbsoluteError implements Metric
{
    public function score(Tensor $predictions, ?Tensor $labels): float
    {
        if ($labels === null) {
            throw new \InvalidArgumentException("MedianAbsoluteError requires ground-truth labels.");
        }
        return -$predictions->sub($labels)->abs()->median();       // negated: higher = better
    }

    public function range(): array { return [-INF, 0.0]; }
}
