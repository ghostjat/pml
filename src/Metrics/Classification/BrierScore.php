<?php
declare(strict_types=1);

namespace Pml\Metrics\Classification;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Brier Score — mean squared error of probability forecasts.
 * Score range [0, 1]: lower is better (0 = perfect).
 * Inverted to [-1, 0] so higher = better (consistent with other metrics).
 *
 * JIT & Memory Optimized: MSE computed via single in-place C subtraction + reduction.
 */
final class BrierScore implements Metric
{
    public function score(Tensor $predictions, ?Tensor $labels): float
    {
        if ($labels === null) {
            throw new \InvalidArgumentException("BrierScore requires ground-truth labels.");
        }
        // MSE in C: mean((p - y)^2)
        $diff = $predictions->sub($labels);
        return -$diff->square()->mean();           // negated: higher = better
    }

    public function range(): array { return [-1.0, 0.0]; }
}
