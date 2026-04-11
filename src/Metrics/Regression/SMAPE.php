<?php

declare(strict_types=1);

namespace Pml\Metrics\Regression;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Symmetric Mean Absolute Percentage Error (SMAPE).
 * An accuracy measure based on percentage errors.
 * * JIT & Memory Optimized:
 * - 100% vector-matrix arithmetic via OpenBLAS.
 * - Handles division safely with in-place scalar additions.
 */
final class SMAPE implements Metric
{
    public function score(Tensor $predictions, Tensor $labels): float
    {
        // Formula: (100 / N) * sum( |y_true - y_pred| / ((|y_true| + |y_pred|) / 2) )
        
        // 1. Numerator: |y_true - y_pred|
        $numerator = $labels->sub($predictions)->abs();

        // 2. Denominator: |y_true| + |y_pred|
        // Uses addInplace and addScalarInplace(epsilon) to prevent division by zero securely
        $denominator = $labels->abs()
            ->addInplace($predictions->abs())
            ->addScalarInplace(1e-8);

        // 3. Compute fraction: numerator / (denominator / 2) is equivalent to (numerator / denominator) * 2
        // Then multiply by 100 for the percentage
        return $numerator->divInplace($denominator)->mean() * 200.0;
    }
}