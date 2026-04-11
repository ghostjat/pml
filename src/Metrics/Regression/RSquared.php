<?php

declare(strict_types=1);

namespace Pml\Metrics\Regression;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * R-Squared (Coefficient of Determination).
 * Measures the proportion of variance in the dependent variable predictable from the features.
 * * JIT & Memory Optimized:
 * - Computes Residual Sum of Squares (SS_res) and Total Sum of Squares (SS_tot) directly in OpenBLAS.
 */
final class RSquared implements Metric
{
    public function score(Tensor $predictions, Tensor $labels): float
    {
        // 1. Residual Sum of Squares: sum((y_true - y_pred)^2)
        $ssRes = $labels->sub($predictions)->square()->sum();

        // 2. Total Sum of Squares: sum((y_true - mean(y_true))^2)
        $meanLabel = $labels->mean();
        $ssTot = $labels->addScalar(-$meanLabel)->square()->sum();

        // Prevent division by zero if variance is completely 0
        if ($ssTot < 1e-8) {
            return 0.0;
        }

        return 1.0 - ($ssRes / $ssTot);
    }
}