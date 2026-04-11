<?php

declare(strict_types=1);

namespace Pml\Losses;

use Pml\Tensor;

/**
 * Mean Squared Error (MSE) Loss function.
 * Highly optimized for Dense layer backpropagation.
 */
final class MeanSquaredError implements Loss
{
    public function compute(Tensor $predictions, Tensor $labels): float
    {
        return $predictions->sub($labels)->square()->mean();
    }

    public function differentiate(Tensor $predictions, Tensor $labels): Tensor
    {
        $n = $predictions->size();
        
        // dY = 2 * (y_pred - y_true) / N
        // Employs Inplace C-mutations to avoid intermediate memory allocations
        $diff = $predictions->sub($labels);
        $diff->mulScalarInplace(2.0 / $n);
        
        return $diff;
    }
}