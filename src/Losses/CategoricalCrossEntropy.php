<?php

declare(strict_types=1);

namespace Pml\Losses;

use Pml\Tensor;

/**
 * Categorical Cross Entropy Loss.
 * Standard loss function for Multi-Class classification paired with Softmax.
 * Expects $labels to be One-Hot Encoded (e.g., [0, 1, 0, 0]).
 */
final class CategoricalCrossEntropy implements Loss
{
    public function compute(Tensor $predictions, Tensor $labels): float
    {
        // Clip to prevent log(0) explosions
        $clipped = $predictions->clip(1e-7, 1.0 - 1e-7);
        
        // Formula: -mean( sum( y_true * log(y_pred), axis=1 ) )
        $logPreds = $clipped->log();
        $product = $labels->mul($logPreds);
        
        return $product->sumAxis(1)->mean() * -1.0;
    }

    public function differentiate(Tensor $predictions, Tensor $labels): Tensor
    {
        $clipped = $predictions->clip(1e-7, 1.0 - 1e-7);
        $batchSize = (float) $predictions->shape()[0];
        
        // Formula: dY = -(y_true / y_pred) / BatchSize
        // Uses inplace mutation on the division output
        $diff = $labels->div($clipped);
        
        return $diff->mulScalarInplace(-1.0 / $batchSize);
    }
}