<?php

declare(strict_types=1);

namespace Pml\Losses;

use Pml\Tensor;

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
        
        // FIXED: Added epsilon scalar to the denominator to prevent gradient explosion
        $denominator = $clipped->addScalar(1e-8);
        $diff = $labels->div($denominator);
        
        return $diff->mulScalarInplace(-1.0 / $batchSize);
    }
}