<?php

declare(strict_types=1);

namespace Pml\Losses;

use Pml\Tensor;

/**
 * Hinge Loss.
 * Standard loss function used for training Support Vector Machines (SVMs).
 * Expects targets to be mapped to {-1, 1}.
 */
final class Hinge implements Loss
{
    public function compute(Tensor $predictions, Tensor $labels): float
    {
        // Loss = max(0, 1 - y_true * y_pred)
        $margin = $labels->mul($predictions);
        $loss = $margin->mulScalar(-1.0)->addScalarInplace(1.0)->relu();
        
        return $loss->mean();
    }

    public function differentiate(Tensor $predictions, Tensor $labels): Tensor
    {
        $n = (float) $predictions->size();
        $margin = $labels->mul($predictions);

        // Mask where margin < 1
        $one = Tensor::zeros(1)->addScalarInplace(1.0);
        $mask = $margin->less($one);

        // dY = -y_true if margin < 1 else 0
        $dY = $labels->mulScalar(-1.0)->mulInplace($mask);

        return $dY->mulScalarInplace(1.0 / $n);
    }
}