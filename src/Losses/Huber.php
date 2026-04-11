<?php

declare(strict_types=1);

namespace Pml\Losses;

use Pml\Tensor;

/**
 * Huber Loss.
 * A robust regression loss function that acts like MSE for small errors and MAE for large errors (outliers).
 * * JIT & Memory Optimized:
 * - Uses zero-copy AVX2 where() masking to switch between L1 and L2 penalties seamlessly.
 */
final class Huber implements Loss
{
    private float $delta;

    public function __construct(float $delta = 1.0)
    {
        $this->delta = $delta;
    }

    public function compute(Tensor $predictions, Tensor $labels): float
    {
        $diff = $predictions->sub($labels);
        $absDiff = $diff->abs();

        $deltaT = Tensor::zeros(1)->addScalarInplace($this->delta);
        $mask = $absDiff->lessEqual($deltaT);

        // L2 Loss: 0.5 * diff^2
        $l2 = $diff->square()->mulScalarInplace(0.5);

        // L1 Loss: delta * absDiff - 0.5 * delta^2
        $l1 = $absDiff->mulScalar($this->delta)->addScalarInplace(-0.5 * $this->delta * $this->delta);

        // Merge using Boolean Mask
        return $mask->where($l2, $l1)->mean();
    }

    public function differentiate(Tensor $predictions, Tensor $labels): Tensor
    {
        $diff = $predictions->sub($labels);
        $absDiff = $diff->abs();
        $n = (float) $predictions->size();

        $deltaT = Tensor::zeros(1)->addScalarInplace($this->delta);
        $mask = $absDiff->lessEqual($deltaT);

        // dL2 = diff
        $dl2 = $diff->copy();

        // dL1 = delta * sign(diff)
        $dl1 = $diff->sign()->mulScalarInplace($this->delta);

        return $mask->where($dl2, $dl1)->mulScalarInplace(1.0 / $n);
    }
}