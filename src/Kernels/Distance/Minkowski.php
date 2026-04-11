<?php

declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;

/**
 * Minkowski Distance.
 * The generalized L-p norm distance metric.
 * (p=1 is Manhattan, p=2 is Euclidean).
 */
final class Minkowski implements Distance
{
    private float $p;

    public function __construct(float $p = 3.0)
    {
        $this->p = $p;
    }

    public function compute(Tensor $a, Tensor $b): Tensor
    {
        // Distance = sum(|A - B|^p, axis=1) ^ (1/p)
        $pTensor = Tensor::zeros(1)->addScalarInplace($this->p);
        $invPTensor = Tensor::zeros(1)->addScalarInplace(1.0 / $this->p);

        $absDiff = $b->sub($a)->abs();
        
        return $absDiff->pow($pTensor)->sumAxis(1)->pow($invPTensor);
    }
}