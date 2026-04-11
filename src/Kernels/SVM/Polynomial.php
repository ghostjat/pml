<?php

declare(strict_types=1);

namespace Pml\Kernels\SVM;

use Pml\Tensor;

/**
 * Polynomial Kernel.
 * Maps inputs into a polynomial feature space: (gamma * (A * B^T) + c)^degree.
 */
final class Polynomial implements Kernel
{
    private int $degree;
    private float $gamma;
    private float $c;

    public function __construct(int $degree = 3, float $gamma = 1.0, float $c = 1.0)
    {
        $this->degree = $degree;
        $this->gamma = $gamma;
        $this->c = $c;
    }

    public function compute(Tensor $a, Tensor $b): Tensor
    {
        // 1. A * B^T
        $dot = $a->matmul($b->transpose());
        
        // 2. gamma * (A * B^T) + c
        $dot->mulScalarInplace($this->gamma)->addScalarInplace($this->c);
        
        // 3. (...) ^ degree
        $degreeT = Tensor::zeros(1)->addScalarInplace((float)$this->degree);
        
        return $dot->pow($degreeT);
    }
}