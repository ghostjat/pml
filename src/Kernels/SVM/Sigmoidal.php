<?php
declare(strict_types=1);

namespace Pml\Kernels\SVM;

use Pml\Tensor;

/**
 * Sigmoidal (Hyperbolic Tangent) SVM Kernel.
 * K(a, b) = tanh(gamma * a·b + coef0)
 */
final class Sigmoidal implements Kernel
{
    public function __construct(
        private readonly float $gamma = 0.001,
        private readonly float $coef0 = 0.0
    ) {}

    public function compute(Tensor $a, Tensor $b): Tensor
    {
        // 1. Compute dot product matrix: a * b^T
        $dot = $a->matmul($b->transpose());
        
        // 2. Apply tanh element-wise: tanh(gamma * dot + coef0)
        $dot->mulScalarInplace($this->gamma)->addScalarInplace($this->coef0);
        
        return $dot->tanh();
    }
}