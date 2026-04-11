<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Initializers;

use Pml\Tensor;

/**
 * He (Kaiming) Normal Initialization — designed for ReLU networks.
 * std = sqrt(2 / fanIn)
 *
 * JIT & Memory Optimized: single C call fills the tensor with Gaussian noise.
 */
final class He implements Initializer
{
    public function initialize(int $fanIn, int $fanOut): Tensor
    {
        $std = sqrt(2.0 / max(1, $fanIn));
        return Tensor::randomNormal([$fanIn, $fanOut], 0.0, $std);
    }
}
