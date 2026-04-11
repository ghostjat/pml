<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Initializers;

use Pml\Tensor;

/**
 * Xavier / Glorot Normal Initialization — balanced for tanh/sigmoid.
 * std = sqrt(2 / (fanIn + fanOut))
 *
 * JIT & Memory Optimized: single C Gaussian fill.
 */
final class Xavier1 implements Initializer
{
    public function initialize(int $fanIn, int $fanOut): Tensor
    {
        $std = sqrt(2.0 / max(1, $fanIn + $fanOut));
        return Tensor::randomNormal([$fanIn, $fanOut], 0.0, $std);
    }
}
