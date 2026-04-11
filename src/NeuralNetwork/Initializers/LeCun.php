<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Initializers;

use Pml\Tensor;

/**
 * LeCun Normal Initialization — designed for SELU networks.
 * std = sqrt(1 / fanIn)
 *
 * JIT & Memory Optimized: single C Gaussian fill.
 */
final class LeCun implements Initializer
{
    public function initialize(int $fanIn, int $fanOut): Tensor
    {
        $std = sqrt(1.0 / max(1, $fanIn));
        return Tensor::randomNormal([$fanIn, $fanOut], 0.0, $std);
    }
}
