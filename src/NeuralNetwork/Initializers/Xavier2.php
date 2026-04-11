<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Initializers;

use Pml\Tensor;

/**
 * Xavier / Glorot Uniform Initialization.
 * limit = sqrt(6 / (fanIn + fanOut));  W ~ Uniform[-limit, +limit]
 *
 * JIT & Memory Optimized: single C uniform fill.
 */
final class Xavier2 implements Initializer
{
    public function initialize(int $fanIn, int $fanOut): Tensor
    {
        $limit = sqrt(6.0 / max(1, $fanIn + $fanOut));
        return Tensor::randomUniform([$fanIn, $fanOut], -$limit, $limit);
    }
}
