<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Initializers;

use Pml\Tensor;

/**
 * Uniform Initialization — W ~ Uniform[min, max].
 */
final class Uniform implements Initializer
{
    public function __construct(
        private readonly float $min = -0.05,
        private readonly float $max = 0.05
    ) {}

    public function initialize(int $fanIn, int $fanOut): Tensor
    {
        return Tensor::randomUniform([$fanIn, $fanOut], $this->min, $this->max);
    }
}
