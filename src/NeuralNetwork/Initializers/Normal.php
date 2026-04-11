<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Initializers;

use Pml\Tensor;

/**
 * Normal Initialization — W ~ N(mean, std).
 */
final class Normal implements Initializer
{
    public function __construct(
        private readonly float $mean = 0.0,
        private readonly float $std  = 0.05
    ) {}

    public function initialize(int $fanIn, int $fanOut): Tensor
    {
        return Tensor::randomNormal([$fanIn, $fanOut], $this->mean, $this->std);
    }
}
