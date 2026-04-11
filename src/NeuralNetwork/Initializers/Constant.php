<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Initializers;

use Pml\Tensor;

/**
 * Constant Initialization — fills the weight tensor with a fixed value.
 * Mainly used for bias initialization (e.g. all-zeros or all-ones).
 */
final class Constant implements Initializer
{
    public function __construct(private readonly float $value = 0.0) {}

    public function initialize(int $fanIn, int $fanOut): Tensor
    {
        return Tensor::zeros($fanIn, $fanOut)->addScalarInplace($this->value);
    }
}
