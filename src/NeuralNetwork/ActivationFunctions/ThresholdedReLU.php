<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\ActivationFunctions;

use Pml\Tensor;

/**
 * Thresholded ReLU — f(z) = z if z > theta, else 0.
 * Allows the activation threshold to be tuned (default theta=1.0).
 *
 * JIT & Memory Optimized: greaterScalar mask in C; where() picks z vs 0 in one pass.
 */
final class ThresholdedReLU implements ActivationFunction
{
    public function __construct(private readonly float $theta = 1.0) {}

    public function activate(Tensor $z): Tensor
    {
        $zero = Tensor::zeros(...$z->shape());
        return $z->greaterScalar($this->theta)->where($z, $zero);
    }

    public function differentiate(Tensor $z): Tensor
    {
        $zero = Tensor::zeros(...$z->shape());
        $one  = Tensor::ones(...$z->shape());
        return $z->greaterScalar($this->theta)->where($one, $zero);
    }
}
