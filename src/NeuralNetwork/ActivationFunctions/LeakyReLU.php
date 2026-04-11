<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\ActivationFunctions;

use Pml\Tensor;

/**
 * Leaky ReLU — f(z) = z if z > 0, else leakage*z.
 * Prevents dying neurons by allowing a small gradient for z < 0.
 *
 * JIT & Memory Optimized: where() picks between z and leakage*z in one C pass.
 */
final class LeakyReLU implements ActivationFunction
{
    public function __construct(private readonly float $leakage = 0.1) {}

    public function activate(Tensor $z): Tensor
    {
        return $z->greaterScalar(0.0)->where($z, $z->mulScalar($this->leakage));
    }

    public function differentiate(Tensor $z): Tensor
    {
        $ones = Tensor::ones(...$z->shape());
        $leak = Tensor::zeros(...$z->shape())->addScalarInplace($this->leakage);
        return $z->greaterScalar(0.0)->where($ones, $leak);
    }
}
