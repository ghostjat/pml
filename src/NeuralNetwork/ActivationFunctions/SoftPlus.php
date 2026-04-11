<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\ActivationFunctions;

use Pml\Tensor;

/**
 * SoftPlus — smooth approximation to ReLU.
 * f(z) = log(1 + exp(z))    f'(z) = sigmoid(z)
 *
 * JIT & Memory Optimized: log1p(exp(z)) maps to a single AVX2 kernel in C.
 */
final class SoftPlus implements ActivationFunction
{
    public function activate(Tensor $z): Tensor
    {
        return $z->exp()->log1p();
    }

    public function differentiate(Tensor $z): Tensor
    {
        return $z->sigmoid();
    }
}
