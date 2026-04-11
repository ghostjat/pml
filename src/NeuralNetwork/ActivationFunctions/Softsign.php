<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\ActivationFunctions;

use Pml\Tensor;

/**
 * Softsign — f(z) = z / (1 + |z|)   f'(z) = 1 / (1 + |z|)^2
 *
 * JIT & Memory Optimized: denom computed once in C; two in-place ops.
 */
final class Softsign implements ActivationFunction
{
    public function activate(Tensor $z): Tensor
    {
        $denom = $z->abs()->addScalarInplace(1.0);
        return $z->div($denom);
    }

    public function differentiate(Tensor $z): Tensor
    {
        $denom = $z->abs()->addScalarInplace(1.0);
        return $denom->square()->pow(
            Tensor::zeros(...$z->shape())->addScalarInplace(-1.0)
        );
    }
}
