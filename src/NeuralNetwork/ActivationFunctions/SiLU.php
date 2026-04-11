<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\ActivationFunctions;

use Pml\Tensor;

/**
 * Sigmoid Linear Unit (SiLU / Swish-1).
 * f(z) = z * sigmoid(z)
 * f'(z) = sigmoid(z) * (1 + z*(1 - sigmoid(z)))
 *
 * JIT & Memory Optimized: sigmoid computed once and reused for derivative.
 */
final class SiLU implements ActivationFunction
{
    public function activate(Tensor $z): Tensor
    {
        return $z->mul($z->sigmoid());
    }

    public function differentiate(Tensor $z): Tensor
    {
        $sig = $z->sigmoid();
        // sig * (1 + z*(1 - sig))
        $one = Tensor::ones(...$z->shape());
        return $sig->mul($one->addInplace($z->mul($one->sub($sig))));
    }
}
