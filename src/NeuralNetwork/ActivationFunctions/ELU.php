<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\ActivationFunctions;

use Pml\Tensor;

/**
 * Exponential Linear Unit (ELU).
 * f(z) = z if z > 0, else alpha*(exp(z) - 1)
 * f'(z) = 1 if z > 0, else f(z) + alpha
 *
 * JIT & Memory Optimized: where() routes between two C-computed branches in one pass.
 */
final class ELU implements ActivationFunction
{
    public function __construct(private readonly float $alpha = 1.0) {}

    public function activate(Tensor $z): Tensor
    {
        $pos = $z->relu();
        $neg = $z->exp()->addScalarInplace(-1.0)->mulScalarInplace($this->alpha);
        return $z->greaterScalar(0.0)->where($pos, $neg);
    }

    public function differentiate(Tensor $z): Tensor
    {
        $a    = $this->activate($z);
        $dPos = Tensor::ones(...$z->shape());
        $dNeg = $a->addScalarInplace($this->alpha);
        return $z->greaterScalar(0.0)->where($dPos, $dNeg);
    }
}
