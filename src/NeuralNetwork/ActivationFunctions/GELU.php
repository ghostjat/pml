<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\ActivationFunctions;

use Pml\Tensor;

/**
 * Gaussian Error Linear Unit (GELU).
 * f(z) ≈ 0.5 * z * (1 + tanh(sqrt(2/π) * (z + 0.044715*z^3)))
 *
 * JIT & Memory Optimized: cubic term and tanh computed in C; no PHP scalar loop.
 */
final class GELU implements ActivationFunction
{
    private const C1 = 0.7978845608;   // sqrt(2/π)
    private const C2 = 0.044715;

    public function activate(Tensor $z): Tensor
    {
        // inner = C1 * (z + C2 * z^3)
        $inner = $z->square()->mul($z)->mulScalarInplace(self::C2)->addInplace($z)->mulScalarInplace(self::C1);
        return $z->mul($inner->tanh()->addScalarInplace(1.0))->mulScalarInplace(0.5);
    }

    public function differentiate(Tensor $z): Tensor
    {
        $inner = $z->square()->mul($z)->mulScalarInplace(self::C2)->addInplace($z)->mulScalarInplace(self::C1);
        $tanh  = $inner->tanh();
        // d/dz ≈ 0.5*(1+tanh(inner)) + 0.5*z*(1-tanh^2)*C1*(1+3*C2*z^2)
        $sech2 = Tensor::ones(...$z->shape())->subInplace($tanh->square());
        $dInner = $z->square()->mulScalarInplace(3.0 * self::C2)->addScalarInplace(1.0)->mulScalarInplace(self::C1);
        return $tanh->addScalarInplace(1.0)->mulScalarInplace(0.5)
               ->addInplace($z->mul($sech2)->mul($dInner)->mulScalarInplace(0.5));
    }
}
