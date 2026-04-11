<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\ActivationFunctions;

use Pml\Tensor;

/**
 * Scaled ELU (SELU) — self-normalizing activation.
 * f(z) = scale * (z if z > 0, else alpha*(exp(z)-1))
 * Constants chosen so the distribution converges to mean=0, var=1.
 *
 * JIT & Memory Optimized: exact same pattern as ELU, just two scalar multiplies added.
 */
final class SELU implements ActivationFunction
{
    private const ALPHA = 1.6732632423543772;
    private const SCALE = 1.0507009873554805;

    public function activate(Tensor $z): Tensor
    {
        $pos = $z->relu();
        $neg = $z->exp()->addScalarInplace(-1.0)->mulScalarInplace(self::ALPHA);
        return $z->greaterScalar(0.0)->where($pos, $neg)->mulScalarInplace(self::SCALE);
    }

    public function differentiate(Tensor $z): Tensor
    {
        $dPos = Tensor::ones(...$z->shape())->mulScalarInplace(self::SCALE);
        $dNeg = $z->exp()->mulScalarInplace(self::ALPHA * self::SCALE);
        return $z->greaterScalar(0.0)->where($dPos, $dNeg);
    }
}
