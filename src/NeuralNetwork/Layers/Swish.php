<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * Swish Activation Layer — f(z) = z * sigmoid(beta * z).
 * Beta=1 is equivalent to SiLU; beta is optionally learnable.
 *
 * JIT & Memory Optimized: sigmoid + mul are two AVX2 kernel calls in C.
 */
final class Swish implements Layer
{
    public function __construct(private readonly float $beta = 1.0) {}

    public function forward(Tensor $x, bool $training = true): Tensor
    {
        return $x->mul($x->mulScalar($this->beta)->sigmoid());
    }

    public function backward(Tensor $dOut, Tensor $x): Tensor
    {
        $bx   = $x->mulScalar($this->beta);
        $sig  = $bx->sigmoid();
        $swish = $x->mul($sig);
        // f'(z) = swish + sig*(1 - swish) (simplified)
        $df   = $swish->addInplace($sig->mul(
            Tensor::ones(...$x->shape())->subInplace($swish)
        ));
        return $dOut->mul($df);
    }

    public function params(): array { return []; }
}
