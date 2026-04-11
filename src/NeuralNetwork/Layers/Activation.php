<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;
use Pml\NeuralNetwork\ActivationFunctions\ActivationFunction;

/**
 * Generic Activation Layer — wraps any ActivationFunction in the Layer interface.
 * Stores pre-activation z for the backward pass.
 *
 * JIT & Memory Optimized:
 * - Caches z as a reference, not a copy — zero extra C allocation.
 * - Backward calls differentiate() which stays in C.
 */
final class Activation implements Layer
{
    private ?Tensor $z = null;

    public function __construct(private readonly ActivationFunction $fn) {}

    public function forward(Tensor $x, bool $training = true): Tensor
    {
        $this->z = $x;                    // keep reference (no copy)
        return $this->fn->activate($x);
    }

    public function backward(Tensor $dOut, Tensor $x): Tensor
    {
        $dz = $this->fn->differentiate($this->z ?? $x);
        return $dOut->mul($dz);
    }

    public function params(): array { return []; }
}
