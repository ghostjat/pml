<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;
use Pml\NeuralNetwork\ActivationFunctions\ActivationFunction;
use RuntimeException;

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

    public function forward(Tensor $input): Tensor
    {
        // Cache the input pointer for the backward pass (Zero-Copy reference)
        $this->z = $input;
        
        return $this->fn->activate($input);
    }

    public function backward(Tensor $dY): Tensor
    {
        if ($this->z === null) {
            throw new RuntimeException("Backward pass called before forward pass.");
        }

        // 1. Compute derivative of the activation function: f'(z)
        $dz = $this->fn->differentiate($this->z);

        // 2. Chain rule: dY * f'(z) (Element-wise multiplication in C)
        return $dY->mul($dz);
    }

    public function getParameters(): array
    {
        // Activation functions have no trainable weights or biases
        return []; 
    }

    public function getGradients(): array
    {
        // No trainable parameters means no gradients to update
        return []; 
    }
}