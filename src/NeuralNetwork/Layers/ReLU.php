<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * Rectified Linear Unit (ReLU) Activation.
 * * Mathematical Derivation:
 * f(x) = max(0, x)
 * f'(x) = 1 if x > 0 else 0
 */
final class ReLU implements Layer
{
    private ?Tensor $input = null;

    public function forward(Tensor $input): Tensor
    {
        $this->input = $input;
        // Native C-level clipping at hardware speed
        return $input->relu(); 
    }

    public function backward(Tensor $dY): Tensor
    {
        // The derivative is a binary mask where input > 0.
        // We broadcast a scalar 0 Tensor to create the mask natively in C.
        $zero = Tensor::zeros(1);
        $derivativeMask = $this->input->greater($zero);

        // dX = dY * mask (Hadamard product)
        return $dY->mul($derivativeMask);
    }

    public function getParameters(): array { return []; }
    public function getGradients(): array { return []; }
}