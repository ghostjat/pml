<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * Hyperbolic Tangent (Tanh) Activation Layer.
 * * Mathematical Derivation:
 * f(x) = tanh(x)
 * f'(x) = 1 - tanh(x)^2
 */
final class Tanh implements Layer
{
    private ?Tensor $output = null;

    public function forward(Tensor $input): Tensor
    {
        // Cache the output for derivative calculation
        $this->output = $input->tanh();
        return $this->output;
    }

    public function backward(Tensor $dY): Tensor
    {
        if ($this->output === null) {
            throw new \RuntimeException("Backward called before forward pass.");
        }

        // dX = dY * (1 - tanh^2)
        // Highly optimized C-level sequence avoiding PHP allocations
        $squared = $this->output->square();
        $derivative = $squared->mulScalar(-1.0)->addScalarInplace(1.0);
        
        return $dY->mulInplace($derivative);
    }

    public function getParameters(): array { return []; }
    public function getGradients(): array { return []; }
}