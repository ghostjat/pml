<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * Sigmoid Activation Layer.
 * * Mathematical Derivation:
 * f(x) = 1 / (1 + exp(-x))
 * f'(x) = f(x) * (1 - f(x))
 */
final class Sigmoid implements Layer
{
    private ?Tensor $output = null;

    public function forward(Tensor $input): Tensor
    {
        // We cache the output instead of the input because the derivative 
        // is calculated efficiently using the forward pass output.
        $this->output = $input->sigmoid();
        return $this->output;
    }

    public function backward(Tensor $dY): Tensor
    {
        if ($this->output === null) {
            throw new \RuntimeException("Backward called before forward pass.");
        }

        // dX = dY * (sig * (1 - sig))
        // Evaluates dynamically in C-memory leveraging inplace math
        $oneMinusSig = $this->output->mulScalar(-1.0)->addScalarInplace(1.0);
        $derivative = $this->output->mul($oneMinusSig);
        
        return $dY->mulInplace($derivative);
    }

    public function getParameters(): array { return []; }
    public function getGradients(): array { return []; }
}