<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * Softmax Activation Layer.
 * Converts a vector of raw logits into a probability distribution that sums to 1.0.
 * * JIT & Memory Optimized:
 * - Employs the "Max-Shift" numerical stability trick natively in C.
 * - Backpropagation uses highly vectorized Jacobian broadcasting instead of PHP loops.
 */
final class Softmax implements Layer
{
    private ?Tensor $output = null;

    public function forward(Tensor $input): Tensor
    {
        // One copy + one in-place C kernel (max-shift + exp + rowsum + divide).
        // Replaces 6 intermediate tensors with 1 memcpy.
        $this->output = $input->copy();
        $this->output->softmaxInplace();
        return $this->output;
    }

    public function backward(Tensor $dY): Tensor
    {
        if ($this->output === null) {
            throw new \RuntimeException("Backward called before forward pass.");
        }

        // The exact Jacobian derivative of Softmax multiplied by the upstream gradient dY.
        // dX = Output * (dY - sum(dY * Output, axis=1))
        
        // 1. dY * Output (Hadamard product)
        $dyDotOut = $dY->mul($this->output);
        
        // 2. Sum along the rows to get the scalar correction per batch: [Batch, 1]
        $sumDyDotOut = $dyDotOut->sumAxis(1)->expandDims(1);
        
        // 3. dY - sum
        $diff = $dY->sub($sumDyDotOut);
        
        // 4. dX
        return $this->output->mul($diff);
    }

    public function getParameters(): array { return []; }
    public function getGradients(): array { return []; }
}