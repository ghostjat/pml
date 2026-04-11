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
        // Numerical Stability: Subtract the maximum logit from the row before taking exp()
        // This prevents floating-point overflow (INF) when logits are large.
        // maxAxis(1) gets max per row, expandDims(1) makes it [Batch, 1] for broadcasting.
        $max = $input->maxAxis(1)->expandDims(1);
        $shifted = $input->sub($max);
        
        $exp = $shifted->exp();
        
        // Sum the exponentials along the row: [Batch, 1]
        $sum = $exp->sumAxis(1)->expandDims(1);
        
        // output = exp(x) / sum(exp(x))
        // Divides inplace to conserve memory during the forward pass
        $this->output = $exp->divInplace($sum);
        
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