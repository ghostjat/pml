<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * Dense (Fully Connected) Linear Layer.
 * Mathematical operation: Y = XW + b
 * * JIT & Memory Optimized:
 * - Employs OpenBLAS for massive O(N^3) Matmuls.
 * - Uses Stride=0 Broadcasting for bias addition.
 * - Backprop utilizes Zero-Copy matrix transpositions.
 */
final class Dense implements Layer
{
    private Tensor $weights;
    private ?Tensor $bias;
    
    // Cached states for Backpropagation
    private ?Tensor $input = null;
    private ?Tensor $dW = null;
    private ?Tensor $dbias = null;

    public function __construct(int $inputDim, int $outputDim, bool $useBias = true)
    {
        // He Initialization (Optimized for ReLU)
        $stddev = sqrt(2.0 / $inputDim);
        $this->weights = Tensor::randomNormal([$inputDim, $outputDim], 0.0, $stddev);
        
        $this->bias = $useBias ? Tensor::zeros(1, $outputDim) : null;
    }

    public function forward(Tensor $input): Tensor
    {
        // Cache the input pointer for the backward pass (No memory is copied)
        $this->input = $input;

        // Y = X * W
        $output = $input->matmul($this->weights);

        // Y = Y + b (Addition applies Stride=0 AVX2 Broadcasting automatically)
        if ($this->bias !== null) {
            $output->addInplace($this->bias);
        }

        return $output;
    }

    public function backward(Tensor $dY): Tensor
    {
        if ($this->input === null) {
            throw new \RuntimeException("Backward pass called before forward pass.");
        }

        // 1. Gradient w.r.t Bias: db = sum(dY, axis=0)
        if ($this->bias !== null) {
            $this->dbias = $dY->sumAxis(0);
        }

        // 2. Gradient w.r.t Weights: dW = X^T * dY
        // transpose() is a <0.01ms Zero-Copy View modification
        $inputT = $this->input->transpose();
        $this->dW = $inputT->matmul($dY);

        // 3. Gradient w.r.t Input (to pass to the previous layer): dX = dY * W^T
        $weightsT = $this->weights->transpose();
        $dX = $dY->matmul($weightsT);

        return $dX;
    }

    public function getParameters(): array
    {
        $params = ['weights' => $this->weights];
        if ($this->bias !== null) {
            $params['bias'] = $this->bias;
        }
        return $params;
    }

    public function getGradients(): array
    {
        $grads = ['weights' => $this->dW];
        if ($this->dbias !== null) {
            $grads['bias'] = $this->dbias;
        }
        return $grads;
    }
}