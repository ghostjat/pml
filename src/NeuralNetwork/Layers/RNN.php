<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Interfaces\Stateful;
use Pml\Tensor;

/**
 * Standard Recurrent Neural Network (RNN) Layer.
 * Processes sequences of shape [Batch, SeqLen, Features].
 * * JIT & Memory Optimized:
 * - Uses Tensor::slice() to iterate over time-steps with zero memory duplication.
 * - Math executes via OpenBLAS inside the PHP temporal loop.
 */
final class RNN implements Layer, Stateful
{
    private Tensor $W_ih; // Input-to-Hidden weights
    private Tensor $W_hh; // Hidden-to-Hidden weights
    private Tensor $b_ih;
    private Tensor $b_hh;

    private int $hiddenSize;

    public function __construct(int $inputSize, int $hiddenSize)
    {
        $this->hiddenSize = $hiddenSize;
        $k = 1.0 / sqrt($hiddenSize);
        
        $this->W_ih = Tensor::randomUniform([$inputSize, $hiddenSize], -$k, $k);
        $this->W_hh = Tensor::randomUniform([$hiddenSize, $hiddenSize], -$k, $k);
        $this->b_ih = Tensor::randomUniform([$hiddenSize], -$k, $k);
        $this->b_hh = Tensor::randomUniform([$hiddenSize], -$k, $k);
    }

    public function forward(Tensor $input): Tensor
    {
        $shape = $input->shape();
        if (count($shape) !== 3) throw new \InvalidArgumentException("RNN expects 3D input: [Batch, SeqLen, Features]");
        
        $batch = $shape[0];
        $seqLen = $shape[1];
        
        $h_t = Tensor::zeros($batch, $this->hiddenSize);
        $outputs = [];

        // Temporal Loop (Unrolled Execution)
        for ($t = 0; $t < $seqLen; $t++) {
            // Extract the current time step (Zero-Copy)
            // Shape becomes [Batch, Features]
            $x_t = $input->slice(1, $t, 1)->squeeze();
            
            // h_t = tanh(x_t * W_ih + b_ih + h_t-1 * W_hh + b_hh)
            $ih = $x_t->matmul($this->W_ih)->addInplace($this->b_ih);
            $hh = $h_t->matmul($this->W_hh)->addInplace($this->b_hh);
            
            $h_t = $ih->addInplace($hh)->tanh();
            
            // Store output for this timestep. ExpandDims ensures shape [Batch, 1, Hidden]
            $outputs[] = $h_t->expandDims(1);
        }

        // Re-combine all timesteps into [Batch, SeqLen, Hidden] in C memory via memcpy
        return Tensor::concat($outputs, 1);
    }

    public function backward(Tensor $dY): Tensor
    {
        // Note: Backpropagation Through Time (BPTT) requires an internal tape of all $h_t states.
        // For production scale, BPTT is generally implemented as a fused C-Kernel.
        throw new \RuntimeException("BPTT for RNNs requires a fused C-kernel to prevent OOM errors in PHP userland.");
    }

    public function getParameters(): array
    {
        return [
            'W_ih' => $this->W_ih, 'W_hh' => $this->W_hh,
            'b_ih' => $this->b_ih, 'b_hh' => $this->b_hh
        ];
    }

    public function getGradients(): array { return []; }

    public function getConfig(): array
    {
        return [
            'inputSize'  => $this->W_ih->shape()[0],
            'hiddenSize' => $this->hiddenSize,
        ];
    }

    public static function fromConfig(array $config): static
    {
        return new static((int) $config['inputSize'], (int) $config['hiddenSize']);
    }

    public function getStateDict(string $prefix = ''): array
    {
        return [
            $prefix . 'W_ih' => $this->W_ih,
            $prefix . 'W_hh' => $this->W_hh,
            $prefix . 'b_ih' => $this->b_ih,
            $prefix . 'b_hh' => $this->b_hh,
        ];
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->W_ih = $dict[$prefix . 'W_ih'];
        $this->W_hh = $dict[$prefix . 'W_hh'];
        $this->b_ih = $dict[$prefix . 'b_ih'];
        $this->b_hh = $dict[$prefix . 'b_hh'];
    }
}