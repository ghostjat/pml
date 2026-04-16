<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Interfaces\Stateful;
use Pml\Tensor;

/**
 * Long Short-Term Memory (LSTM) Layer.
 * Advanced Sequence processing utilizing 4 independent gating mechanisms.
 * * JIT & Memory Optimized:
 * - Groups gates sequentially utilizing OpenBLAS GEMM matrix multiplications.
 * - Extremely heavy C-level computations managed seamlessly via PHP orchestration.
 */
final class LSTM implements Layer, Stateful
{
    private int $hiddenSize;

    // Weights (Combined into matrices for performance: [InputSize, 4 * HiddenSize])
    private Tensor $W_ih;
    private Tensor $W_hh;
    private Tensor $b_ih;
    private Tensor $b_hh;

    public function __construct(int $inputSize, int $hiddenSize)
    {
        $this->hiddenSize = $hiddenSize;
        $k = 1.0 / sqrt($hiddenSize);
        
        // PyTorch groups the 4 gates (Input, Forget, Cell, Output) into single large matrices
        // to compute all 4 gates in a single massive OpenBLAS multiplication.
        $this->W_ih = Tensor::randomUniform([$inputSize, 4 * $hiddenSize], -$k, $k);
        $this->W_hh = Tensor::randomUniform([$hiddenSize, 4 * $hiddenSize], -$k, $k);
        $this->b_ih = Tensor::randomUniform([4 * $hiddenSize], -$k, $k);
        $this->b_hh = Tensor::randomUniform([4 * $hiddenSize], -$k, $k);
    }

    public function forward(Tensor $input): Tensor
    {
        $shape = $input->shape();
        $batch = $shape[0];
        $seqLen = $shape[1];
        
        $h_t = Tensor::zeros($batch, $this->hiddenSize);
        $c_t = Tensor::zeros($batch, $this->hiddenSize);
        $outputs = [];

        for ($t = 0; $t < $seqLen; $t++) {
            $x_t = $input->slice(1, $t, 1)->squeeze();
            
            // 1. Compute all 4 gates in one massive Matrix Multiplication
            // gates = (x_t * W_ih + b_ih) + (h_t-1 * W_hh + b_hh)
            $gates_x = $x_t->matmul($this->W_ih)->addInplace($this->b_ih);
            $gates_h = $h_t->matmul($this->W_hh)->addInplace($this->b_hh);
            $gates = $gates_x->addInplace($gates_h); // Shape: [Batch, 4 * HiddenSize]
            
            // 2. Slice the massive matrix into the 4 distinct gates (Zero-Copy slices)
            $h = $this->hiddenSize;
            $i_gate = $gates->slice(1, 0, $h)->sigmoid();       // Input gate
            $f_gate = $gates->slice(1, $h, $h)->sigmoid();       // Forget gate
            $g_gate = $gates->slice(1, $h * 2, $h)->tanh();      // Cell gate (modulation)
            $o_gate = $gates->slice(1, $h * 3, $h)->sigmoid();   // Output gate
            
            // 3. Update Cell State: c_t = f * c_t-1 + i * g
            $c_t = $f_gate->mulInplace($c_t)->addInplace( $i_gate->mulInplace($g_gate) );
            
            // 4. Update Hidden State: h_t = o * tanh(c_t)
            $h_t = $o_gate->mulInplace( $c_t->tanh() );
            
            $outputs[] = $h_t->expandDims(1);
        }

        return Tensor::concat($outputs, 1);
    }

    public function backward(Tensor $dY): Tensor
    {
        throw new \RuntimeException("BPTT for LSTMs requires a fused C-kernel to prevent OOM errors.");
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