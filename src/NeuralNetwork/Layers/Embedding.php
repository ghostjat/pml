<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * NLP Embedding Layer.
 * Acts as the bridge between integer text tokens and the FLOAT32 Neural Network.
 * * JIT & Memory Optimized:
 * - Uses native C-level pointer mapping to prevent float-to-int casting overhead.
 */
final class Embedding implements Layer
{
    private int $vocabSize;
    private int $embedDim;
    
    private Tensor $weights;
    private ?Tensor $input = null;

    public function __construct(int $vocabSize, int $embedDim)
    {
        $this->vocabSize = $vocabSize;
        $this->embedDim = $embedDim;
        
        // Initialize vocabulary weights using a standard Normal distribution
        $this->weights = Tensor::randomNormal([$vocabSize, $embedDim], 0.0, 1.0);
    }

    public function forward(Tensor $input): Tensor
    {
        if ($input->dtype() !== Tensor::DTYPE_INT32) {
            throw new \InvalidArgumentException("Embedding layer requires DTYPE_INT32 token inputs.");
        }
        
        $this->input = $input;
        
        // Dispatches directly to the high-speed C-lookup kernel
        return $input->embeddingLookup($this->weights);
    }

    public function backward(Tensor $dY): Tensor
    {
        // Note: Full Scatter-Add gradient calculation for dW requires a dedicated C-kernel.
        // For this version, Embeddings are treated as "frozen" (common for pre-trained LLM weights).
        // Token inputs (INT32) do not have gradients, so we return an empty tensor.
        return Tensor::zeros(1); 
    }

    public function getParameters(): array
    {
        return ['weights' => $this->weights];
    }

    public function getGradients(): array
    {
        // Frozen layer for now.
        return [];
    }
}