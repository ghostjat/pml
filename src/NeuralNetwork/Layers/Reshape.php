<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

final class Reshape implements Layer
{
    private array $targetShape;
    private ?array $cacheInputShape = null;

    public function __construct(array $targetShape)
    {
        // e.g., [50, 128] for [SeqLen, dModel]
        $this->targetShape = $targetShape;
    }

    public function forward(Tensor $input): Tensor
    {
        // 1. Detach from Dataset zero-copy view
        $input = $input->copy();
        
        $this->cacheInputShape = $input->shape();
        $B = $this->cacheInputShape[0];
        $originalD = $this->cacheInputShape[1] ?? 1;
        
        $expectedD = array_product($this->targetShape);
        
        // 2. Hardware-level memory adjustments (Padding or Trimming)
        if ($originalD !== $expectedD) {
            if ($originalD > $expectedD) {
                // Dataset is too large: C-Level slice to trim excess
                $input = $input->slice(1, 0, $expectedD)->copy();
            } else {
                // Dataset is too small (e.g., 5126 < 6400): C-Level Padding
                $missingColumns = $expectedD - $originalD;
                
                // Allocate a zero tensor and concat on the GPU/CPU level
                $padding = Tensor::zeros($B, $missingColumns);
                $input = Tensor::concat([$input, $padding], 1);
            }
        }
        
        // 3. Safe hardware reshape
        return $input->reshape($B, ...$this->targetShape);
    }

    public function backward(Tensor $dY): Tensor
    {
        $dY = $dY->copy();
        
        $B = $this->cacheInputShape[0];
        $originalD = $this->cacheInputShape[1] ?? 1;
        $expectedD = array_product($this->targetShape);

        // 1. Flatten the gradient back to 2D
        $flatGrad = $dY->reshape($B, $expectedD);

        // 2. Reverse the memory adjustments to match upstream layer/dataset
        if ($originalD !== $expectedD) {
            if ($originalD < $expectedD) {
                // If we padded the input forward, we must slice off the padding gradient backward
                $flatGrad = $flatGrad->slice(1, 0, $originalD)->copy();
            } else {
                // If we trimmed the input forward, we must restore the missing gradient with zeros
                $missing = $originalD - $expectedD;
                $padding = Tensor::zeros($B, $missing);
                $flatGrad = Tensor::concat([$flatGrad, $padding], 1);
            }
        }

        return $flatGrad;
    }

    public function getParameters(): array { return []; }
    public function getGradients(): array { return []; }
}