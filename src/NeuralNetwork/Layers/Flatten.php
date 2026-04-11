<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * Flattens multi-dimensional inputs (like CNN feature maps) into 2D matrices [Batch, Features].
 * * JIT & Memory Optimized:
 * - 100% Zero-Copy. Reshapes alter the C-struct metadata without touching the data payload.
 */
final class Flatten implements Layer
{
    private ?array $inputShape = null;

    public function forward(Tensor $input): Tensor
    {
        $this->inputShape = $input->shape();
        
        $batchSize = $this->inputShape[0];
        $flatFeatures = array_product(array_slice($this->inputShape, 1));
        
        // Zero-copy reshape
        return $input->reshape($batchSize, $flatFeatures);
    }

    public function backward(Tensor $dY): Tensor
    {
        if ($this->inputShape === null) {
            throw new \RuntimeException("Backward called before forward.");
        }
        
        // Zero-copy reshape back to the original CNN spatial dimensions
        return $dY->reshape(...$this->inputShape);
    }

    public function getParameters(): array { return []; }
    public function getGradients(): array { return []; }
}