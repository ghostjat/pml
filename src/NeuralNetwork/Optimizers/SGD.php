<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

use Pml\NeuralNetwork\Layers\Layer;

/**
 * Stochastic Gradient Descent (SGD) with In-Place C Mutations.
 */
final class SGD implements Optimizer
{
    private float $learningRate;

    public function __construct(float $learningRate = 0.01)
    {
        $this->learningRate = $learningRate;
    }

    public function step(array $layers): void
    {
        foreach ($layers as $layer) {
            $params = $layer->getParameters();
            $grads = $layer->getGradients();

            foreach ($params as $name => $paramTensor) {
                if (isset($grads[$name])) {
                    // W = W - (dW * lr)
                    // 1. Scale gradient by learning rate (Creates temporary C-Tensor)
                    $scaledGradient = $grads[$name]->mulScalar($this->learningRate);
                    
                    // 2. Subtract directly from the active Weight matrix IN-PLACE
                    // This guarantees zero PHP heap fragmentation during millions of training steps.
                    $paramTensor->subInplace($scaledGradient);
                    
                    // The temporary $scaledGradient tensor falls out of scope here 
                    // and its C-memory is instantly freed by __destruct().
                }
            }
        }
    }
}