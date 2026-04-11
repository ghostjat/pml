<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;
use InvalidArgumentException;

/**
 * Multi-Layer Perceptron (MLP) Composite Layer.
 * Acts as a highly optimized feed-forward block commonly used in Transformers and Deep Feature extractors.
 * * JIT & Memory Optimized:
 * - Orchestrates multiple Dense layers internally with zero memory overhead.
 * - Flattens parameter and gradient retrieval dynamically for the Optimizer.
 */
final class MLP implements Layer
{
    /** @var Layer[] */
    private array $layers = [];

    /**
     * @param int $inputSize The feature dimension of the input data.
     * @param array<int> $hiddenSizes An array defining the size of each hidden Dense layer.
     * @param int $outputSize The final output feature dimension.
     * @param string $activation The activation function to use between hidden layers (relu, sigmoid, tanh).
     */
    public function __construct(int $inputSize, array $hiddenSizes, int $outputSize, string $activation = 'relu')
    {
        $currentSize = $inputSize;
        
        // Build the deep hidden layers
        foreach ($hiddenSizes as $hiddenSize) {
            $this->layers[] = new Dense($currentSize, $hiddenSize);
            $this->layers[] = $this->createActivation($activation);
            $currentSize = $hiddenSize;
        }
        
        // Final projection layer (standard MLPs end with a linear projection / no activation)
        $this->layers[] = new Dense($currentSize, $outputSize);
    }

    private function createActivation(string $name): Layer
    {
        return match (strtolower($name)) {
            'relu'    => new ReLU(),
            'sigmoid' => new Sigmoid(),
            'tanh'    => new Tanh(),
            default   => throw new InvalidArgumentException("Unsupported activation function: {$name}"),
        };
    }

    public function forward(Tensor $input): Tensor
    {
        $current = $input;
        // The pointer is passed seamlessly through the C-memory sequence
        foreach ($this->layers as $layer) {
            $current = $layer->forward($current);
        }
        return $current;
    }

    public function backward(Tensor $dY): Tensor
    {
        $currentGradient = $dY;
        // Chain rule executed in perfect reverse order
        for ($i = count($this->layers) - 1; $i >= 0; $i--) {
            $currentGradient = $this->layers[$i]->backward($currentGradient);
        }
        return $currentGradient;
    }

    public function getParameters(): array
    {
        $params = [];
        foreach ($this->layers as $i => $layer) {
            foreach ($layer->getParameters() as $name => $tensor) {
                // Flatten the multi-layer parameters into a unique 1D associative array for the Optimizer
                $params["layer_{$i}_{$name}"] = $tensor;
            }
        }
        return $params;
    }

    public function getGradients(): array
    {
        $grads = [];
        foreach ($this->layers as $i => $layer) {
            foreach ($layer->getGradients() as $name => $tensor) {
                $grads["layer_{$i}_{$name}"] = $tensor;
            }
        }
        return $grads;
    }
}