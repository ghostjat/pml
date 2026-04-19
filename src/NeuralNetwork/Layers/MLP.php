<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Interfaces\Stateful;
use Pml\Tensor;
use InvalidArgumentException;

/**
 * Multi-Layer Perceptron (MLP) Composite Layer.
 * Acts as a highly optimized feed-forward block commonly used in Transformers and Deep Feature extractors.
 * * JIT & Memory Optimized:
 * - Orchestrates multiple Dense layers internally with zero memory overhead.
 * - Flattens parameter and gradient retrieval dynamically for the Optimizer.
 * - Implements Stateful so Sequential::save() uses the SafeTensors path (no PHP serialize of CData).
 */
final class MLP implements Layer, Stateful
{
    /** @var Layer[] */
    private array $layers = [];

    // Stored for getConfig() / fromConfig() reconstruction
    private int    $inputSize;
    private array  $hiddenSizes;
    private int    $outputSize;
    private string $activation;

    /**
     * @param int        $inputSize   The feature dimension of the input data.
     * @param array<int> $hiddenSizes An array defining the size of each hidden Dense layer.
     * @param int        $outputSize  The final output feature dimension.
     * @param string     $activation  Activation between hidden layers (relu, sigmoid, tanh).
     */
    public function __construct(int $inputSize, array $hiddenSizes, int $outputSize, string $activation = 'relu')
    {
        $this->inputSize  = $inputSize;
        $this->hiddenSizes = $hiddenSizes;
        $this->outputSize = $outputSize;
        $this->activation = $activation;

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

    // =========================================================================
    // Stateful — SafeTensors checkpoint interface (delegates to Dense sub-layers)
    // =========================================================================

    public function getStateDict(string $prefix = ''): array
    {
        $dict = [];
        foreach ($this->layers as $i => $layer) {
            if ($layer instanceof Stateful) {
                foreach ($layer->getStateDict("{$prefix}layer_{$i}.") as $key => $tensor) {
                    $dict[$key] = $tensor;
                }
            }
        }
        return $dict;
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        foreach ($this->layers as $i => $layer) {
            if ($layer instanceof Stateful) {
                $layer->loadStateDict($dict, "{$prefix}layer_{$i}.");
            }
        }
    }

    // =========================================================================
    // Config — JSON-safe descriptor for checkpoint rebuild
    // =========================================================================

    public function getConfig(): array
    {
        return [
            'inputSize'   => $this->inputSize,
            'hiddenSizes' => $this->hiddenSizes,
            'outputSize'  => $this->outputSize,
            'activation'  => $this->activation,
        ];
    }

    public static function fromConfig(array $config): static
    {
        return new static(
            (int)    $config['inputSize'],
            (array)  $config['hiddenSizes'],
            (int)    $config['outputSize'],
            (string) ($config['activation'] ?? 'relu')
        );
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