<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

use Pml\NeuralNetwork\Layers\Layer;

/**
 * Base interface for Gradient Descent Optimizers.
 */
interface Optimizer
{
    /**
     * Update the trainable parameters of all layers using the calculated gradients.
     * @param Layer[] $layers
     */
    public function step(array $layers): void;
}