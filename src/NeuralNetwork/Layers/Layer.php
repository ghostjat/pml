<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * Base interface for all Deep Learning layers.
 * Strictly typed to ensure inputs and gradients never leave C-memory.
 */
interface Layer
{
    /**
     * Compute the forward pass of the layer.
     * Must cache any necessary C-pointers (like $input) for the backward pass.
     */
    public function forward(Tensor $input): Tensor;

    /**
     * Compute the backward pass (Backpropagation).
     * Calculates internal gradients (dW, db) and returns the gradient with respect to the input (dX).
     */
    public function backward(Tensor $dY): Tensor;

    /**
     * Return associative array of trainable parameters (e.g., ['weights' => Tensor, 'bias' => Tensor]).
     * Returns empty array if the layer has no trainable parameters (like Activations).
     */
    public function getParameters(): array;

    /**
     * Return associative array of computed gradients (e.g., ['weights' => Tensor, 'bias' => Tensor]).
     */
    public function getGradients(): array;
}