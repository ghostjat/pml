<?php

declare(strict_types=1);

namespace Pml\Losses;

use Pml\Tensor;

/**
 * Interface for all Loss Functions used in optimization (Neural Networks / Gradient Descent).
 */
interface Loss
{
    /**
     * Compute the scalar loss over the entire batch.
     */
    public function compute(Tensor $predictions, Tensor $labels): float;

    /**
     * Calculate the gradient (dY) of the loss with respect to the predictions.
     * This is the signal passed backward through the neural network.
     * * @return Tensor A continuous C-memory pointer containing the gradients.
     */
    public function differentiate(Tensor $predictions, Tensor $labels): Tensor;
}