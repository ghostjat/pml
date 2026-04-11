<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Initializers;

use Pml\Tensor;

/**
 * Weight initializer interface.
 * Implementations allocate and fill a weight Tensor in C memory.
 */
interface Initializer
{
    /**
     * Initialize a weight matrix of shape [fanIn × fanOut].
     */
    public function initialize(int $fanIn, int $fanOut): Tensor;
}
