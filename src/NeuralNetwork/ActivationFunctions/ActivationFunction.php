<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\ActivationFunctions;

use Pml\Tensor;

/**
 * Activation function interface.
 * Both forward and backward passes must return Tensors (stay in C memory).
 */
interface ActivationFunction
{
    /** Forward pass: compute activation. */
    public function activate(Tensor $z): Tensor;

    /** Backward pass: element-wise derivative w.r.t. pre-activation z. */
    public function differentiate(Tensor $z): Tensor;
}
