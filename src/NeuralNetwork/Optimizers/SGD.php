<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

use Pml\Tensor;

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
            $grads  = $layer->getGradients();

            foreach ($params as $name => $param) {
                if (isset($grads[$name])) {
                    Tensor::fusedSgdStep($param, $grads[$name], $this->learningRate);
                }
            }
        }
    }
}