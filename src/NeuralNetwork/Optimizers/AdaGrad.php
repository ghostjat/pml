<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

use Pml\Tensor;

/**
 * Adaptive Gradient Algorithm (AdaGrad).
 * Adapts the learning rate by accumulating the sum of squared gradients. 
 * Performs larger updates for infrequent parameters and smaller updates for frequent ones.
 * * JIT & Memory Optimized:
 * - Employs purely In-Place cache mutations to prevent memory leaks.
 */
final class AdaGrad implements Optimizer
{
    private float $learningRate;
    private float $epsilon;
    private array $cache = [];

    public function __construct(float $learningRate = 0.01, float $epsilon = 1e-8)
    {
        $this->learningRate = $learningRate;
        $this->epsilon = $epsilon;
    }

    public function step(array $layers): void
    {
        foreach ($layers as $layer) {
            $params = $layer->getParameters();
            $grads  = $layer->getGradients();

            foreach ($params as $name => $param) {
                if (!isset($grads[$name])) continue;

                $oid = spl_object_id($param);
                if (!isset($this->cache[$oid])) {
                    $this->cache[$oid] = Tensor::zeros(...$param->shape());
                }

                Tensor::fusedAdaGradStep(
                    $param, $grads[$name], $this->cache[$oid],
                    $this->learningRate, $this->epsilon
                );
            }
        }
    }

    public function __sleep(): array { return ['learningRate', 'epsilon']; }
    public function __wakeup(): void { $this->cache = []; }
}