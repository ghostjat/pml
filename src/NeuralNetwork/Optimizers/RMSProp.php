<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

use Pml\Tensor;

/**
 * Root Mean Square Propagation (RMSProp).
 * Adapts learning rate by dividing it by an exponentially decaying average of squared gradients.
 * * JIT & Memory Optimized:
 * - Employs purely In-Place cache mutations.
 */
final class RMSProp implements Optimizer
{
    private float $learningRate;
    private float $decay;
    private float $epsilon;

    private array $cache = [];

    public function __construct(float $learningRate = 0.001, float $decay = 0.9, float $epsilon = 1e-8)
    {
        $this->learningRate = $learningRate;
        $this->decay = $decay;
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

                Tensor::fusedRmsPropStep(
                    $param, $grads[$name], $this->cache[$oid],
                    $this->learningRate, $this->decay, $this->epsilon
                );
            }
        }
    }

    public function __sleep(): array { return ['learningRate', 'decay', 'epsilon']; }
    public function __wakeup(): void { $this->cache = []; }
}