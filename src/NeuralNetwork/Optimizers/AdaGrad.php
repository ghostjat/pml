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
            foreach ($layer->getParameters() as $name => $paramTensor) {
                $grads = $layer->getGradients();
                
                if (isset($grads[$name])) {
                    $oid = spl_object_id($paramTensor);
                    $g = $grads[$name];

                    if (!isset($this->cache[$oid])) {
                        $this->cache[$oid] = Tensor::zeros(...$paramTensor->shape());
                    }

                    $cache = $this->cache[$oid];

                    // Cache += g^2
                    $gSq = $g->square();
                    $cache->addInplace($gSq);

                    // Update = (lr / sqrt(Cache + eps)) * g
                    $denom = $cache->addScalar($this->epsilon)->sqrt();
                    $update = $g->divInplace($denom)->mulScalarInplace($this->learningRate);

                    // Apply update In-Place
                    $paramTensor->subInplace($update);
                }
            }
        }
    }

    public function __sleep(): array { return ['learningRate', 'epsilon']; }
    public function __wakeup(): void { $this->cache = []; }
}