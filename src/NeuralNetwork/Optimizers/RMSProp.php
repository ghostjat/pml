<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

use Pml\NeuralNetwork\Layers\Layer;
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
            foreach ($layer->getParameters() as $name => $paramTensor) {
                $grads = $layer->getGradients();
                
                if (isset($grads[$name])) {
                    $oid = spl_object_id($paramTensor);
                    $g = $grads[$name];

                    if (!isset($this->cache[$oid])) {
                        $this->cache[$oid] = Tensor::zeros(...$paramTensor->shape());
                    }

                    $cache = $this->cache[$oid];

                    // Cache = decay * Cache + (1 - decay) * g^2
                    $gSq = $g->square()->mulScalarInplace(1.0 - $this->decay);
                    $cache->mulScalarInplace($this->decay)->addInplace($gSq);

                    // Update = (lr / sqrt(Cache + eps)) * g
                    $denom = $cache->addScalar($this->epsilon)->sqrt();
                    $update = $g->divInplace($denom)->mulScalarInplace($this->learningRate);

                    // Apply update In-Place
                    $paramTensor->subInplace($update);
                }
            }
        }
    }

    public function __sleep(): array { return ['learningRate', 'decay', 'epsilon']; }
    public function __wakeup(): void { $this->cache = []; }
}