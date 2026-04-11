<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

use Pml\NeuralNetwork\Layers\Layer;
use Pml\Tensor;

/**
 * Stochastic Gradient Descent with Momentum.
 * Accelerates SGD in the relevant direction and dampens oscillations.
 */
final class Momentum implements Optimizer
{
    private float $learningRate;
    private float $momentum;

    private array $velocities = [];

    public function __construct(float $learningRate = 0.01, float $momentum = 0.9)
    {
        $this->learningRate = $learningRate;
        $this->momentum = $momentum;
    }

    public function step(array $layers): void
    {
        foreach ($layers as $layer) {
            foreach ($layer->getParameters() as $name => $paramTensor) {
                $grads = $layer->getGradients();
                
                if (isset($grads[$name])) {
                    $oid = spl_object_id($paramTensor);
                    $g = $grads[$name];

                    if (!isset($this->velocities[$oid])) {
                        $this->velocities[$oid] = Tensor::zeros(...$paramTensor->shape());
                    }

                    $v = $this->velocities[$oid];

                    // v = momentum * v + lr * g
                    $scaledG = $g->mulScalar($this->learningRate);
                    $v->mulScalarInplace($this->momentum)->addInplace($scaledG);

                    // Apply momentum update In-Place
                    $paramTensor->subInplace($v);
                }
            }
        }
    }

    public function __sleep(): array { return ['learningRate', 'momentum']; }
    public function __wakeup(): void { $this->velocities = []; }
}