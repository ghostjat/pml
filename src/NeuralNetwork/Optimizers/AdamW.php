<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

use Pml\Tensor;

/**
 * AdamW — Adam with decoupled weight decay (Loshchilov & Hutter, 2019).
 *
 * Differs from L2-regularised Adam in that weight decay is applied directly
 * to the parameters before the gradient step, not folded into the gradient.
 * This gives better generalisation, especially for transformer models.
 *
 * Memory: one m + one v buffer per parameter tensor, identical to Adam.
 * Compute: single fused C kernel — zero PHP tensor allocations per step.
 */
final class AdamW implements Optimizer, LearningRateAware
{
    private float $learningRate;
    private float $beta1;
    private float $beta2;
    private float $epsilon;
    private float $weightDecay;

    private int $t = 0;

    /** @var array<int, Tensor> */
    private array $m = [];
    /** @var array<int, Tensor> */
    private array $v = [];

    public function __construct(
        float $learningRate = 1e-3,
        float $beta1        = 0.9,
        float $beta2        = 0.999,
        float $epsilon      = 1e-8,
        float $weightDecay  = 1e-2,
    ) {
        $this->learningRate = $learningRate;
        $this->beta1        = $beta1;
        $this->beta2        = $beta2;
        $this->epsilon      = $epsilon;
        $this->weightDecay  = $weightDecay;
    }

    public function step(array $layers): void
    {
        $this->t++;

        foreach ($layers as $layer) {
            $params = $layer->getParameters();
            $grads  = $layer->getGradients();

            foreach ($params as $name => $param) {
                if (!isset($grads[$name])) continue;

                $oid = spl_object_id($param);
                if (!isset($this->m[$oid])) {
                    $this->m[$oid] = Tensor::zeros(...$param->shape());
                    $this->v[$oid] = Tensor::zeros(...$param->shape());
                }

                Tensor::fusedAdamWStep(
                    $param, $grads[$name],
                    $this->m[$oid], $this->v[$oid],
                    $this->learningRate,
                    $this->beta1, $this->beta2, $this->epsilon,
                    $this->t,
                    $this->weightDecay
                );
            }
        }
    }

    public function getLearningRate(): float  { return $this->learningRate; }
    public function setLearningRate(float $lr): void { $this->learningRate = $lr; }

    public function __sleep(): array
    {
        return ['learningRate', 'beta1', 'beta2', 'epsilon', 'weightDecay', 't'];
    }

    public function __wakeup(): void
    {
        $this->m = [];
        $this->v = [];
    }
}
