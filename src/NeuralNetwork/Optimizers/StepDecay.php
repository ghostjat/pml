<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

use Pml\Tensor;

/**
 * Step Decay LR Scheduler — reduces the learning rate by a factor every N steps.
 *
 * JIT & Memory Optimized:
 * - LR scheduling is pure PHP scalar comparison and multiply.
 * - Parameter update is a single in-place C scalar multiply + subtract.
 */
final class StepDecay implements Optimizer
{
    private int   $step  = 0;
    private float $lr;

    public function __construct(
        private readonly float $initialLr  = 0.01,
        private readonly float $decayRate  = 0.5,
        private readonly int   $stepEvery  = 100
    ) {
        $this->lr = $initialLr;
    }

    public function update(Tensor $param, Tensor $grad): void
    {
        $this->step++;
        if ($this->step % $this->stepEvery === 0) {
            $this->lr *= $this->decayRate;
        }
        $param->subInplace($grad->mulScalar($this->lr));
    }

    public function learningRate(): float { return $this->lr; }
}
