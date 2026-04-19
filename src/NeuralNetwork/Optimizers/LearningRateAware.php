<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

/**
 * Optional mixin for optimizers whose learning rate can be inspected
 * and adjusted at runtime — required for LR scheduling support in Trainer.
 *
 * Any Optimizer that implements this interface will have its learning rate
 * updated automatically at each epoch boundary by Trainer's LRScheduler.
 */
interface LearningRateAware
{
    public function getLearningRate(): float;

    /**
     * Replace the active learning rate.
     * The change takes effect on the very next call to step().
     */
    public function setLearningRate(float $lr): void;
}
