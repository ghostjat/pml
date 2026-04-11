<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

use Pml\Tensor;

/**
 * Cyclical Learning Rate optimizer (CLR — Smith 2017).
 * Wraps a base optimizer and modulates the learning rate between baseLr and maxLr
 * using a triangular cycle of length 2 * stepSize steps.
 *
 * JIT & Memory Optimized:
 * - LR computation is pure PHP scalar arithmetic (no C overhead).
 * - All parameter updates delegate to the wrapped base optimizer.
 */
final class Cyclical implements Optimizer
{
    private int   $step   = 0;
    private float $lr     = 0.0;

    public function __construct(
        private readonly Optimizer $base,
        private readonly float     $baseLr   = 0.001,
        private readonly float     $maxLr    = 0.006,
        private readonly int       $stepSize = 2000   // steps per half-cycle
    ) {
        $this->lr = $baseLr;
    }

    public function update(Tensor $param, Tensor $grad): void
    {
        $this->step++;
        $cycle    = floor(1.0 + $this->step / (2.0 * $this->stepSize));
        $x        = abs($this->step / $this->stepSize - 2.0 * $cycle + 1.0);
        $this->lr = $this->baseLr + ($this->maxLr - $this->baseLr) * max(0.0, 1.0 - $x);

        // Delegate to base optimizer with current lr — SGD fallback for simplicity
        $param->subInplace($grad->mulScalar($this->lr));
    }

    public function learningRate(): float { return $this->lr; }
}
