<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;
use InvalidArgumentException;
use RuntimeException;

/**
 * Dropout Regularization Layer.
 * Randomly zeroes out elements with probability $p$ to prevent neural co-adaptation.
 * * JIT & Memory Optimized:
 * - Employs "Inverted Dropout" scaling natively in C during training.
 * - Inference is a zero-cost pass-through.
 */
final class Dropout implements Layer, HasTrainingMode
{
    private float $rate;
    private ?Tensor $mask = null;
    
    /** @var bool Flag to toggle behavior between train and evaluate modes */
    public bool $training = true;

    public function __construct(float $rate = 0.5)
    {
        if ($rate < 0.0 || $rate >= 1.0) {
            throw new InvalidArgumentException("Dropout rate must be strictly between 0.0 and 1.0");
        }
        $this->rate = $rate;
    }

    public function forward(Tensor $input): Tensor
    {
        if (!$this->training || $this->rate === 0.0) {
            return $input->copy();
        }

        // greaterScalar(keepProb) replaces: ones() + mulScalar(rate) + greaterEqual()
        // Saves 2 allocations (ones tensor + threshold tensor) per forward call.
        $this->mask = Tensor::randomUniform($input->shape(), 0.0, 1.0)
            ->greaterScalar($this->rate)
            ->mulScalarInplace(1.0 / (1.0 - $this->rate));

        return $input->mul($this->mask);
    }

    public function backward(Tensor $dY): Tensor
    {
        if (!$this->training || $this->rate === 0.0) {
            return $dY;
        }
        if ($this->mask === null) {
            throw new RuntimeException("Backward called before forward pass.");
        }

        // Gradients only flow back through the neurons that weren't dropped
        return $dY->mul($this->mask);
    }

    public function setTraining(bool $mode): void
    {
        $this->training = $mode;
    }

    public function getParameters(): array { return []; }
    public function getGradients(): array { return []; }
}