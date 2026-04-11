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
final class Dropout implements Layer
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
        // During inference, Dropout acts as an identity function.
        if (!$this->training || $this->rate === 0.0) {
            return $input->copy(); 
        }

        // 1. Generate a boolean mask in C-memory (1.0 to keep, 0.0 to drop)
        $rand = Tensor::randomUniform($input->shape(), 0.0, 1.0);
        
        // Threshold tensor for the keep probability: (1.0 - rate)
        $threshold = Tensor::ones(...$input->shape())->mulScalarInplace($this->rate);
        
        $this->mask = $rand->greaterEqual($threshold);
        
        // 2. Inverted Dropout Scaling
        // Scale the kept weights up during training by 1 / (1 - rate)
        // This ensures the expected sum remains consistent, avoiding scaling during inference.
        $this->mask->mulScalarInplace(1.0 / (1.0 - $this->rate));

        // 3. Apply the mask natively
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

    public function getParameters(): array { return []; }
    public function getGradients(): array { return []; }
}