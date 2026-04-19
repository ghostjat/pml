<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * Gaussian Noise Layer — adds zero-mean Gaussian noise during training only.
 * Acts as a regularizer by preventing co-adaptation of units.
 *
 * JIT & Memory Optimized:
 * - Noise tensor allocated in C; addition is in-place — zero extra PHP allocations.
 * - At inference (training=false) the layer is a pure identity pass.
 */
final class Noise implements Layer
{
    public function __construct(private readonly float $stddev = 0.1) {}

    public function forward(Tensor $x, bool $training = true): Tensor
    {
        if (!$training) return $x;
        $noise = Tensor::randomNormal($x->shape(), 0.0, $this->stddev);
        return $x->add($noise);
    }

    public function backward(Tensor $dOut, Tensor $x): Tensor
    {
        return $dOut;   // noise has no learnable parameters
    }

    public function params(): array { return []; }

    #[\Override]
    public function getGradients(): array {
        
    }

    #[\Override]
    public function getParameters(): array {
        
    }
}
