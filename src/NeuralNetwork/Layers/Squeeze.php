<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * Squeeze Layer — removes dimensions of size 1.
 *
 * Primary use-case: collapse binary-classifier output [N, 1] → [N]
 * so that BinaryCrossEntropy can compare predictions to flat [N] labels.
 *
 * Zero-copy: delegates to Tensor::squeeze() which is a C-level reshape.
 * Backward restores the original shape via reshape so gradients flow correctly.
 */
final class Squeeze implements Layer
{
    private ?array $inputShape = null;

    public function forward(Tensor $input): Tensor
    {
        $this->inputShape = $input->shape();
        return $input->squeeze();
    }

    public function backward(Tensor $dY): Tensor
    {
        if ($this->inputShape === null) {
            throw new \RuntimeException('Squeeze::backward called before forward.');
        }
        return $dY->reshape(...$this->inputShape);
    }

    public function getParameters(): array { return []; }

    public function getGradients(): array { return []; }
}
