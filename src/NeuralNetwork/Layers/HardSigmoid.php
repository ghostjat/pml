<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * Hard-Sigmoid activation — f(x) = clamp(x+3, 0, 6) / 6
 *
 * Used as the gating function in MobileNetV3 SE blocks.
 * Piecewise-linear approximation of sigmoid.
 * All computation in C via tensor_hard_sigmoid (AVX2 + OpenMP).
 */
final class HardSigmoid implements Layer
{
    private ?Tensor $input = null;

    public function forward(Tensor $input): Tensor
    {
        $this->input = $input;
        return $input->hardSigmoid();
    }

    public function backward(Tensor $dY): Tensor
    {
        return $dY->hardSigmoidBackward($this->input);
    }

    public function getParameters(): array { return []; }
    public function getGradients(): array  { return []; }
    public function getConfig(): array     { return []; }
    public static function fromConfig(array $config): static { return new static(); }
}
