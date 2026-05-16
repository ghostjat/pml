<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * Hard-Swish activation — f(x) = x · clamp(x+3, 0, 6) / 6
 *
 * Drop-in replacement for Swish in MobileNetV3 / EfficientNet models.
 * Faster than Swish (no sigmoid), hardware-friendly integer approximation.
 * All computation in C via tensor_hard_swish (AVX2 + OpenMP).
 */
final class HardSwish implements Layer
{
    private ?Tensor $input = null;

    public function forward(Tensor $input): Tensor
    {
        $this->input = $input;
        return $input->hardSwish();
    }

    public function backward(Tensor $dY): Tensor
    {
        return $dY->hardSwishBackward($this->input);
    }

    public function getParameters(): array { return []; }
    public function getGradients(): array  { return []; }
    public function getConfig(): array     { return []; }
    public static function fromConfig(array $config): static { return new static(); }
}
