<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Interfaces\Stateful;
use Pml\Tensor;

/**
 * LayerNorm with learnable γ (weight) and β (bias).
 *
 * Forward:  out = (x − μ) / √(σ²+eps) · γ + β
 * Backward: returns dx; accumulates dγ and dβ in-place.
 *
 * State-dict keys: "weight", "bias"
 */
final class LayerNorm implements Layer, Stateful
{
    private Tensor $weight;   // [D]  γ — initialized to ones
    private Tensor $bias;     // [D]  β — initialized to zeros

    private Tensor $dWeight;  // [D]  gradient accumulator for γ
    private Tensor $dBias;    // [D]  gradient accumulator for β

    private ?Tensor $cachedInput = null;

    public function __construct(private readonly int $dim, private readonly float $eps = 1e-5)
    {
        $this->weight  = Tensor::ones($dim);
        $this->bias    = Tensor::zeros($dim);
        $this->dWeight = Tensor::zeros($dim);
        $this->dBias   = Tensor::zeros($dim);
    }

    public function forward(Tensor $input): Tensor
    {
        $this->cachedInput = $input;
        return $input->layernormForward($this->weight, $this->bias, $this->eps);
    }

    public function backward(Tensor $dY): Tensor
    {
        $x = $this->cachedInput;
        if ($x === null) throw new \RuntimeException('LayerNorm::backward called before forward');
        $this->cachedInput = null;
        return $dY->layernormBackward($x, $this->weight, $this->eps, $this->dWeight, $this->dBias);
    }

    public function zeroGrads(): void
    {
        $this->dWeight->fill(0.0);
        $this->dBias->fill(0.0);
    }

    public function getParameters(): array
    {
        return ['weight' => $this->weight, 'bias' => $this->bias];
    }

    public function getGradients(): array
    {
        return ['weight' => $this->dWeight, 'bias' => $this->dBias];
    }

    public function getStateDict(string $prefix = ''): array
    {
        return [
            $prefix . 'weight' => $this->weight,
            $prefix . 'bias'   => $this->bias,
        ];
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        if (isset($dict[$prefix . 'weight'])) $this->weight->copyFrom($dict[$prefix . 'weight']);
        if (isset($dict[$prefix . 'bias']))   $this->bias->copyFrom($dict[$prefix . 'bias']);
    }

    public function getConfig(): array
    {
        return ['dim' => $this->dim, 'eps' => $this->eps];
    }
}
