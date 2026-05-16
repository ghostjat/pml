<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Interfaces\Stateful;
use Pml\Tensor;

/**
 * MobileNetV3 Inverted Residual (Bottleneck) block.
 *
 * Structure:
 *   [optional] Pointwise expand (1×1 conv) → BN → activation
 *              Depthwise conv (k×k)         → BN → activation
 *   [optional] SE block
 *              Pointwise project (1×1 conv) → BN  (no activation)
 *   [optional] Residual add (when in_c == out_c && stride == 1)
 *
 * Activation: 'relu' → ReLU,  'hs' → HardSwish
 * All computation in C; PHP only chains layers.
 */
final class InvertedResidual implements Layer, Stateful
{
    /** @var Layer[] */
    private array $layers = [];
    private bool  $useResidual;
    private ?Tensor $residualInput = null;

    public function __construct(
        int    $inC,
        int    $expandC,
        int    $outC,
        int    $kernelSize,
        int    $stride,
        string $activation = 'relu',   // 'relu' | 'hs'
        bool   $useSE      = false,
        int    $seReduction = 4,
    ) {
        $this->useResidual = ($stride === 1 && $inC === $outC);
        $act = fn() => $activation === 'hs' ? new HardSwish() : new ReLU();
        $pad = (int)(($kernelSize - 1) / 2);

        // Expand phase (skip if expandC == inC)
        if ($expandC !== $inC) {
            $this->layers[] = new Conv2D($inC, $expandC, kernelSize: 1, padding: 0);
            $this->layers[] = new BatchNorm2D($expandC);
            $this->layers[] = $act();
        }

        // Depthwise phase
        $this->layers[] = new DepthwiseConv2D($expandC, $kernelSize, stride: $stride, padding: $pad);
        $this->layers[] = new BatchNorm2D($expandC);
        $this->layers[] = $act();

        // SE block (optional)
        if ($useSE) {
            $this->layers[] = new SEBlock($expandC, $seReduction);
        }

        // Project phase
        $this->layers[] = new Conv2D($expandC, $outC, kernelSize: 1, padding: 0);
        $this->layers[] = new BatchNorm2D($outC);
    }

    public function forward(Tensor $x): Tensor
    {
        $this->residualInput = $x;
        foreach ($this->layers as $layer) {
            $x = $layer->forward($x);
        }
        if ($this->useResidual) {
            $x = $x->add($this->residualInput);
        }
        return $x;
    }

    public function backward(Tensor $dY): Tensor
    {
        $dResidual = $this->useResidual ? $dY : null;

        foreach (array_reverse($this->layers) as $layer) {
            $dY = $layer->backward($dY);
        }

        return $dResidual !== null ? $dY->add($dResidual) : $dY;
    }

    public function getParameters(): array
    {
        $params = [];
        foreach ($this->layers as $i => $layer) {
            foreach ($layer->getParameters() as $k => $v) {
                $params["layer{$i}.{$k}"] = $v;
            }
        }
        return $params;
    }

    public function getGradients(): array
    {
        $grads = [];
        foreach ($this->layers as $i => $layer) {
            foreach ($layer->getGradients() as $k => $v) {
                $grads["layer{$i}.{$k}"] = $v;
            }
        }
        return $grads;
    }

    public function getStateDict(string $prefix = ''): array
    {
        $d = [];
        foreach ($this->layers as $i => $layer) {
            if ($layer instanceof Stateful) {
                $d = array_merge($d, $layer->getStateDict("{$prefix}layer{$i}."));
            }
        }
        return $d;
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        foreach ($this->layers as $i => $layer) {
            if ($layer instanceof Stateful) {
                $layer->loadStateDict($dict, "{$prefix}layer{$i}.");
            }
        }
        $this->residualInput = null;
    }
}
