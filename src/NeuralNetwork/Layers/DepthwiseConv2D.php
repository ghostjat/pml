<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Interfaces\Stateful;
use Pml\Tensor;

/**
 * Depthwise 2D Convolution — filters each input channel with its own kernel.
 *
 * Input  : [N, C, H, W]
 * Weights: [C, 1, kH, kW]   (one kernel per channel)
 * Output : [N, C, oH, oW]
 *
 * Used in MobileNetV3 inverted residuals, SSDLite, NanoDet, PicoDet, YOLO11n.
 * All computation delegated to tensor_depthwise_conv2d in C (AVX2 + OpenMP).
 */
final class DepthwiseConv2D implements Layer, Stateful
{
    private Tensor $weights;    // [C, 1, kH, kW]
    private ?Tensor $bias;      // [C] or null

    private ?Tensor $input = null;
    private ?Tensor $dW    = null;
    private ?Tensor $dbias = null;

    private readonly int $stride_;
    private readonly int $padding_;

    public function __construct(
        int  $channels,
        int  $kernelSize,
        int  $stride  = 1,
        int  $padding = 0,
        bool $useBias = false,
    ) {
        $this->stride_  = $stride;
        $this->padding_ = $padding;

        // Kaiming uniform for depthwise
        $fan_in = $kernelSize * $kernelSize;
        $bound  = sqrt(1.0 / $fan_in);
        $this->weights = Tensor::randomUniform([$channels, 1, $kernelSize, $kernelSize], -$bound, $bound);
        $this->bias    = $useBias ? Tensor::zeros($channels) : null;
    }

    public function forward(Tensor $input): Tensor
    {
        $this->input = $input;
        return $input->depthwiseConv2d(
            $this->weights, $this->bias,
            $this->stride_, $this->stride_,
            $this->padding_, $this->padding_
        );
    }

    public function backward(Tensor $dY): Tensor
    {
        $grads = $dY->depthwiseConv2dBackward(
            $this->input, $this->weights,
            $this->stride_, $this->stride_,
            $this->padding_, $this->padding_
        );
        $this->dW    = $grads['dW'];
        $this->dbias = $grads['dbias'];
        return $grads['dX'];
    }

    public function getParameters(): array
    {
        $p = ['weights' => $this->weights];
        if ($this->bias !== null) $p['bias'] = $this->bias;
        return $p;
    }

    public function getGradients(): array
    {
        $g = ['weights' => $this->dW];
        if ($this->dbias !== null) $g['bias'] = $this->dbias;
        return $g;
    }

    public function getConfig(): array
    {
        $s = $this->weights->shape();
        return [
            'channels'   => $s[0],
            'kernelSize' => $s[2],
            'stride'     => $this->stride_,
            'padding'    => $this->padding_,
            'useBias'    => $this->bias !== null,
        ];
    }

    public static function fromConfig(array $config): static
    {
        return new static(
            channels:   (int)  $config['channels'],
            kernelSize: (int)  $config['kernelSize'],
            stride:     (int)  $config['stride'],
            padding:    (int)  $config['padding'],
            useBias:    (bool) $config['useBias'],
        );
    }

    public function getStateDict(string $prefix = ''): array
    {
        $d = [$prefix . 'weights' => $this->weights];
        if ($this->bias !== null) $d[$prefix . 'bias'] = $this->bias;
        return $d;
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->weights = $dict[$prefix . 'weights'];
        $this->bias    = $dict[$prefix . 'bias'] ?? null;
        $this->dW = $this->dbias = $this->input = null;
    }
}
