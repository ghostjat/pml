<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Interfaces\Stateful;
use Pml\Tensor;

/**
 * 2D Convolutional Layer for Image Processing.
 * Hardware accelerated using our C-level `im2col` and `col2im` unrolling matrices.
 */
final class Conv2D implements Layer, Stateful
{
    private Tensor $weights;
    private ?Tensor $bias;
    
    private int $stride;
    private int $padding;

    // Cached state
    private ?Tensor $input = null;
    private ?Tensor $dW = null;
    private ?Tensor $dbias = null;

    public function __construct(int $inChannels, int $outChannels, int $kernelSize, int $stride = 1, int $padding = 0, bool $useBias = true)
    {
        $this->stride = $stride;
        $this->padding = $padding;

        // Xavier Initialization
        $stddev = sqrt(2.0 / ($inChannels * $kernelSize * $kernelSize));
        // Shape: [Out_C, In_C, K_H, K_W]
        $this->weights = Tensor::randomNormal([$outChannels, $inChannels, $kernelSize, $kernelSize], 0.0, $stddev);
        
        $this->bias = $useBias ? Tensor::zeros($outChannels) : null;
    }

    public function forward(Tensor $input): Tensor
    {
        $this->input = $input;
        return $input->conv2d($this->weights, $this->bias, $this->stride, $this->stride, $this->padding, $this->padding);
    }

    public function backward(Tensor $dY): Tensor
    {
        if ($this->input === null) {
            throw new \RuntimeException("Backward pass called before forward pass.");
        }

        // Invokes our dedicated C-kernel that computes dX, dW, and dbias simultaneously 
        // using highly-optimized OpenBLAS GEMM operations on im2col matrices.
        $grads = $dY->conv2dBackward(
            $this->input,
            $this->weights,
            $this->stride, $this->stride,
            $this->padding, $this->padding
        );

        $this->dW = $grads['dW'];
        $this->dbias = $grads['dbias'];

        return $grads['dX']; // Return the gradient to the previous layer
    }

    public function getParameters(): array
    {
        $params = ['weights' => $this->weights];
        if ($this->bias !== null) $params['bias'] = $this->bias;
        return $params;
    }

    public function getGradients(): array
    {
        $grads = ['weights' => $this->dW];
        if ($this->dbias !== null) $grads['bias'] = $this->dbias;
        return $grads;
    }

    public function getConfig(): array
    {
        $s = $this->weights->shape(); // [outC, inC, kH, kW]
        return [
            'inChannels'  => $s[1],
            'outChannels' => $s[0],
            'kernelSize'  => $s[2],
            'stride'      => $this->stride,
            'padding'     => $this->padding,
            'useBias'     => $this->bias !== null,
        ];
    }

    public static function fromConfig(array $config): static
    {
        return new static(
            (int)  $config['inChannels'],
            (int)  $config['outChannels'],
            (int)  $config['kernelSize'],
            (int)  $config['stride'],
            (int)  $config['padding'],
            (bool) $config['useBias']
        );
    }

    public function getStateDict(string $prefix = ''): array
    {
        $dict = [$prefix . 'weights' => $this->weights];
        if ($this->bias !== null) { $dict[$prefix . 'bias'] = $this->bias; }
        return $dict;
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->weights = $dict[$prefix . 'weights'];
        $this->bias    = $dict[$prefix . 'bias'] ?? null;
        $this->dW      = null;
        $this->dbias   = null;
        $this->input   = null;
    }
}