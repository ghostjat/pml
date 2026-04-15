<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

final class Dense implements Layer
{
    private Tensor $weights;
    private ?Tensor $bias;

    private ?Tensor $input = null;

    // Reusable buffers
    private Tensor $dW;
    private ?Tensor $dbias;

    // Cached feature flags (avoid method_exists in hot path)
    private bool $hasMatmulInto;
    private bool $hasSumInto;

    public function __construct(int $inputDim, int $outputDim, bool $useBias = true)
    {
        $stddev = \sqrt(2.0 / $inputDim);

        $this->weights = Tensor::randomNormal([$outputDim, $inputDim], 0.0, $stddev);

        if ($useBias) {
            $this->bias = Tensor::zeros($outputDim);
            $this->dbias = Tensor::zeros($outputDim);
        } else {
            $this->bias = null;
            $this->dbias = null;
        }

        $this->dW = Tensor::zeros($outputDim, $inputDim);

        // Cache capabilities (JIT friendly)
        $this->hasMatmulInto = method_exists($this->dW, 'matmulInto');
        $this->hasSumInto = $this->dbias !== null && method_exists($this->dbias, 'sumAxisInto');
    }

    public function forward(Tensor $input): Tensor
    {
        // Optional: only enforce if your backend requires it
        if (!$input->isContiguous()) {
            $input = $input->contiguous();
        }

        $this->input = $input;

        return $input->linear($this->weights, $this->bias);
    }

    public function backward(Tensor $dY): Tensor
    {
        $input = $this->input;

        if ($input === null) {
            throw new \RuntimeException("Backward before forward.");
        }

        // Early release (GOOD optimization)
        $this->input = null;

        if (!$dY->isContiguous()) {
            $dY = $dY->contiguous();
        }

        /*
         * 1. dW = dY^T @ X (NO TRANSPOSE)
         */
        if ($this->hasMatmulInto) {
            $this->dW->matmulInto($dY, $input, true, false);
        } else {
            // fallback WITHOUT breaking reuse
            $tmp = $dY->matmul($input, true, false);
            $this->dW->copyFrom($tmp);
        }

        /*
         * 2. dbias
         */
        if ($this->dbias !== null) {
            if ($this->hasSumInto) {
                $this->dbias->sumAxisInto($dY, 0);
            } else {
                $tmp = $dY->sumAxis(0);
                $this->dbias->copyFrom($tmp);
            }
        }

        /*
         * 3. dX
         */
        if ($this->hasMatmulInto) {
            $dX = Tensor::emptyLike($input);
            $dX->matmulInto($dY, $this->weights);
        } else {
            $dX = $dY->matmul($this->weights);
        }

        return $dX;
    }

    public function getParameters(): array
    {
        $params = ['weights' => $this->weights];

        if ($this->bias !== null) {
            $params['bias'] = $this->bias;
        }

        return $params;
    }

    public function getGradients(): array
    {
        $grads = ['weights' => $this->dW];

        if ($this->dbias !== null) {
            $grads['bias'] = $this->dbias;
        }

        return $grads;
    }
}