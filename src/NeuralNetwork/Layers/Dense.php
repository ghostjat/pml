<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;
use Pml\Interfaces\Stateful;

/**
 * Fully-connected linear layer with optional bias.
 * Implements Stateful for zero-copy SafeTensors checkpoint I/O.
 *
 * State-dict keys (relative to prefix):
 *   "weight" — [outputDim, inputDim] FLOAT32
 *   "bias"   — [outputDim]           FLOAT32  (omitted when useBias=false)
 */
final class Dense implements Layer, Stateful, HasTrainingMode
{
    private Tensor $weights;
    private ?Tensor $bias;

    private ?Tensor $input = null;
    private bool $training = true;   // false during inference — skips input cache

    // Reusable buffers
    private Tensor $dW;
    private ?Tensor $dbias;

    // Cached feature flags (avoid method_exists in hot path)
    private bool $hasMatmulInto;
    private bool $hasSumInto;

    /**
     * 
     * @param int $inputDim
     * @param int $outputDim
     * @param bool $useBias
     */
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

        // Only cache during training — avoids pinning large tensors in inference mode.
        $this->input = $this->training ? $input : null;

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

    public function setTraining(bool $mode): void
    {
        $this->training = $mode;
        if (!$mode) {
            $this->input = null;  // Release any cached input immediately
        }
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

    // =========================================================================
    // Layer config — JSON-safe constructor descriptor for checkpoint rebuild
    // =========================================================================

    /**
     * Return the constructor arguments needed to recreate this layer.
     * Shapes are read from the live C-structs; no PHP copies, no FFI.
     */
    public function getConfig(): array
    {
        return [
            'inputDim'  => $this->weights->ptr->shape[1],
            'outputDim' => $this->weights->ptr->shape[0],
            'useBias'   => $this->bias !== null,
        ];
    }

    /**
     * Reconstruct a Dense layer from a config array (as stored in config.json).
     * The returned layer has freshly initialised random weights; call
     * loadStateDict() immediately after to replace them with checkpoint data.
     */
    public static function fromConfig(array $config): static
    {
        return new static(
            (int) $config['inputDim'],
            (int) $config['outputDim'],
            (bool) $config['useBias']
        );
    }

    // =========================================================================
    // Stateful — SafeTensors checkpoint interface
    // =========================================================================

    /**
     * Export live C-memory tensors as a flat name → Tensor map.
     * No copies: returns references to the actual weight/bias C-buffers.
     *
     * {@inheritdoc}
     */
    public function getStateDict(string $prefix = ''): array
    {
        $dict = [$prefix . 'weight' => $this->weights];

        if ($this->bias !== null) {
            $dict[$prefix . 'bias'] = $this->bias;
        }

        return $dict;
    }

    /**
     * O(1) weight ingestion: swaps internal Tensor references to the provided
     * (typically mmap-backed) tensors.  No memcpy.
     *
     * After loading, gradient buffers (dW, dbias) are reinitialised to the
     * correct shape so the layer is immediately ready for both inference and
     * fine-tuning without any further setup.
     *
     * {@inheritdoc}
     */
    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $wKey = $prefix . 'weight';
        if (isset($dict[$wKey])) {
            $this->weights = $dict[$wKey];
            // Reinitialise gradient buffer to match new weight shape
            $outDim = $this->weights->ptr->shape[0];
            $inDim  = $this->weights->ptr->shape[1];
            $this->dW = Tensor::zeros($outDim, $inDim);
            // Refresh capability flags (class is the same, but good hygiene)
            $this->hasMatmulInto = method_exists($this->dW, 'matmulInto');
        }

        $bKey = $prefix . 'bias';
        if (isset($dict[$bKey])) {
            $this->bias   = $dict[$bKey];
            $this->dbias  = Tensor::zeros($this->bias->ptr->shape[0]);
            $this->hasSumInto = method_exists($this->dbias, 'sumAxisInto');
        } elseif (!array_key_exists($bKey, $dict) && $this->bias !== null) {
            // Key absent but layer was constructed with bias: leave existing bias
        }
    }
}