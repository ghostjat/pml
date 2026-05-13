<?php

declare(strict_types=1);

namespace Pml\SLM;

use Pml\Interfaces\Stateful;
use Pml\NeuralNetwork\Layers\Layer;
use Pml\Tensor;

/**
 * Embedding layer with a trainable weight matrix and correct scatter-add backward.
 *
 * Differs from the frozen Pml\NeuralNetwork\Layers\Embedding in that:
 *   - backward() accumulates dWeights via tensor_embedding_backward (scatter-add)
 *   - zeroGrads() must be called after each optimizer step
 *
 * Forward:  token_ids [B, T] INT32  →  embeddings [B, T, D] FLOAT32
 * Backward: dY [B, T, D]            →  zeros (no grad for discrete tokens)
 *           side-effect: dWeights [V, D] accumulated in-place
 *
 * The optimizer reads dWeights via getGradients()['weights'] after the backward
 * phase.  Call zeroGrads() once the optimizer step is complete so the next
 * accumulation window starts clean.
 */
final class TrainableEmbedding implements Layer, Stateful
{
    private readonly int $vocabSize;
    private readonly int $embedDim;

    private Tensor $weights;   // [V, D]
    private Tensor $dWeights;  // [V, D]  — accumulated by scatter-add

    private ?Tensor $cachedIds = null;  // [B, T]  kept for backward

    public function __construct(int $vocabSize, int $embedDim)
    {
        $this->vocabSize = $vocabSize;
        $this->embedDim  = $embedDim;

        // Kaiming-normal initialisation with scale 1/sqrt(D).
        $std           = 1.0 / sqrt($embedDim);
        $this->weights  = Tensor::randomNormal([$vocabSize, $embedDim], 0.0, $std);
        $this->dWeights = Tensor::zeros($vocabSize, $embedDim);
    }

    // ── Forward ───────────────────────────────────────────────────────────────

    /**
     * @param Tensor $input  INT32 [B, T]
     * @return Tensor        FLOAT32 [B, T, D]
     */
    public function forward(Tensor $input): Tensor
    {
        if ($input->dtype() !== Tensor::DTYPE_INT32) {
            throw new \InvalidArgumentException('TrainableEmbedding: input must be DTYPE_INT32');
        }

        $this->cachedIds = $input;

        $shape = $input->shape();
        $B     = $shape[0];
        $T     = $shape[1];

        // Flatten to [B*T] for embeddingLookup, then reshape to [B, T, D].
        $flat      = $input->reshape($B * $T);
        $embedded  = $flat->embeddingLookup($this->weights);   // [B*T, D]
        return $embedded->reshape($B, $T, $this->embedDim);    // [B, T, D]
    }

    // ── Backward ──────────────────────────────────────────────────────────────

    /**
     * Accumulate embedding gradients via scatter-add.
     * Returns Tensor::zeros(1) — token ids have no real gradient.
     *
     * @param Tensor $dY  FLOAT32 [B, T, D]  upstream gradient
     * @return Tensor     zeros (placeholder; ignored by callers)
     */
    public function backward(Tensor $dY): Tensor
    {
        if ($this->cachedIds === null) {
            throw new \RuntimeException('TrainableEmbedding::backward called before forward');
        }

        $shape = $this->cachedIds->shape();
        $B     = $shape[0];
        $T     = $shape[1];

        $flatDY  = $dY->reshape($B * $T, $this->embedDim);  // [B*T, D]
        $flatIds = $this->cachedIds->reshape($B * $T);       // [B*T] INT32

        // scatter-add: dWeights[flatIds[i]] += flatDY[i]
        $flatDY->embeddingBackward($flatIds, $this->dWeights);

        $this->cachedIds = null;
        return Tensor::zeros(1);
    }

    // ── Gradient management ───────────────────────────────────────────────────

    /**
     * Zero the gradient accumulator.  MUST be called after each optimizer step.
     */
    public function zeroGrads(): void
    {
        $this->dWeights->fill(0.0);
    }

    // ── Layer interface ───────────────────────────────────────────────────────

    public function getParameters(): array
    {
        return ['weights' => $this->weights];
    }

    public function getGradients(): array
    {
        return ['weights' => $this->dWeights];
    }

    // ── Stateful interface ────────────────────────────────────────────────────

    public function getStateDict(string $prefix = ''): array
    {
        return [$prefix . 'weights' => $this->weights];
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $key = $prefix . 'weights';
        if (!isset($dict[$key])) return;

        $src = $dict[$key];
        if ($src->size() === $this->weights->size()) {
            // Same shape: copy into existing owned tensor (safe against mmap PROT_READ).
            $this->weights->copyFrom($src);
        } else {
            // Shape changed (architecture switch): make an owned copy of the source.
            $this->weights = $src->copy();
        }

        $V = $this->weights->ptr->shape[0];
        $D = $this->weights->ptr->shape[1];
        $this->dWeights = Tensor::zeros($V, $D);
    }

    public function getConfig(): array
    {
        return ['vocabSize' => $this->vocabSize, 'embedDim' => $this->embedDim];
    }

    public static function fromConfig(array $config): static
    {
        return new static((int) $config['vocabSize'], (int) $config['embedDim']);
    }
}
