<?php

declare(strict_types=1);

namespace Pml\Losses;

use Pml\Tensor;

/**
 * Binary Cross Entropy (Log Loss).
 * Standard loss function for binary classification and Sigmoid activations.
 * * JIT & Memory Optimized:
 * - Leverages AVX2 SIMD via `log1p` for numerical stability and speed.
 * - Maximizes `*Inplace` C-mutations to prevent PHP heap fragmentation.
 * - Operates entirely via Zero-Copy C-Pointers.
 */
final class BinaryCrossEntropy implements Loss
{
    /** Pre-allocated gradient buffer — reallocated only when batch size changes. */
    private ?Tensor $gradBuffer = null;

    public function compute(Tensor $predictions, Tensor $labels): float
    {
        // Fused C kernel: clip + loss computation in one pass. No PHP allocs.
        return Tensor::fusedBceLossAndGrad($predictions, $labels);
    }

    public function differentiate(Tensor $predictions, Tensor $labels): Tensor
    {
        // Reuse gradient buffer across steps — reallocate only on batch size change.
        $size = $predictions->size();
        if ($this->gradBuffer === null || $this->gradBuffer->size() !== $size) {
            $this->gradBuffer = Tensor::zeros(...$predictions->shape());
        }

        // Single C kernel: clip + grad in one pass, writes into pre-allocated buffer.
        Tensor::fusedBceLossAndGrad($predictions, $labels, $this->gradBuffer);
        return $this->gradBuffer;
    }
}