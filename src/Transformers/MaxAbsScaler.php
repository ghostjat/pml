<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Stateful;
use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Maximum Absolute Scaler.
 * Scales each feature by its maximum absolute value, ensuring data sits within [-1, 1].
 * Highly recommended for Sparse Data (like NLP TF-IDF sets) because it does not center the data 
 * (which would destroy sparsity and crash memory).
 * * JIT & Memory Optimized:
 * - Employs pure C-level `abs()` and `maxAxis()` functions instantly.
 */
final class MaxAbsScaler implements Transformer, Stateful
{
    private ?Tensor $maxAbs = null;

    public function fit(Dataset $dataset): void
    {
        // Extracts the maximum absolute float per column
        $this->maxAbs = $dataset->samples()->abs()->maxAxis(0)->clip(1e-8, INF);
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("MaxAbsScaler is not fitted.");
        }

        // X_scaled = X / max_abs
        $scaled = $dataset->samples()->div($this->maxAbs);

        return new Dataset($scaled, $dataset->labels());
    }

    public function fitted(): bool
    {
        return $this->maxAbs !== null;
    }

    public function getStateDict(string $prefix = ''): array
    {
        $dict = [];
        if ($this->maxAbs !== null) { $dict[$prefix . 'maxAbs'] = $this->maxAbs; }
        return $dict;
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->maxAbs = $dict[$prefix . 'maxAbs'] ?? null;
    }
}