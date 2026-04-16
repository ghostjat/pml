<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Stateful;
use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Standard Scaler (Z-Score Normalization).
 * Standardizes features by removing the mean and scaling to unit variance.
 * * JIT & Memory Optimized:
 * - Computes Variance natively in C using the formula: Var = E[X^2] - E[X]^2.
 * - Applies scaling via zero-allocation In-Place broadcasting.
 */
final class StandardScaler implements Transformer, Stateful
{
    private ?Tensor $means = null;
    private ?Tensor $stds = null;

    public function fit(Dataset $dataset): void
    {
        $x = $dataset->samples();

        // 1. Calculate the Means (E[X])
        $this->means = $x->meanAxis(0);

        // 2. Calculate the Mean of Squares (E[X^2])
        $meanOfSquares = $x->square()->meanAxis(0);

        // 3. Calculate Variance: E[X^2] - E[X]^2
        $squaredMeans = $this->means->square();
        $variance = $meanOfSquares->sub($squaredMeans);

        // 4. Calculate Standard Deviation (with epsilon to prevent division by zero)
        $this->stds = $variance->sqrt()->clip(1e-8, INF);
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("StandardScaler has not been fitted.");
        }

        // Standardize to Z-Score: X_std = (X - Mean) / StdDev
        // Creates a new tensor to avoid mutating the original dataset's memory
        $scaled = $dataset->samples()->sub($this->means)->divInplace($this->stds);

        return new Dataset($scaled, $dataset->labels());
    }

    public function fitted(): bool
    {
        return $this->means !== null && $this->stds !== null;
    }

    public function getStateDict(string $prefix = ''): array
    {
        $dict = [];
        if ($this->means !== null) {
            $dict[$prefix . 'means'] = $this->means;
        }
        if ($this->stds !== null) {
            $dict[$prefix . 'stds'] = $this->stds;
        }
        return $dict;
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->means = $dict[$prefix . 'means'] ?? null;
        $this->stds  = $dict[$prefix . 'stds']  ?? null;
    }
}