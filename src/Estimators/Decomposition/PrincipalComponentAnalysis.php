<?php

declare(strict_types=1);

namespace Pml\Estimators\Decomposition;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Principal Component Analysis (PCA).
 * Reduces dataset dimensionality while preserving variance.
 * * JIT & Memory Optimized:
 * - Direct LAPACKE SVD extraction.
 * - Zero-Copy slicing for principal components.
 */
final class PrincipalComponentAnalysis implements Learner
{
    private int $nComponents;
    private ?Tensor $components = null;
    private ?Tensor $means = null;

    public function __construct(int $nComponents)
    {
        if ($nComponents < 1) {
            throw new \InvalidArgumentException("Number of components must be >= 1.");
        }
        $this->nComponents = $nComponents;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();

        // 1. Calculate column means for centering (Shape: [D])
        $this->means = $x->meanAxis(0);

        // 2. Center the dataset (AVX2 Broadcasting)
        $centered = $x->sub($this->means);

        // 3. Compute Singular Value Decomposition (SVD)
        $svd = $centered->svd();

        // 4. The Principal Components are the top K rows of V^T
        // Slice operates in <0.01ms as a zero-copy pointer adjustment. We copy() to safely own it.
        $this->components = $svd['Vt']->slice(0, 0, $this->nComponents)->copy();
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new \RuntimeException("PCA has not been fitted.");
        }

        // 1. Center the inference data
        $centered = $dataset->samples()->sub($this->means);

        // 2. Project onto the principal components: X_c * V
        // $this->components is V^T, so we transpose it back to V.
        return $centered->matmul($this->components->transpose());
    }

    public function trained(): bool
    {
        return $this->components !== null;
    }
}