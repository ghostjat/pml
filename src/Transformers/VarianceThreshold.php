<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Variance Threshold Feature Selector.
 * Automatically drops feature columns that have zero or near-zero variance.
 * * JIT & Memory Optimized:
 * - Computes variance concurrently for all columns via OpenBLAS broadcasting.
 * - Extracts selected columns natively using zero-copy `tensor_take()` slicing.
 */
final class VarianceThreshold implements Transformer
{
    private float $minVariance;
    private ?array $selectedColumns = null;

    /**
     * @param float $minVariance The threshold below which a feature will be dropped.
     */
    public function __construct(float $minVariance = 1e-4)
    {
        $this->minVariance = $minVariance;
    }

    public function fit(Dataset $dataset): void
    {
        $x = $dataset->samples();
        
        // Variance: mean((X - E[X])^2)
        // Calculated natively in C-cache for the entire matrix simultaneously
        $mean = $x->meanAxis(0);
        $variance = $x->sub($mean)->square()->meanAxis(0)->toFlatArray();
        
        $this->selectedColumns = [];
        
        foreach ($variance as $colIndex => $var) {
            if ($var >= $this->minVariance) {
                $this->selectedColumns[] = $colIndex;
            }
        }
        
        if (empty($this->selectedColumns)) {
            throw new RuntimeException("VarianceThreshold: No features meet the minimum variance threshold of {$this->minVariance}.");
        }
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("VarianceThreshold is not fitted.");
        }
        
        // Leverage hardware-accelerated memory extraction (tensor_take) to drop the columns
        return $dataset->select($this->selectedColumns);
    }

    public function fitted(): bool
    {
        return $this->selectedColumns !== null;
    }
}