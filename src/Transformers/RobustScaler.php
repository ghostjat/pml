<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Stateful;
use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Robust Scaler.
 * Standardizes features using statistics that are robust to outliers (Median and IQR).
 * * JIT & Memory Optimized:
 * - Uses single-column extraction and PHP `sort()` to evaluate percentiles at JIT speed.
 * - Compiles statistics back into C-Pointers for mass OpenBLAS broadcasting during transformation.
 */
final class RobustScaler implements Transformer, Stateful
{
    private ?Tensor $medians = null;
    private ?Tensor $iqrs = null;

    public function fit(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $rows = $dataset->numRows();
        $cols = $dataset->numColumns();

        if ($rows === 0) return;

        // Establish Quartile Indices
        $q1Idx = (int) floor($rows * 0.25);
        $medIdx = (int) floor($rows * 0.50);
        $q3Idx = (int) floor($rows * 0.75);

        $medians = [];
        $iqrs = [];

        for ($i = 0; $i < $cols; $i++) {
            // Extract a 1D Flat Array representation of the column
            $colData = $x->col($i)->toFlatArray();
            
            // PHP's native C-compiled QuickSort is insanely fast for 1D scalar arrays
            sort($colData);

            $median = $colData[$medIdx];
            $iqr = $colData[$q3Idx] - $colData[$q1Idx];

            $medians[] = $median;
            $iqrs[] = $iqr;
        }

        // Load the computed metrics back into C-Memory for transformation
        $this->medians = Tensor::fromArray($medians);
        $this->iqrs = Tensor::fromArray($iqrs)->clip(1e-8, INF);
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("RobustScaler is not fitted.");
        }

        // X_scaled = (X - median) / IQR
        // Executes purely in C via Vector-Matrix hardware broadcasting
        $scaled = $dataset->samples()->sub($this->medians)->divInplace($this->iqrs);

        return new Dataset($scaled, $dataset->labels());
    }

    public function fitted(): bool
    {
        return $this->medians !== null && $this->iqrs !== null;
    }

    public function getStateDict(string $prefix = ''): array
    {
        $dict = [];
        if ($this->medians !== null) { $dict[$prefix . 'medians'] = $this->medians; }
        if ($this->iqrs    !== null) { $dict[$prefix . 'iqrs']    = $this->iqrs; }
        return $dict;
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->medians = $dict[$prefix . 'medians'] ?? null;
        $this->iqrs    = $dict[$prefix . 'iqrs']    ?? null;
    }
}