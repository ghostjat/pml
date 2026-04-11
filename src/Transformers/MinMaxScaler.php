<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Min-Max Scaler.
 * Scales features to a specific range (typically 0.0 to 1.0).
 * * JIT & Memory Optimized:
 * - Uses C-level axis reductions to find Min and Max instantly.
 * - Applies scaling via zero-allocation In-Place broadcasting.
 */
final class MinMaxScaler implements Transformer
{
    private array $featureRange;
    private ?Tensor $min = null;
    private ?Tensor $range = null;

    public function __construct(float $min = 0.0, float $max = 1.0)
    {
        if ($min >= $max) {
            throw new \InvalidArgumentException("Min value must be strictly less than Max value.");
        }
        $this->featureRange = [$min, $max];
    }

    public function fit(Dataset $dataset): void
    {
        $x = $dataset->samples();

        // Find the min and max for each feature column (Axis 0)
        $this->min = $x->minAxis(0);
        $max = $x->maxAxis(0);

        // Compute the range: (Max - Min)
        $this->range = $max->sub($this->min);

        // Numerical Stability: Prevent division by zero for constant features.
        // Clip the minimum range to a tiny epsilon natively in C.
        $this->range = $this->range->clip(1e-8, INF);
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new \RuntimeException("Scaler has not been fitted.");
        }

        $x = $dataset->samples();

        // 1. Standardize to [0, 1]: X_std = (X - X.min) / Range
        // Creates a new tensor to avoid mutating the original dataset's memory
        $scaled = $x->sub($this->min)->divInplace($this->range);

        // 2. Scale to target range: X_scaled = X_std * (max - min) + min
        if ($this->featureRange[0] !== 0.0 || $this->featureRange[1] !== 1.0) {
            $scaleDiff = $this->featureRange[1] - $this->featureRange[0];
            $scaled->mulScalarInplace($scaleDiff)->addScalarInplace($this->featureRange[0]);
        }

        return new Dataset($scaled, $dataset->labels());
    }

    public function fitted(): bool
    {
        return $this->min !== null && $this->range !== null;
    }
}