<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Boolean Converter — clamps all feature values to {0.0, 1.0}.
 * Values > 0 become 1.0; values <= 0 become 0.0.
 *
 * JIT & Memory Optimized: single C-level greaterScalar call — no PHP loops.
 */
final class BooleanConverter implements Transformer
{
    private bool $fitted = false;

    public function fit(Dataset $dataset): void { $this->fitted = true; }

    public function transform(Dataset $dataset): Dataset
    {
        return new Dataset(
            $dataset->samples()->greaterScalar(0.0),              // [N × D] 0/1 float
            $dataset->labels()
        );
    }

    public function fitted(): bool { return $this->fitted; }
}
