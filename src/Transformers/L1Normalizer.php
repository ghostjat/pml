<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;

/**
 * L1 Normalizer — scales each sample so its L1 norm equals 1.
 * Row-wise: x_i /= sum(|x_i|)
 *
 * JIT & Memory Optimized:
 * - Absolute sum per row via C-level sumAxis; division is in-place broadcast.
 * - No stateful fitting needed — transform is purely sample-local.
 */
final class L1Normalizer implements Transformer
{
    private bool $fitted = false;

    public function fit(Dataset $dataset): void
    {
        $this->fitted = true;               // stateless — fit is a no-op
    }

    public function transform(Dataset $dataset): Dataset
    {
        $x    = $dataset->samples();                               // [N × D]
        $norm = $x->abs()->sumAxis(1)->expandDims(1);             // [N × 1]
        $norm = $norm->clip(1e-8, INF);                           // avoid /0
        return new Dataset($x->div($norm), $dataset->labels());
    }

    public function fitted(): bool { return $this->fitted; }
}
