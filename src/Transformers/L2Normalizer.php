<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;

/**
 * L2 Normalizer — scales each sample to unit Euclidean norm.
 * Row-wise: x_i /= ||x_i||_2
 *
 * JIT & Memory Optimized:
 * - ||x||_2 computed via square → sumAxis → sqrt — all in C.
 * - Division is a single in-place broadcast (no PHP loop).
 */
final class L2Normalizer implements Transformer
{
    private bool $fitted = false;

    public function fit(Dataset $dataset): void
    {
        $this->fitted = true;
    }

    public function transform(Dataset $dataset): Dataset
    {
        $x    = $dataset->samples();                               // [N × D]
        $norm = $x->square()->sumAxis(1)->sqrt()->expandDims(1);  // [N × 1]
        $norm = $norm->clip(1e-8, INF);
        return new Dataset($x->div($norm), $dataset->labels());
    }

    public function fitted(): bool { return $this->fitted; }
}
