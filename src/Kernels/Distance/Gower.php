<?php
declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;

/**
 * Gower Distance — mixed-type distance averaging per-feature dissimilarities.
 * For continuous features: |a_i - b_i| / range_i.
 * For binary features: 1 if a_i != b_i, else 0.
 *
 * JIT & Memory Optimized:
 * - Continuous dissimilarity computed entirely in C (abs + div + mean).
 * - Requires fit() to learn per-feature ranges before use.
 */
final class Gower implements Distance
{
    private ?Tensor $ranges = null;

    /**
     * Learn per-feature ranges from a training matrix [N × D].
     */
    public function fit(Tensor $x): void
    {
        $this->ranges = $x->maxAxis(0)->sub($x->minAxis(0))->clip(1e-10, INF);
    }

    public function compute(Tensor $a, Tensor $b): float
    {
        if ($this->ranges === null) {
            throw new \RuntimeException("Gower distance must be fitted before use.");
        }
        return $a->sub($b)->abs()->div($this->ranges)->mean();
    }
}
