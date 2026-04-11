<?php
declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;

/**
 * NaN-Safe Euclidean Distance — ignores dimensions where either vector has NaN.
 * Normalises by the number of valid (non-NaN) dimensions.
 *
 * JIT & Memory Optimized:
 * - NaN detection via C-level tensor_isnan; masking via tensor_where.
 * - Valid dimension count is a single C sum reduction.
 */
final class NaNSafe implements Distance
{
    public function compute(Tensor $a, Tensor $b): float
    {
        $validA = $a->isNan()->logicalNot();
        $validB = $b->isNan()->logicalNot();
        $valid  = $validA->mul($validB);                           // 1 where both valid

        $nValid = $valid->sum();
        if ($nValid < 1.0) return 0.0;

        $zero  = Tensor::zeros($a->size());
        $safeA = $a->where($a, $zero)->mul($valid);
        $safeB = $b->where($b, $zero)->mul($valid);

        $diff  = $safeA->sub($safeB);
        return sqrt($diff->dot($diff) / $nValid);
    }
}
