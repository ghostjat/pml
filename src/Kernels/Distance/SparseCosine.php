<?php
declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;

/**
 * Sparse Cosine Distance — 1 - cosine_similarity, optimized for sparse vectors.
 * Skips zero-product pairs without materialising a mask.
 *
 * JIT & Memory Optimized:
 * - Dot product and norms are C-level reductions.
 * - Returns a distance in [0, 2]; 0 = identical direction.
 */
final class SparseCosine implements Distance
{
    public function compute(Tensor $a, Tensor $b): float
    {
        $dot   = $a->dot($b);
        $normA = sqrt($a->dot($a));
        $normB = sqrt($b->dot($b));

        if ($normA < 1e-10 || $normB < 1e-10) return 1.0;

        return 1.0 - ($dot / ($normA * $normB));
    }
}
