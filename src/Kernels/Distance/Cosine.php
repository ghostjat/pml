<?php

declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;

/**
 * Cosine Distance.
 * Measures orientation rather than magnitude. Excellent for NLP text embeddings.
 * Distance = 1 - Cosine Similarity
 * * JIT & Memory Optimized:
 * - Highly parallelized `matmul` and Euclidean norm (`square()->sum()->sqrt()`) broadcasting.
 */
final class Cosine implements Distance
{
    public function compute(Tensor $a, Tensor $b): Tensor
    {
        // 1. Dot Product: A * B^T
        $dot = $b->matmul($a->transpose())->squeeze();
        
        // 2. Magnitudes: ||A|| and ||B||
        $normA = $a->square()->sum()->sqrt();
        $normB = $b->square()->sumAxis(1)->sqrt();
        
        // 3. Similarity: Dot / (||A|| * ||B||)
        $sim = $dot->divInplace($normB->mulScalarInplace($normA));
        
        // 4. Distance = 1.0 - Similarity
        return Tensor::ones($b->shape()[0])->subInplace($sim);
    }
}