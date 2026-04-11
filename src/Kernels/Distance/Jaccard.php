<?php

declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;

/**
 * Jaccard Distance.
 * Measures dissimilarity between sample sets. Perfect for binary/boolean vectors.
 * Distance = 1.0 - (Intersection / Union)
 * * JIT & Memory Optimized:
 * - Employs purely vectorized C-Level boolean masking (`greater()`) to find intersections and unions.
 */
final class Jaccard implements Distance
{
    public function compute(Tensor $a, Tensor $b): Tensor
    {
        $zero = Tensor::zeros(1);
        
        // Ensure inputs are strictly boolean [1.0 or 0.0]
        $aBool = $a->greater($zero);
        $bBool = $b->greater($zero);

        // Intersection: (A AND B). Using multiplication for boolean arrays.
        // Broadcasts [D] against [N, D] instantly.
        $intersection = $aBool->mul($bBool)->sumAxis(1);
        
        // Union: (A OR B).
        $union = $aBool->add($bBool)->greater($zero)->sumAxis(1);
        
        // Distance = 1.0 - (Intersection / Union)
        // Pad union with epsilon to prevent div by zero
        $similarity = $intersection->divInplace($union->addScalarInplace(1e-8));

        return Tensor::ones($b->shape()[0])->subInplace($similarity);
    }
}