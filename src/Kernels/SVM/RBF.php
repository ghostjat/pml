<?php

declare(strict_types=1);

namespace Pml\Kernels\SVM;

use Pml\Tensor;

/**
 * Radial Basis Function (RBF) / Gaussian Kernel.
 * Projects data into an infinite-dimensional space using exponential distance decay.
 * * JIT & Memory Optimized:
 * - Computes pairwise Euclidean distances using the algebraic expansion: ||A - B||^2 = A^2 + B^2 - 2AB.
 * - Broadcasts row-wise and column-wise sums via AVX2 without single PHP loop.
 */
final class RBF implements Kernel
{
    private float $gamma;

    /**
     * @param float $gamma The kernel coefficient. Defines how far the influence of a single training example reaches.
     */
    public function __construct(float $gamma = 1e-3)
    {
        $this->gamma = $gamma;
    }

    public function compute(Tensor $a, Tensor $b): Tensor
    {
        // 1. A^2: Shape [N, 1]
        $aSq = $a->square()->sumAxis(1)->expandDims(1);
        
        // 2. B^2: Shape [1, M]
        $bSqT = $b->square()->sumAxis(1)->expandDims(0);
        
        // 3. -2 * A * B^T : Shape [N, M]
        $dot = $a->matmul($b->transpose())->mulScalarInplace(-2.0);
        
        // 4. Combine via AVX2 In-Place Broadcasting
        $distSq = $dot->addInplace($aSq)->addInplace($bSqT)->clip(0.0, INF);
        
        // 5. exp(-gamma * distSq)
        return $distSq->mulScalarInplace(-$this->gamma)->exp();
    }
}