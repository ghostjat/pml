<?php

declare(strict_types=1);

namespace Pml\Kernels\SVM;

use Pml\Tensor;

/**
 * Linear Kernel.
 * Computes the standard dot product between two matrices.
 * * JIT & Memory Optimized:
 * - Direct OpenBLAS execution via `matmul`.
 */
final class Linear implements Kernel
{
    public function compute(Tensor $a, Tensor $b): Tensor
    {
        return $a->matmul($b->transpose());
    }
}