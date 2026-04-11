<?php

declare(strict_types=1);

namespace Pml\Kernels\SVM;

use Pml\Tensor;

/**
 * Interface for Support Vector Machine Kernels.
 * Maps data into high-dimensional feature spaces for non-linear boundaries.
 */
interface Kernel
{
    /**
     * Compute the Gram matrix between two datasets.
     * @param Tensor $a Matrix A of shape [N, D]
     * @param Tensor $b Matrix B of shape [M, D]
     * @return Tensor A continuous C-memory matrix of shape [N, M]
     */
    public function compute(Tensor $a, Tensor $b): Tensor;
}