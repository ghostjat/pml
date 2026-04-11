<?php

declare(strict_types=1);

namespace Pml\Metrics;

use Pml\Tensor;

/**
 * Interface for all evaluation metrics.
 * JIT Optimized: Strictly accepts FFI Tensors for zero-copy evaluation.
 */
interface Metric
{
    /**
     * Compute the evaluation score.
     * * @param Tensor $predictions The model's predictions.
     * @param Tensor $labels The ground truth labels.
     * @return float The calculated score.
     */
    public function score(Tensor $predictions, Tensor $labels): float;
}