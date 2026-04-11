<?php

declare(strict_types=1);

namespace Pml\Interfaces;

use Pml\Tensor;
use Pml\Dataset;

/**
 * The base interface for all Machine Learning models.
 * JIT Optimized: Strictly typed to guarantee zero-copy C-memory interactions.
 */
interface Estimator
{
    /**
     * Make predictions on a dataset.
     * * @param Dataset $dataset The testing/inference data.
     * @return Tensor A continuous C-memory pointer containing the predictions.
     */
    public function predict(Dataset $dataset): Tensor;
}