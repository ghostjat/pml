<?php

declare(strict_types=1);

namespace Pml\Interfaces;

use Pml\Dataset;

/**
 * Interface for models that can evaluate their own performance.
 */
interface Scoring extends Estimator
{
    /**
     * Compute a performance score on the given validation dataset.
     * Calculated natively in C via AVX2 SIMD for lightning speed.
     * * @param Dataset $dataset
     * @return float The raw evaluation score.
     */
    public function score(Dataset $dataset): float;
}