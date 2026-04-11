<?php

declare(strict_types=1);

namespace Pml\Interfaces;

use Pml\Tensor;
use Pml\Dataset;

/**
 * Interface for estimators that can output continuous probability distributions.
 */
interface Probabilistic extends Estimator
{
    /**
     * Return the joint probability estimates for each sample.
     * * @param Dataset $dataset
     * @return Tensor A continuous 2D matrix of probabilities.
     */
    public function proba(Dataset $dataset): Tensor;
}