<?php

declare(strict_types=1);

namespace Pml\Interfaces;

use Pml\Tensor;

/**
 * Interface for models (like Random Forests or Linear Regression) that can rank feature importance.
 */
interface RanksFeatures extends Estimator
{
    /**
     * Return the importance scores of each feature.
     * * @return Tensor A 1D C-memory float array of importance scores.
     */
    public function featureImportances(): Tensor;
}