<?php

declare(strict_types=1);

namespace Pml\Classic;

use Pml\Tensor;

/**
 * Predictor: any estimator that can produce output predictions.
 *
 * Mirrors sklearn's predict() contract (RegressorMixin / ClassifierMixin).
 */
interface Predictor extends Estimator
{
    /**
     * Predict target values for samples in $X.
     *
     * @param Tensor $X  Feature matrix [n_samples, n_features]
     * @return Tensor    Predictions    [n_samples] (regression) or [n_samples] int (classification)
     */
    public function predict(Tensor $X): Tensor;
}