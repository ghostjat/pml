<?php

declare(strict_types=1);

namespace Pml\Interfaces;

use Pml\Dataset;

/**
 * Interface for estimators that can be trained on a dataset.
 */
interface Learner extends Estimator
{
    /**
     * Train the model natively in C using the provided FFI dataset.
     * * @param Dataset $dataset The training data.
     */
    public function train(Dataset $dataset, mixed ...$options): void;

    /**
     * Check if the estimator has been trained.
     */
    public function trained(): bool;
}