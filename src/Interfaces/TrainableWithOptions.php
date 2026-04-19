<?php

declare(strict_types=1);

namespace Pml\Interfaces;

use Pml\Dataset;

/**
 * Extension of Learner for estimators that accept additional training
 * options beyond the dataset (e.g. epochs, validation set, patience).
 *
 * Sequential implements this so Pipeline can forward its variadic $args
 * cleanly without violating the base Learner contract.
 *
 * PHP's type system requires a discriminated interface rather than widening
 * Learner::train() with variadic params (which would force every Learner
 * implementation to be updated or trigger a fatal declaration-mismatch error).
 */
interface TrainableWithOptions extends Learner
{
    /**
     * Train the model, forwarding additional backend-specific options.
     * Implementations are free to ignore options they do not recognise.
     */
    public function train(Dataset $dataset, mixed ...$options): void;
}
