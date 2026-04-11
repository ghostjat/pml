<?php

declare(strict_types=1);

namespace Pml\Interfaces;

use Pml\Dataset;

/**
 * Interface for estimators that can be trained iteratively (mini-batch/online learning).
 * Perfectly paired with the Dataset->batches() zero-copy generator.
 */
interface Online extends Learner
{
    /**
     * Perform a partial training update using a mini-batch of data.
     * * @param Dataset $dataset A zero-copy mini-batch slice.
     */
    public function partial(Dataset $dataset): void;
}