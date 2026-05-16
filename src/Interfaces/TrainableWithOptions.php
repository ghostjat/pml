<?php

declare(strict_types=1);

namespace Pml\Interfaces;

use Pml\Dataset;

/**
 * @deprecated Use Learner directly — its train() now accepts mixed ...$options (§16).
 *
 * TrainableWithOptions was a sub-interface of Learner that widened train() to
 * accept variadic options for neural-network backends.  That widening has been
 * merged into Learner itself, making this interface redundant.  It is kept as a
 * transparent alias so existing code that type-checks for TrainableWithOptions
 * keeps working without change.  It will be removed in the next major version.
 */
interface TrainableWithOptions extends Learner
{
    // Intentionally empty — all semantics are now in Learner::train().
}
