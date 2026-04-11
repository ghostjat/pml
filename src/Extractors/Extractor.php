<?php

declare(strict_types=1);

namespace Pml\Extractors;

use Traversable;
use IteratorAggregate;

/**
 * Interface for Data Extractors.
 */
interface Extractor extends IteratorAggregate
{
    /**
     * Yields records from the data source sequentially.
     * @return Traversable<array>
     */
    public function getIterator(): Traversable;
}