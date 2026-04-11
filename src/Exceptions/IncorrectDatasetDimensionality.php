<?php
declare(strict_types=1);

namespace Pml\Exceptions;

class IncorrectDatasetDimensionality extends InvalidArgumentException
{
    public function __construct(int $expected, int $given)
    {
        parent::__construct("Dataset dimensionality mismatch: expected {$expected} features, {$given} given.");
    }
}
