<?php
declare(strict_types=1);

namespace Pml\Exceptions;

class ClassRevisionMismatch extends RuntimeException
{
    public function __construct(string $class, string $expected, string $given)
    {
        parent::__construct("Class revision mismatch for {$class}: expected revision {$expected}, got {$given}.");
    }
}
