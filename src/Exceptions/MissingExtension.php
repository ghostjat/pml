<?php
declare(strict_types=1);

namespace Pml\Exceptions;

class MissingExtension extends RuntimeException
{
    public function __construct(string $extension)
    {
        parent::__construct("Required PHP extension '{$extension}' is not loaded.");
    }
}
