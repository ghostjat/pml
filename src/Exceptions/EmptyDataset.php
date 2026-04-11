<?php
declare(strict_types=1);

namespace Pml\Exceptions;

class EmptyDataset extends InvalidArgumentException
{
    public function __construct(string $context = '')
    {
        parent::__construct("Dataset must not be empty" . ($context !== '' ? ": {$context}" : '.'));
    }
}
