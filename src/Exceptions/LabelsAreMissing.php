<?php
declare(strict_types=1);

namespace Pml\Exceptions;

class LabelsAreMissing extends InvalidArgumentException
{
    public function __construct()
    {
        parent::__construct("Dataset must be labeled (contain target labels).");
    }
}
