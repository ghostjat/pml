<?php

declare(strict_types=1);

namespace Pml\Interfaces;

use Psr\Log\LoggerInterface;

/**
 * Interface for estimators that emit training progress logs.
 */
interface Verbose
{
    /**
     * Attach a PSR-3 compliant logger to the model.
     * * @param LoggerInterface $logger
     */
    public function setLogger(LoggerInterface $logger): void;
}