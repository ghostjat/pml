<?php
declare(strict_types=1);

namespace Pml\Traits;

use Pml\Loggers\Logger;
use Pml\Loggers\BlackHole;

/**
 * Adds optional logging support. Defaults to BlackHole (no-op) for zero overhead.
 */
trait LoggerAware
{
    private Logger $logger;

    public function setLogger(Logger $logger): void
    {
        $this->logger = $logger;
    }

    protected function log(string $message, string $level = 'info'): void
    {
        ($this->logger ??= new BlackHole())->{$level}($message);
    }
}
