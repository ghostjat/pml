<?php
declare(strict_types=1);

namespace Pml\Loggers;

/**
 * PSR-3 compatible logger interface.
 */
interface Logger
{
    public function info(string $message): void;
    public function warning(string $message): void;
    public function error(string $message): void;
    public function debug(string $message): void;
}
