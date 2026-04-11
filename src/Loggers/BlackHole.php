<?php
declare(strict_types=1);

namespace Pml\Loggers;

/**
 * Discards all log messages with zero overhead.
 * JIT optimized: empty final methods are elided entirely by the JIT compiler.
 */
final class BlackHole implements Logger
{
    public function info(string $message): void    {}
    public function warning(string $message): void {}
    public function error(string $message): void   {}
    public function debug(string $message): void   {}
}
