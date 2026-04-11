<?php
declare(strict_types=1);

namespace Pml\Loggers;

/**
 * Writes log messages to STDOUT with timestamp prefix.
 * JIT optimized: final class, single fwrite() syscall per message.
 */
final class Screen implements Logger
{
    public function __construct(private readonly string $channel = 'PML') {}

    public function info(string $message): void
    {
        $this->write('INFO', $message);
    }

    public function warning(string $message): void
    {
        $this->write('WARNING', $message);
    }

    public function error(string $message): void
    {
        $this->write('ERROR', $message);
    }

    public function debug(string $message): void
    {
        $this->write('DEBUG', $message);
    }

    private function write(string $level, string $message): void
    {
        $timestamp = date('Y-m-d H:i:s');
        fwrite(STDOUT, "[{$timestamp}] [{$this->channel}] [{$level}] {$message}\n");
    }
}
