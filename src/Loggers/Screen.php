<?php
declare(strict_types=1);

namespace Pml\Loggers;

use Psr\Log\LoggerInterface;
use Psr\Log\LoggerTrait;
use Stringable;

/**
 * Writes log messages to STDOUT with timestamp prefix.
 * JIT optimized: final class, single fwrite() syscall per message.
 */
final class Screen implements Logger, LoggerInterface
{
    use LoggerTrait;
    
    public function __construct(private readonly string $channel = 'PML') {}

    /**
     * Logs with an arbitrary level.
     *
     * @param mixed $level
     * @param string|Stringable $message
     * @param mixed[] $context
     */
    public function log($level, string|Stringable $message, array $context = []): void
    {
        $timestamp = date('Y-m-d H:i:s');
        $level = strtoupper((string) $level);
        
        // Format the context array into a string if it exists
        $contextStr = !empty($context) ? ' ' . json_encode($context) : '';

        // Print directly to the terminal screen
        echo "[{$timestamp}] {$level}: {$message}{$contextStr}" . PHP_EOL;
    }

 
}
