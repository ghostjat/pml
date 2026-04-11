<?php
declare(strict_types=1);

namespace Pml\Backends;

/**
 * Serial Backend — executes tasks one at a time in the current process.
 * Zero-overhead fallback when no parallel extension is available.
 *
 * JIT optimized: final class + tight foreach loop — JIT can inline the closure call.
 */
final class Serial implements Backend
{
    public function run(array $tasks): array
    {
        $results = [];
        foreach ($tasks as $task) {
            $results[] = $task();
        }
        return $results;
    }

    public function workers(): int { return 1; }
}
