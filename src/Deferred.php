<?php
declare(strict_types=1);

namespace Pml;

/**
 * Lazy-evaluation wrapper for deferred computation.
 * The callable is invoked at most once; result is cached in PHP userland.
 * JIT optimized: final class + typed property avoids late-binding overhead.
 */
final class Deferred
{
    private bool $resolved = false;
    private mixed $result  = null;

    public function __construct(private readonly \Closure $callback) {}

    public static function wrap(\Closure $callback): self
    {
        return new self($callback);
    }

    /**
     * Resolve and cache the result. Subsequent calls return the cached value.
     */
    public function resolve(): mixed
    {
        if (!$this->resolved) {
            $this->result   = ($this->callback)();
            $this->resolved = true;
        }
        return $this->result;
    }

    public function resolved(): bool
    {
        return $this->resolved;
    }
}
