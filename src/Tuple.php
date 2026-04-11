<?php
declare(strict_types=1);

namespace Pml;

/**
 * Immutable fixed-length value container.
 * Zero-allocation destructuring via array access.
 */
final class Tuple implements \ArrayAccess, \Countable
{
    private array $values;

    public function __construct(mixed ...$values)
    {
        $this->values = $values;
    }

    public static function of(mixed ...$values): self
    {
        return new self(...$values);
    }

    public function offsetExists(mixed $offset): bool
    {
        return isset($this->values[$offset]);
    }

    public function offsetGet(mixed $offset): mixed
    {
        return $this->values[$offset] ?? throw new \OutOfBoundsException("Tuple offset {$offset} does not exist.");
    }

    public function offsetSet(mixed $offset, mixed $value): void
    {
        throw new \RuntimeException("Tuple is immutable.");
    }

    public function offsetUnset(mixed $offset): void
    {
        throw new \RuntimeException("Tuple is immutable.");
    }

    public function count(): int
    {
        return count($this->values);
    }

    public function toArray(): array
    {
        return $this->values;
    }
}
