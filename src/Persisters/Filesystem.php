<?php
declare(strict_types=1);

namespace Pml\Persisters;

use Pml\Encoding;

/**
 * Filesystem Persister — saves model blobs to disk atomically.
 *
 * JIT & Memory Optimized:
 * - Writes via a temp file + rename for atomic swap (no torn reads).
 * - file_get_contents reads directly into a PHP string — single syscall.
 */
final class Filesystem implements Persister
{
    public function __construct(private readonly string $path) {}

    public function save(Encoding $encoding): void
    {
        $dir = dirname($this->path);
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }

        $tmp = $this->path . '.tmp.' . bin2hex(random_bytes(4));

        if (file_put_contents($tmp, $encoding->data(), LOCK_EX) === false) {
            throw new \RuntimeException("Failed to write model to: {$tmp}");
        }

        if (!rename($tmp, $this->path)) {
            unlink($tmp);
            throw new \RuntimeException("Failed to atomically rename model file to: {$this->path}");
        }
    }

    public function load(): Encoding
    {
        if (!file_exists($this->path)) {
            throw new \RuntimeException("Model file not found: {$this->path}");
        }

        $data = file_get_contents($this->path);
        if ($data === false) {
            throw new \RuntimeException("Failed to read model file: {$this->path}");
        }

        return Encoding::wrap($data);
    }
}
