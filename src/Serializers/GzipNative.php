<?php
declare(strict_types=1);

namespace Pml\Serializers;

use Pml\Encoding;
use Pml\Interfaces\Persistable;

/**
 * Gzip-compressed Native Serializer.
 * Applies gzip level-9 compression on top of PHP serialize — typical 60-80% size reduction.
 *
 * JIT & Memory Optimized:
 * - gzcompress / gzuncompress are single-pass C-level zlib operations.
 * - The intermediate PHP string is the only cross-boundary allocation.
 */
final class GzipNative implements Serializer
{
    public function __construct(private readonly int $level = 9) {}

    public function serialize(Persistable $model): Encoding
    {
        $raw        = serialize($model);
        $compressed = gzcompress($raw, $this->level);
        if ($compressed === false) {
            throw new \RuntimeException("Failed to gzip-compress serialized model.");
        }
        return Encoding::wrap($compressed);
    }

    public function unserialize(Encoding $encoding): Persistable
    {
        $raw = gzuncompress($encoding->data());
        if ($raw === false) {
            throw new \RuntimeException("Failed to decompress model encoding.");
        }
        $model = unserialize($raw);
        if (!$model instanceof Persistable) {
            throw new \RuntimeException("Unserialized object does not implement Persistable.");
        }
        return $model;
    }
}
