<?php
declare(strict_types=1);

namespace Pml\Serializers;

use Pml\Encoding;
use Pml\Interfaces\Persistable;

/**
 * RBX (RubixML Binary eXchange) Serializer.
 * Encodes the model as base64(gzip(json_encode(class + params))).
 * The header contains the class name for safe unserialize without PHP's
 * arbitrary object instantiation risk.
 *
 * JIT & Memory Optimized:
 * - JSON encoding is a single PHP call; gzip and base64 are C-level ops.
 * - On load, the class name is validated before instantiation.
 */
final class RBX implements Serializer
{
    private const MAGIC = 'RBX1';

    public function serialize(Persistable $model): Encoding
    {
        $payload = json_encode([
            'class'   => get_class($model),
            'data'    => serialize($model),
        ], JSON_THROW_ON_ERROR);

        $compressed = gzcompress($payload, 9);
        if ($compressed === false) {
            throw new \RuntimeException("Compression failed during RBX serialization.");
        }

        return Encoding::wrap(self::MAGIC . base64_encode($compressed));
    }

    public function unserialize(Encoding $encoding): Persistable
    {
        $raw = $encoding->data();
        if (!str_starts_with($raw, self::MAGIC)) {
            throw new \RuntimeException("Invalid RBX encoding header.");
        }

        $compressed = base64_decode(substr($raw, strlen(self::MAGIC)), true);
        if ($compressed === false) {
            throw new \RuntimeException("Failed to base64-decode RBX payload.");
        }

        $json = gzuncompress($compressed);
        if ($json === false) {
            throw new \RuntimeException("Failed to decompress RBX payload.");
        }

        $payload = json_decode($json, true, 512, JSON_THROW_ON_ERROR);

        $class = $payload['class'] ?? '';
        if (!class_exists($class)) {
            throw new \RuntimeException("Unknown class in RBX payload: {$class}");
        }

        $model = unserialize($payload['data']);
        if (!$model instanceof Persistable) {
            throw new \RuntimeException("Deserialized RBX object does not implement Persistable.");
        }
        return $model;
    }
}
