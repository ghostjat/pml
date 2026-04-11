<?php
declare(strict_types=1);

namespace Pml\Serializers;

use Pml\Encoding;
use Pml\Interfaces\Persistable;

/**
 * Native PHP Serializer — uses serialize() / unserialize().
 *
 * IMPORTANT: Before serializing, all Tensor parameters are converted to flat
 * PHP arrays via toFlatArray(), then restored via Tensor::fromArray() on load.
 * This is the ONLY point where C↔PHP data transfer happens for persistence.
 */
final class Native implements Serializer
{
    public function serialize(Persistable $model): Encoding
    {
        return Encoding::wrap(serialize($model));
    }

    public function unserialize(Encoding $encoding): Persistable
    {
        $model = unserialize($encoding->data());
        if (!$model instanceof Persistable) {
            throw new \RuntimeException("Unserialized object does not implement Persistable.");
        }
        return $model;
    }
}
