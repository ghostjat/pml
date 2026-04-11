<?php
declare(strict_types=1);

namespace Pml\Serializers;

use Pml\Encoding;
use Pml\Interfaces\Persistable;

/**
 * Serializer interface — encodes and decodes Persistable models.
 */
interface Serializer
{
    public function serialize(Persistable $model): Encoding;
    public function unserialize(Encoding $encoding): Persistable;
}
