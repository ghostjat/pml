<?php
declare(strict_types=1);

namespace Pml\Persisters;

use Pml\Encoding;

/**
 * Persister interface — saves and loads serialized model blobs.
 */
interface Persister
{
    /**
     * Persist a serialized model encoding.
     */
    public function save(Encoding $encoding): void;

    /**
     * Load and return the last saved encoding.
     */
    public function load(): Encoding;
}
