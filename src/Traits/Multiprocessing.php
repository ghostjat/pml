<?php
declare(strict_types=1);

namespace Pml\Traits;

use Pml\Backends\Backend;
use Pml\Backends\Serial;

/**
 * Injects a parallel Backend dependency.
 */
trait Multiprocessing
{
    private Backend $backend;

    public function setBackend(Backend $backend): void
    {
        $this->backend = $backend;
    }

    protected function backend(): Backend
    {
        return $this->backend ??= new Serial();
    }
}
