<?php
declare(strict_types=1);

namespace Pml\Traits;

/**
 * Tracks a monotonically increasing revision counter for change detection.
 * Used by PersistentModel to detect stale checkpoints.
 */
trait AutotrackRevisions
{
    private int $revision = 0;

    public function revision(): int
    {
        return $this->revision;
    }

    protected function incrementRevision(): void
    {
        ++$this->revision;
    }
}
