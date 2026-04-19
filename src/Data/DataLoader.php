<?php

declare(strict_types=1);

namespace Pml\Data;

use Pml\Dataset;

/**
 * Wraps a Dataset and yields DataBatch objects epoch by epoch.
 *
 * Features:
 * - Optional shuffle before each epoch (calls Dataset::randomize()).
 * - Optional drop_last to discard the final incomplete batch.
 * - Pluggable DataCollator (defaults to DefaultDataCollator).
 * - steps(): int for computing progress bars / LR schedules.
 *
 * Memory: all batches are zero-copy Dataset slices → no PHP heap duplication.
 */
final class DataLoader
{
    private readonly DataCollator $collator;

    public function __construct(
        private readonly Dataset $dataset,
        private readonly int $batchSize = 32,
        private readonly bool $shuffle = false,
        private readonly bool $dropLast = false,
        ?DataCollator $collator = null
    ) {
        $this->collator = $collator ?? new DefaultDataCollator();
    }

    /**
     * Iterate over one epoch.  Shuffles the dataset in-place if requested,
     * then yields one DataBatch per mini-batch.
     *
     * @return \Generator<int, DataBatch>
     */
    public function batches(): \Generator
    {
        if ($this->shuffle) {
            $this->dataset->randomize();
        }

        $step = 0;
        foreach ($this->dataset->batches($this->batchSize) as $rawBatch) {
            // Drop the final undersized batch when requested.
            if ($this->dropLast && $rawBatch->numRows() < $this->batchSize) {
                continue;
            }
            yield $step++ => $this->collator->collate($rawBatch);
        }
    }

    /**
     * Number of complete batches per epoch (used by LR schedulers / progress bars).
     * When drop_last is true, the partial trailing batch is excluded.
     */
    public function steps(): int
    {
        $n = $this->dataset->numRows();
        if ($this->dropLast) {
            return (int) ($n / $this->batchSize);
        }
        return (int) ceil($n / $this->batchSize);
    }

    public function batchSize(): int { return $this->batchSize; }

    public function dataset(): Dataset { return $this->dataset; }
}
