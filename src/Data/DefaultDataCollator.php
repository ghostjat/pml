<?php

declare(strict_types=1);

namespace Pml\Data;

use Pml\Dataset;

/**
 * Pass-through collator.
 *
 * Wraps the batch's sample Tensor and label Tensor directly into a
 * DataBatch without any extra processing.  Suitable for fixed-length,
 * homogeneous float32 datasets (tabular, pre-embedded, etc.).
 */
final class DefaultDataCollator implements DataCollator
{
    public function collate(Dataset $batch): DataBatch
    {
        return new DataBatch(
            inputs: $batch->samples(),
            labels: $batch->labels(),
        );
    }
}
