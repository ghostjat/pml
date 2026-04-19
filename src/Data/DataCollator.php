<?php

declare(strict_types=1);

namespace Pml\Data;

use Pml\Dataset;

/**
 * Converts a raw Dataset mini-batch into a DataBatch ready for the model.
 *
 * Implement this to pad sequences, build attention masks, apply
 * per-sample weights, or perform any other batch-level collation.
 * The returned DataBatch must not copy Tensor memory — use slices and
 * views where possible.
 */
interface DataCollator
{
    /**
     * @param Dataset $batch  A single mini-batch Dataset slice.
     * @return DataBatch      Collated, model-ready batch.
     */
    public function collate(Dataset $batch): DataBatch;
}
