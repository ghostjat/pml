<?php

declare(strict_types=1);

namespace Pml\Data;

use Pml\Tensor;

/**
 * Immutable container for a single mini-batch of data.
 *
 * Holds the raw input Tensor, optional label Tensor, and arbitrary
 * metadata produced by a DataCollator (e.g. attention masks, sequence
 * lengths, sample weights).  All Tensors are zero-copy references —
 * this object owns no C memory of its own.
 */
final class DataBatch
{
    /**
     * @param Tensor       $inputs  Feature matrix [batch × features] or sequence tensor.
     * @param Tensor|null  $labels  Target vector/matrix [batch] or [batch × outputs].
     * @param array<string,mixed> $meta  Collator-supplied extras (masks, weights, …).
     */
    public function __construct(
        private readonly Tensor $inputs,
        private readonly ?Tensor $labels = null,
        private readonly array $meta = []
    ) {}

    public function inputs(): Tensor { return $this->inputs; }

    public function labels(): ?Tensor { return $this->labels; }

    /** @return array<string,mixed> */
    public function meta(): array { return $this->meta; }

    public function hasLabels(): bool { return $this->labels !== null; }

    /** Number of samples in this batch (leading dimension of inputs). */
    public function size(): int { return $this->inputs->shape()[0]; }
}
