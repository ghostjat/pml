<?php

declare(strict_types=1);

namespace Pml\Interfaces;

use Pml\Dataset;

/**
 * Interface for all Dataset Preprocessing Transformers.
 * JIT Optimized: Strictly accepts and returns zero-copy Datasets.
 */
interface Transformer
{
    /**
     * Learn the parameters from the dataset (e.g., Min, Max, Mean, Categories).
     */
    public function fit(Dataset $dataset): void;

    /**
     * Apply the learned transformation to the dataset, returning a new zero-copy view or modified dataset.
     */
    public function transform(Dataset $dataset): Dataset;

    /**
     * Check if the transformer has been fitted.
     */
    public function fitted(): bool;
}