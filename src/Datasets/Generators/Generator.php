<?php

declare(strict_types=1);

namespace Pml\Datasets\Generators;

use Pml\Tensor\Dataset;

/**
 * Interface for Synthetic Data Generators.
 */
interface Generator
{
    /**
     * Generate a synthetic dataset.
     * @param int $n The number of samples to generate.
     * @return Dataset
     */
    public function generate(int $n): Dataset;
}