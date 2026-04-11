<?php

declare(strict_types=1);

namespace Pml\Estimators\Clusterers\Seeders;

use Pml\Tensor;
use Pml\Dataset;

/**
 * Interface for Clustering Seeders.
 * Responsible for initializing the starting centroids for iterative clusterers.
 */
interface Seeder
{
    /**
     * Generate the initial K centroids from the dataset.
     * @param Dataset $dataset
     * @param int $k
     * @return Tensor A [K, D] continuous C-memory pointer containing the initialized centroids.
     */
    public function seed(Dataset $dataset, int $k): Tensor;
}