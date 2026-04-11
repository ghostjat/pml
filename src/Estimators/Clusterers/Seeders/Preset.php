<?php
declare(strict_types=1);

namespace Pml\Estimators\Clusterers\Seeders;

use Pml\Tensor;
use Pml\Dataset;

/**
 * Preset seeder — use user-supplied centroids as the initial seed.
 * Allows warm-starting KMeans from a known-good prior solution.
 */
final class Preset implements Seeder
{
    /**
     * @param Tensor $centroids  Pre-computed centroid matrix [k × D].
     */
    public function __construct(private readonly Tensor $centroids) {}

    public function seed(Dataset $dataset, int $k): Tensor
    {
        $shape = $this->centroids->shape();
        if ($shape[0] < $k) {
            throw new \InvalidArgumentException(
                "Preset centroids have {$shape[0]} rows, but {$k} seeds are required."
            );
        }

        // Return the first k rows as a zero-copy view
        return $this->centroids->slice(0, 0, $k);
    }
}
