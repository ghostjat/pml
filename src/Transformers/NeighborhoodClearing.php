<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Neighborhood Clearing (Edited Nearest Neighbors).
 * Undersamples the majority class by deleting points that are surrounded by a minority class.
 * * JIT & Memory Optimized:
 * - 100% Vectorized 3-NN search via OpenBLAS distance matrices.
 */
final class NeighborhoodClearing implements Transformer
{
    private int $k;

    public function __construct(int $k = 3)
    {
        $this->k = $k;
    }

    public function fit(Dataset $dataset): void
    {
        // Stateless Transformer
    }

    public function transform(Dataset $dataset): Dataset
    {
        $x = $dataset->samples();
        $y = $dataset->labels();
        $n = $x->shape()[0];

        $labelsFlat = $y->toFlatArray();
        $counts = array_count_values($labelsFlat);
        $majorityClass = (float) array_search(max($counts), $counts);

        // 1. Full Pairwise Euclidean Distance Matrix natively in C
        $xSq = $x->square()->sumAxis(1)->expandDims(1);
        $distSq = $x->matmul($x->transpose())
                     ->mulScalarInplace(-2.0)
                     ->addInplace($xSq)
                     ->addInplace($xSq->transpose())
                     ->clip(0.0, INF);

        // Ignore self-distance
        $distSq->addInplace(Tensor::eye($n)->mulScalarInplace(INF));

        // 2. Find Top K Neighbors for all points simultaneously
        // argsort() returns indices. We slice the top K.
        $kIndices = $distSq->argsort()->slice(1, 0, $this->k);

        $toKeep = [];
        $kIndicesFlat = $kIndices->toFlatArray();

        // 3. Clear the neighborhood
        for ($i = 0; $i < $n; $i++) {
            if ($labelsFlat[$i] !== $majorityClass) {
                // Always keep minority class points
                $toKeep[] = $i;
                continue;
            }

            // Find the classes of the K-nearest neighbors
            $neighborClasses = [];
            for ($j = 0; $j < $this->k; $j++) {
                $nIdx = (int) $kIndicesFlat[$i * $this->k + $j];
                $neighborClasses[] = $labelsFlat[$nIdx];
            }

            // If the majority of neighbors do NOT belong to the majority class, this point is noisy!
            $neighborCounts = array_count_values($neighborClasses);
            arsort($neighborCounts);
            $predClass = array_key_first($neighborCounts);

            if ($predClass === $majorityClass) {
                $toKeep[] = $i; // Safe, keep it
            }
        }

        $idxT = Tensor::fromArray($toKeep);
        return new Dataset($x->take($idxT, 0), $y->take($idxT, 0));
    }

    public function fitted(): bool
    {
        return true;
    }
}