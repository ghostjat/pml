<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Tomek Links Undersampling.
 * Detects pairs of opposing instances that are each other's nearest neighbors and deletes the majority point.
 * * JIT & Memory Optimized:
 * - Calculates the 1-NN of the entire dataset simultaneously using `argmin()` natively in C.
 */
final class TomekLinks implements Transformer
{
    public function fit(Dataset $dataset): void
    {
        // Stateless Transformer
    }

    public function transform(Dataset $dataset): Dataset
    {
        $x = $dataset->samples();
        $y = $dataset->labels();
        $n = $x->shape()[0];

        // 1. Calculate Full Pairwise Distance Matrix
        $xSq = $x->square()->sumAxis(1)->expandDims(1);
        $distSq = $x->matmul($x->transpose())
                     ->mulScalarInplace(-2.0)
                     ->addInplace($xSq)
                     ->addInplace($xSq->transpose())
                     ->clip(0.0, INF);

        // 2. Ignore self-distances
        $infDiag = Tensor::eye($n)->mulScalarInplace(INF);
        $distSq->addInplace($infDiag);

        // 3. Find index of 1st Nearest Neighbor for every single point instantly in C
        $nnIndices = $distSq->argmin(1)->toFlatArray();
        $labels = $y->toFlatArray();

        // 4. Identify Tomek Links
        $toKeep = array_fill(0, $n, true);
        
        // Find majority class to safely drop points
        $counts = array_count_values($labels);
        $majorityClass = array_search(max($counts), $counts);

        for ($i = 0; $i < $n; $i++) {
            $j = $nnIndices[$i];
            
            // If they are mutual nearest neighbors AND have different classes
            if ($nnIndices[$j] === $i && $labels[$i] !== $labels[$j]) {
                // Drop the one belonging to the majority class
                if ($labels[$i] === $majorityClass) $toKeep[$i] = false;
                if ($labels[$j] === $majorityClass) $toKeep[$j] = false;
            }
        }

        // 5. Extract cleaned dataset via zero-copy C-pointer masking
        $keepIndices = [];
        foreach ($toKeep as $idx => $keep) {
            if ($keep) $keepIndices[] = $idx;
        }

        $idxT = Tensor::fromArray($keepIndices);
        
        return new Dataset($x->take($idxT, 0), $y->take($idxT, 0));
    }

    public function fitted(): bool
    {
        return true;
    }
}