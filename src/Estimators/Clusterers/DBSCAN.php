<?php

declare(strict_types=1);

namespace Pml\Estimators\Clusterers;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
 * Groups dense continuous regions of data into clusters and flags sparse points as noise.
 * * JIT & Memory Optimized:
 * - Vectorizes O(N) neighborhood queries using AVX2 Euclidean distance broadcasting.
 * - Extracts neighborhood masks via C-level boolean indexing (`booleanIndex`).
 * - Transductive inference maps new samples to cached core points efficiently.
 */
final class DBSCAN implements Learner
{
    private float $epsilon;
    private int $minSamples;

    private ?Tensor $clusterLabels = null;
    private ?Tensor $coreSamples = null;
    private ?Tensor $coreLabels = null;

    /**
     * @param float $epsilon The maximum distance between two samples to be considered neighbors.
     * @param int $minSamples The number of samples required in a neighborhood to define a core point.
     */
    public function __construct(float $epsilon = 0.5, int $minSamples = 5)
    {
        $this->epsilon = $epsilon;
        $this->minSamples = $minSamples;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $n = $x->shape()[0];

        $visited = array_fill(0, $n, false);
        
        // -1.0 represents Noise/Outliers
        $labelsFlat = array_fill(0, $n, -1.0); 
        $clusterId = 0;
        $coreIndices = [];

        // Cache the Squared Epsilon natively in C to avoid repeating the operation
        $epsSq = Tensor::zeros(1)->addScalarInplace($this->epsilon * $this->epsilon);
        
        // Generate an integer index tensor [0, 1 ... N-1] for instant C-level index extraction
        $allIndices = Tensor::range(0, $n - 1, 1);

        for ($i = 0; $i < $n; $i++) {
            if ($visited[$i]) {
                continue;
            }
            $visited[$i] = true;

            // Highly Optimized Vectorized Query
            $neighbors = $this->regionQuery($x, $i, $epsSq, $allIndices);

            if (count($neighbors) < $this->minSamples) {
                $labelsFlat[$i] = -1.0; // Noise
            } else {
                $coreIndices[] = $i;
                $this->expandCluster(
                    $x, $i, $neighbors, $clusterId, 
                    $visited, $labelsFlat, $epsSq, $allIndices, $coreIndices
                );
                $clusterId++;
            }
        }

        $this->clusterLabels = Tensor::fromArray($labelsFlat);

        // Cache core samples natively for fast transductive prediction/inference
        if (!empty($coreIndices)) {
            $coreIndicesT = Tensor::fromArray($coreIndices);
            $this->coreSamples = $x->take($coreIndicesT, 0);
            $this->coreLabels = $this->clusterLabels->take($coreIndicesT, 0);
        }
    }

    /**
     * Finds all neighbors within the epsilon radius.
     * Executes entirely in OpenBLAS C-Memory.
     */
    private function regionQuery(Tensor $x, int $idx, Tensor $epsSq, Tensor $allIndices): array
    {
        // 1. Extract the current point [1, D]
        $point = $x->row($idx);
        
        // 2. Broadcast Subtraction & Squared Euclidean Distance: sum((X - p)^2, axis=1)
        $sqDist = $x->sub($point)->square()->sumAxis(1);
        
        // 3. Boolean mask for points within epsilon radius
        $mask = $sqDist->lessEqual($epsSq);
        
        // 4. Return the exact matching integer indices to PHP instantly
        return $allIndices->booleanIndex($mask)->toFlatArray();
    }

    /**
     * Graph traversal to expand the dense region into a unified cluster.
     */
    private function expandCluster(
        Tensor $x, int $pointIdx, array $neighbors, int $clusterId, 
        array &$visited, array &$labelsFlat, 
        Tensor $epsSq, Tensor $allIndices, array &$coreIndices
    ): void {
        $labelsFlat[$pointIdx] = (float) $clusterId;
        
        // Use array keys as an O(1) hash map to manage the expanding seed set
        $seedSet = array_flip($neighbors);
        unset($seedSet[$pointIdx]);

        while (!empty($seedSet)) {
            $currentP = array_key_first($seedSet);
            unset($seedSet[$currentP]);

            if (!$visited[$currentP]) {
                $visited[$currentP] = true;
                $currentNeighbors = $this->regionQuery($x, $currentP, $epsSq, $allIndices);

                if (count($currentNeighbors) >= $this->minSamples) {
                    $coreIndices[] = $currentP;
                    foreach ($currentNeighbors as $nIdx) {
                        if (!$visited[$nIdx]) {
                            $seedSet[$nIdx] = true;
                        }
                    }
                }
            }

            if ($labelsFlat[$currentP] === -1.0) {
                $labelsFlat[$currentP] = (float) $clusterId;
            }
        }
    }

    /**
     * Transductive inference: Assigns new data to the nearest core sample's cluster.
     */
    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("DBSCAN has not been trained.");
        }

        $testX = $dataset->samples();
        $nTest = $testX->shape()[0];

        // If no dense regions were found during training, everything is noise (-1.0)
        if ($this->coreSamples === null) {
            return Tensor::zeros($nTest)->addScalarInplace(-1.0);
        }

        $preds = [];
        $flatCoreLabels = $this->coreLabels->toFlatArray();
        $maxDistSq = $this->epsilon * $this->epsilon;

        // JIT prediction loop mapping to cached Core Points
        for ($i = 0; $i < $nTest; $i++) {
            $point = $testX->row($i);
            
            // Distance to all core samples natively in C
            $sqDist = $this->coreSamples->sub($point)->square()->sumAxis(1);
            $minIdx = $sqDist->argmin();
            
            // Extract the single minimum distance
            $minDist = $sqDist->toFlatArray()[$minIdx];

            if ($minDist <= $maxDistSq) {
                $preds[] = $flatCoreLabels[$minIdx];
            } else {
                $preds[] = -1.0; // Point is too far from any cluster
            }
        }

        return Tensor::fromArray($preds);
    }

    public function trained(): bool
    {
        return $this->clusterLabels !== null;
    }

    /**
     * Returns the cluster labels for the training set natively assigned during fit().
     * * @return Tensor A continuous C-memory pointer containing the labels (-1.0 for noise).
     */
    public function labels(): ?Tensor
    {
        return $this->clusterLabels;
    }
}