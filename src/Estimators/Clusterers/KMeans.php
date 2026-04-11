<?php

declare(strict_types=1);

namespace Pml\Estimators\Clusterers;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;

/**
 * K-Means Clustering.
 * Groups data into K distinct clusters.
 * * JIT & Memory Optimized:
 * - 100% Vectorized Expectation-Maximization loop.
 * - Uses AVX2 Boolean Masking to calculate cluster averages without flattening or iterating.
 */
final class KMeans implements Learner
{
    private int $k;
    private int $maxIter;
    private float $tolerance;
    private ?Tensor $centroids = null;

    public function __construct(int $k = 3, int $maxIter = 300, float $tolerance = 1e-4)
    {
        $this->k = $k;
        $this->maxIter = $maxIter;
        $this->tolerance = $tolerance;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $n = $x->shape()[0];

        // 1. Initialize centroids by picking K random samples from the dataset natively in C
        $this->centroids = $x->randomChoice($this->k, false);

        for ($iter = 0; $iter < $this->maxIter; $iter++) {
            // --- EXPECTATION STEP (Assign points to nearest centroid) ---
            
            // Mask of best (lowest) distances initialized to Infinity
            $bestDistances = Tensor::ones($n)->mulScalarInplace(INF);
            $assignments = Tensor::zeros($n);

            for ($c = 0; $c < $this->k; $c++) {
                $centroid = $this->centroids->row($c);
                
                // Vectorized Squared Euclidean Distance: sum((X - C)^2, axis=1)
                $dist = $x->sub($centroid)->square()->sumAxis(1);
                
                // Mask where this centroid is closer than previous ones
                $mask = $bestDistances->greater($dist);
                
                $cVal = Tensor::zeros($n)->addScalarInplace((float) $c);
                
                // In-place pointer updates based on the boolean mask
                $assignments = $mask->where($cVal, $assignments);
                $bestDistances = $mask->where($dist, $bestDistances);
            }

            // --- MAXIMIZATION STEP (Update centroids to the mean of assigned points) ---
            
            $newCentroidRows = [];
            for ($c = 0; $c < $this->k; $c++) {
                $cVal = Tensor::zeros($n)->addScalarInplace((float) $c);
                $mask = $assignments->equal($cVal); // 1.0 where assigned, 0.0 else
                
                $count = $mask->sum();
                if ($count < 1.0) {
                    // Empty cluster, retain old centroid to prevent NaN
                    $newCentroidRows[] = $this->centroids->row($c)->copy()->expandDims(0);
                    continue;
                }
                
                // Expand mask to [N, 1] to broadcast against [N, D] data
                $maskExpanded = $mask->expandDims(1);
                
                // Zero out all points that DO NOT belong to this cluster
                $maskedX = $x->mul($maskExpanded);
                
                // Sum the remaining valid points vertically and divide by the count
                $newCentroid = $maskedX->sumAxis(0)
                    ->mulScalarInplace(1.0 / $count)
                    ->expandDims(0); // Shape [1, D]
                    
                $newCentroidRows[] = $newCentroid;
            }

            // Concatenate the new [1, D] rows into a single [K, D] matrix
            $newCentroidsTensor = Tensor::concat($newCentroidRows, 0);

            // --- CONVERGENCE CHECK ---
            // Max absolute difference between old and new centroids
            $shift = $this->centroids->sub($newCentroidsTensor)->abs()->max();
            $this->centroids = $newCentroidsTensor;

            if ($shift < $this->tolerance) {
                break; // Model converged
            }
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new \RuntimeException("K-Means has not been fitted.");
        }

        $x = $dataset->samples();
        $n = $x->shape()[0];
        
        $bestDistances = Tensor::ones($n)->mulScalarInplace(INF);
        $assignments = Tensor::zeros($n);

        // Vectorized inference
        for ($c = 0; $c < $this->k; $c++) {
            $centroid = $this->centroids->row($c);
            $dist = $x->sub($centroid)->square()->sumAxis(1);
            
            $mask = $bestDistances->greater($dist);
            $cVal = Tensor::zeros($n)->addScalarInplace((float) $c);
            
            $assignments = $mask->where($cVal, $assignments);
            $bestDistances = $mask->where($dist, $bestDistances);
        }

        return $assignments;
    }

    public function trained(): bool
    {
        return $this->centroids !== null;
    }
}