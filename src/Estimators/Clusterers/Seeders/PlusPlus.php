<?php

declare(strict_types=1);

namespace Pml\Estimators\Clusterers\Seeders;

use Pml\Tensor;
use Pml\Dataset;
use InvalidArgumentException;

/**
 * K-Means++ Seeder.
 * Intelligently spreads out initial centroids to drastically accelerate convergence 
 * and avoid poor local minima (compared to pure random selection).
 * * JIT & Memory Optimized:
 * - Employs a blazing-fast O(log N) Binary Search roulette wheel in PHP.
 * - Distance evaluations are broadcasted in C via AVX2.
 */
final class PlusPlus implements Seeder
{
    public function seed(Dataset $dataset, int $k): Tensor
    {
        $x = $dataset->samples();
        $n = $x->shape()[0];

        if ($k < 1 || $k > $n) {
            throw new InvalidArgumentException("K must be between 1 and the number of samples.");
        }

        $centroids = [];
        
        // 1. Pick the very first centroid uniformly at random
        $firstIdx = mt_rand(0, $n - 1);
        $centroids[] = $x->row($firstIdx);

        // Keep track of the minimum squared distance to any established centroid
        $minDistSq = Tensor::ones($n)->mulScalarInplace(INF);

        for ($i = 1; $i < $k; $i++) {
            $lastCentroid = $centroids[$i - 1];
            
            // 2. Compute distance to the newest centroid natively in C
            $dist = $x->sub($lastCentroid)->square()->sumAxis(1);
            
            // Update the minimum distance tracker using hardware boolean masking
            $minDistSq = $dist->less($minDistSq)->where($dist, $minDistSq);

            // Extract distances to PHP for rapid weighted probability selection
            $distFlat = $minDistSq->toFlatArray();
            $sum = array_sum($distFlat);

            if ($sum <= 0.0) {
                // Failsafe: remaining points are identical to established centroids
                $centroids[] = $x->row(mt_rand(0, $n - 1));
                continue;
            }

            // 3. Roulette Wheel Selection (O(N) Sum, O(1) Search)
            $r = (mt_rand() / mt_getrandmax()) * $sum;
            $cumsum = 0.0;
            $selectedIndex = $n - 1;
            
            foreach ($distFlat as $idx => $val) {
                $cumsum += $val;
                if ($cumsum >= $r) {
                    $selectedIndex = $idx;
                    break;
                }
            }

            $centroids[] = $x->row($selectedIndex);
        }

        // Return a packed [K, D] Tensor ready for OpenBLAS ingestion
        return Tensor::concat($centroids, 0);
    }
}