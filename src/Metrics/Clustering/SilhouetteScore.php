<?php

declare(strict_types=1);

namespace Pml\Metrics\Clustering;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Silhouette Score (Centroid Approximation).
 * Evaluates clustering performance measuring cohesion versus separation.
 * * JIT & Memory Optimized:
 * - Exact pairwise distances create O(N^2) matrices that crash memory (e.g. 100k samples = 40GB).
 * - This implementation computes distances against K-Centroids, resolving in O(N*K) natively in C.
 */
final class SilhouetteScore implements Metric
{
    /**
     * @param Tensor $predictions The cluster assignments from the model.
     * @param Tensor $samples The continuous feature matrix (Dataset->samples()).
     */
    public function score(Tensor $predictions, Tensor $samples): float
    {
        $n = $samples->shape()[0];
        $k = (int) $predictions->max() + 1;

        if ($k <= 1 || $k >= $n) return 0.0;

        // 1. Calculate Centroids dynamically in C using boolean masks
        $centroids = [];
        for ($c = 0; $c < $k; $c++) {
            $cVal = Tensor::zeros($n)->addScalarInplace((float)$c);
            $mask = $predictions->equal($cVal);
            $count = $mask->sum();
            
            if ($count > 0.0) {
                $maskExpanded = $mask->expandDims(1);
                $centroids[$c] = $samples->mul($maskExpanded)->sumAxis(0)->mulScalarInplace(1.0 / $count);
            } else {
                // Empty cluster fallback
                $centroids[$c] = Tensor::zeros($samples->shape()[1]); 
            }
        }

        // 2. Broadcast distances from all points to all centroids
        $a = Tensor::zeros($n); // Cohesion (distance to own centroid)
        $b = Tensor::ones($n)->mulScalarInplace(INF); // Separation (distance to nearest other centroid)

        for ($c = 0; $c < $k; $c++) {
            // Euclidean Distance to centroid C: shape [N]
            $dist = $samples->sub($centroids[$c])->square()->sumAxis(1)->sqrt();

            $cVal = Tensor::zeros($n)->addScalarInplace((float)$c);
            $inClusterMask = $predictions->equal($cVal);
            $outClusterMask = $inClusterMask->logicalNot();

            // a(i) = distance to own cluster centroid
            $a = $inClusterMask->where($dist, $a);

            // b(i) = min distance to all other cluster centroids
            $validB = $outClusterMask->where($dist, Tensor::ones($n)->mulScalarInplace(INF));
            $b = $b->less($validB)->where($b, $validB);
        }

        // 3. Silhouette Formula: (b - a) / max(a, b)
        $diff = $b->sub($a);
        
        // C-Level hardware fallback to prevent division by zero
        $maxAB = $a->greater($b)->where($a, $b)->addScalarInplace(1e-8);

        return $diff->divInplace($maxAB)->mean();
    }
}