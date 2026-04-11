<?php

declare(strict_types=1);

namespace Pml\Metrics\Clustering;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * V-Measure Clustering Metric.
 * The harmonic mean between Homogeneity and Completeness.
 * * JIT & Memory Optimized:
 * - Extracts Contingency/Intersection tables via an AVX2 Math-Hashing trick.
 * - Resolves Mutual Information in O(K) loops rather than O(N) loops.
 */
final class VMeasure implements Metric
{
    public function score(Tensor $predictions, Tensor $labels): float
    {
        $n = (float) $labels->size();
        if ($n === 0.0) return 0.0;

        $preds = $predictions->round();
        $y = $labels->round();
        
        $numClasses = (int) max($y->max(), $preds->max()) + 1;

        // THE AVX2 HASHING TRICK:
        // Hash formula: (y_true * numClasses) + y_pred
        // Creates a unique intersection identifier for the Contingency table
        $hash = $y->mulScalar((float) $numClasses)->addInplace($preds);
        
        // Bincount safely summarizes the entire dataset in C-memory
        $contingency = $hash->bincount()->toFlatArray();
        $predCounts = $preds->bincount()->toFlatArray();
        $labelCounts = $y->bincount()->toFlatArray();

        $mutualInfo = 0.0;

        // O(K) Mutual Information extraction
        foreach ($contingency as $hashVal => $count) {
            if ($count > 0) {
                $r = (int) floor($hashVal / $numClasses);
                $c = $hashVal % $numClasses;
                
                $pi = $labelCounts[$r] ?? 0;
                $pj = $predCounts[$c] ?? 0;
                
                if ($pi > 0 && $pj > 0) {
                    $mutualInfo += ($count / $n) * log(($n * $count) / ($pi * $pj));
                }
            }
        }

        $hLabels = 0.0;
        foreach ($labelCounts as $count) {
            if ($count > 0) $hLabels -= ($count / $n) * log($count / $n);
        }

        $hPreds = 0.0;
        foreach ($predCounts as $count) {
            if ($count > 0) $hPreds -= ($count / $n) * log($count / $n);
        }

        if ($hLabels === 0.0 && $hPreds === 0.0) return 1.0;
        if ($hLabels === 0.0 || $hPreds === 0.0) return 0.0;

        $homogeneity = $mutualInfo / $hLabels;
        $completeness = $mutualInfo / $hPreds;

        if ($homogeneity + $completeness === 0.0) return 0.0;

        return (2.0 * $homogeneity * $completeness) / ($homogeneity + $completeness);
    }
}