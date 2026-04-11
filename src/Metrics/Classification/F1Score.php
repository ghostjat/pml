<?php

declare(strict_types=1);

namespace Pml\Metrics\Classification;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * F1 Score Metric.
 * The harmonic mean of precision and recall. Best for imbalanced datasets.
 * * JIT & Memory Optimized:
 * - Computes True Positives, False Positives, and False Negatives using 100% C-Level Vector masks.
 */
final class F1Score implements Metric
{
    public function score(Tensor $predictions, Tensor $labels): float
    {
        // Snap any probabilities to hard 1.0 or 0.0 classes
        $preds = $predictions->round();
        $y = $labels->round();

        // True Positives: sum(y_true * y_pred)
        $tp = $y->mul($preds)->sum();
        
        // False Positives: sum((1 - y_true) * y_pred)
        $fpMask = $y->mulScalar(-1.0)->addScalarInplace(1.0);
        $fp = $fpMask->mulInplace($preds)->sum();
        
        // False Negatives: sum(y_true * (1 - y_pred))
        $fnMask = $preds->mulScalar(-1.0)->addScalarInplace(1.0);
        $fn = $y->mul($fnMask)->sum();
        
        $denominator = $tp + 0.5 * ($fp + $fn);
        
        return $denominator > 0.0 ? $tp / $denominator : 0.0;
    }
}