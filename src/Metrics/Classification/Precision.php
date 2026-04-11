<?php

declare(strict_types=1);

namespace Pml\Metrics\Classification;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Precision Metric.
 * Measures the accuracy of the positive predictions (TP / (TP + FP)).
 */
final class Precision implements Metric
{
    public function score(Tensor $predictions, Tensor $labels): float
    {
        $preds = $predictions->round();
        $y = $labels->round();

        $tp = $y->mul($preds)->sum();
        
        // Total predicted positives: sum(y_pred)
        $predictedPositives = $preds->sum();
        
        return $predictedPositives > 0.0 ? $tp / $predictedPositives : 0.0;
    }
}