<?php

declare(strict_types=1);

namespace Pml\Metrics\Classification;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Recall (Sensitivity) Metric.
 * Measures the model's ability to find all positive instances (TP / (TP + FN)).
 */
final class Recall implements Metric
{
    public function score(Tensor $predictions, Tensor $labels): float
    {
        $preds = $predictions->round();
        $y = $labels->round();

        $tp = $y->mul($preds)->sum();
        
        // Total actual positives: sum(y_true)
        $actualPositives = $y->sum();
        
        return $actualPositives > 0.0 ? $tp / $actualPositives : 0.0;
    }
}