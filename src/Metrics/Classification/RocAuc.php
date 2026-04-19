<?php

declare(strict_types=1);

namespace Pml\Metrics\Classification;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * ROC-AUC (Area Under the Receiver Operating Characteristic Curve).
 *
 * Measures the probability that a randomly chosen positive sample is ranked
 * higher than a randomly chosen negative sample (Wilcoxon-Mann-Whitney statistic).
 *
 * Accepts:
 *   $probabilities — [N] or [N,1] or [N,2] float32 positive-class scores
 *   $labels        — [N] binary ground-truth (0.0 / 1.0)
 *
 * Complexity: O(N log N) — dominated by argsort.
 * Allocations: one PHP float[] for probabilities/labels via toFlatArray().
 */
final class RocAuc implements Metric
{
    public function score(Tensor $probabilities, Tensor $labels): float
    {
        // Normalise to 1-D positive-class probabilities
        $probs = $this->toPositiveProbs($probabilities);
        $y     = $labels->squeeze()->toFlatArray();

        $n = count($probs);
        if ($n === 0) {
            return 0.0;
        }

        // Sort by descending probability score
        $order = range(0, $n - 1);
        usort($order, static fn(int $a, int $b) => $probs[$b] <=> $probs[$a]);

        // Walk the sorted list building the ROC curve via the trapezoidal rule.
        // Each step is either a TPR increase (positive sample) or FPR increase (negative).
        $nPos = array_sum($y);
        $nNeg = $n - $nPos;

        if ($nPos === 0 || $nNeg === 0) {
            return 0.5;   // Degenerate: only one class present
        }

        $auc  = 0.0;
        $tpr  = 0.0;
        $fpr  = 0.0;
        $prevFpr = 0.0;
        $prevTpr = 0.0;

        foreach ($order as $i) {
            if ($y[$i] == 1) {
                $tpr += 1.0 / $nPos;
            } else {
                $fpr += 1.0 / $nNeg;
                // Trapezoid between previous FPR and current FPR
                $auc += ($fpr - $prevFpr) * ($tpr + $prevTpr) * 0.5;
                $prevFpr = $fpr;
                $prevTpr = $tpr;
            }
        }
        // Close to (1,1)
        $auc += (1.0 - $prevFpr) * ($tpr + $prevTpr) * 0.5;

        return (float) $auc;
    }

    private function toPositiveProbs(Tensor $probabilities): array
    {
        $shape = $probabilities->shape();

        if (count($shape) === 1 || (count($shape) === 2 && $shape[1] === 1)) {
            // [N] or [N,1] — already positive-class probs
            return $probabilities->squeeze()->toFlatArray();
        }

        if (count($shape) === 2 && $shape[1] === 2) {
            // [N,2] softmax output — take column 1 (positive class)
            return $probabilities->col(1)->toFlatArray();
        }

        // Fallback: flatten and treat as positive-class scores
        return $probabilities->flatten()->toFlatArray();
    }
}
