<?php

declare(strict_types=1);

namespace Pml\Metrics\Classification;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * PR-AUC (Area Under the Precision-Recall Curve).
 *
 * More informative than ROC-AUC on imbalanced datasets because it focuses
 * entirely on the positive class.  Integrates using the interpolated
 * trapezoid rule (same as sklearn's average_precision_score).
 *
 * Accepts the same inputs as RocAuc:
 *   $probabilities — [N], [N,1], or [N,2] positive-class scores
 *   $labels        — [N] binary ground-truth (0.0 / 1.0)
 *
 * Complexity: O(N log N).
 */
final class PrAuc implements Metric
{
    public function score(Tensor $probabilities, Tensor $labels): float
    {
        $probs = $this->toPositiveProbs($probabilities);
        $y     = $labels->squeeze()->toFlatArray();

        $n = count($probs);
        if ($n === 0) {
            return 0.0;
        }

        // Sort by descending score
        $order = range(0, $n - 1);
        usort($order, static fn(int $a, int $b) => $probs[$b] <=> $probs[$a]);

        $nPos = (float) array_sum($y);
        if ($nPos === 0.0) {
            return 0.0;
        }

        $tp      = 0.0;
        $fp      = 0.0;
        $auc     = 0.0;
        $prevRec = 0.0;
        $prevPre = 1.0;   // Convention: precision at recall=0 is 1.0

        foreach ($order as $i) {
            if ($y[$i] == 1) {
                $tp++;
            } else {
                $fp++;
            }

            $recall    = $tp / $nPos;
            $precision = $tp / ($tp + $fp);

            // Trapezoidal area between previous and current (recall, precision) point
            $auc     += ($recall - $prevRec) * ($precision + $prevPre) * 0.5;
            $prevRec  = $recall;
            $prevPre  = $precision;
        }

        return (float) $auc;
    }

    private function toPositiveProbs(Tensor $probabilities): array
    {
        $shape = $probabilities->shape();

        if (count($shape) === 1 || (count($shape) === 2 && $shape[1] === 1)) {
            return $probabilities->squeeze()->toFlatArray();
        }

        if (count($shape) === 2 && $shape[1] === 2) {
            return $probabilities->col(1)->toFlatArray();
        }

        return $probabilities->flatten()->toFlatArray();
    }
}
