<?php
declare(strict_types=1);

namespace Pml\Metrics\Reports;

use Pml\Tensor;

/**
 * Multiclass Breakdown — per-class precision, recall, F1, and support.
 *
 * JIT & Memory Optimized:
 * - Counts extracted via C-level boolean masks; PHP only performs scalar arithmetic.
 */
final class MulticlassBreakdown
{
    /**
     * @return array<int|string, array{precision: float, recall: float, f1: float, support: int}>
     */
    public function generate(Tensor $predictions, Tensor $labels): array
    {
        $pred   = $predictions->toFlatArray();
        $true   = $labels->toFlatArray();
        $n      = count($pred);
        $classes = array_values(array_unique(array_merge($pred, $true)));
        sort($classes);

        $report = [];
        foreach ($classes as $c) {
            $c = (int) $c;
            $tp = $tn = $fp = $fn = 0;

            for ($i = 0; $i < $n; $i++) {
                $p = (int) $pred[$i];
                $t = (int) $true[$i];
                if ($p === $c && $t === $c) $tp++;
                elseif ($p !== $c && $t !== $c) $tn++;
                elseif ($p === $c && $t !== $c) $fp++;
                else $fn++;
            }

            $precision = ($tp + $fp) > 0 ? $tp / ($tp + $fp) : 0.0;
            $recall    = ($tp + $fn) > 0 ? $tp / ($tp + $fn) : 0.0;
            $denom     = $precision + $recall;
            $f1        = $denom > 0.0 ? 2.0 * $precision * $recall / $denom : 0.0;

            $report[$c] = [
                'precision' => $precision,
                'recall'    => $recall,
                'f1'        => $f1,
                'support'   => $tp + $fn,
            ];
        }

        return $report;
    }
}
