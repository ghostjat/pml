<?php
declare(strict_types=1);

namespace Pml\Metrics\Clustering;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Completeness — all members of a true class are assigned to the same cluster.
 * Range [0, 1]: 1 = all true-class members in one cluster.
 *
 * Uses conditional entropy H(C|K) where C = true classes, K = cluster assignments.
 * Completeness = 1 - H(C|K) / H(C)
 */
final class Completeness implements Metric
{
    public function score(Tensor $predictions, ?Tensor $labels): float
    {
        if ($labels === null) {
            throw new \InvalidArgumentException("Completeness requires ground-truth labels.");
        }

        $pred  = $predictions->toFlatArray();
        $true  = $labels->toFlatArray();
        $n     = count($true);

        // Build contingency table
        $contingency = [];
        $classCount  = [];
        $clusterCount = [];
        foreach ($true as $i => $c) {
            $k = (int) $pred[$i];
            $c = (int) $c;
            $contingency[$c][$k] = ($contingency[$c][$k] ?? 0) + 1;
            $classCount[$c]      = ($classCount[$c] ?? 0) + 1;
        }

        // H(C) — entropy of true labels
        $hC = 0.0;
        foreach ($classCount as $count) {
            $p   = $count / $n;
            $hC -= $p * log($p);
        }
        if ($hC === 0.0) return 1.0;

        // H(C|K) — conditional entropy
        $hCK = 0.0;
        foreach ($contingency as $c => $clusters) {
            foreach ($clusters as $k => $cnt) {
                $p    = $cnt / $n;
                $pK   = array_sum($clusters) / $n;
                $hCK -= $p * log($p / $pK);
            }
        }

        return 1.0 - $hCK / $hC;
    }

    public function range(): array { return [0.0, 1.0]; }
}
