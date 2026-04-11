<?php
declare(strict_types=1);

namespace Pml\Metrics\Clustering;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Rand Index — fraction of sample pairs that are either in the same cluster and same class,
 * or in different clusters and different classes.
 * Range [0, 1]: 1 = perfect agreement.
 *
 * NOTE: O(N^2) pair counting — keep N < 10k for reasonable performance.
 */
final class RandIndex implements Metric
{
    public function score(Tensor $predictions, ?Tensor $labels): float
    {
        if ($labels === null) {
            throw new \InvalidArgumentException("RandIndex requires ground-truth labels.");
        }

        $pred = $predictions->toFlatArray();
        $true = $labels->toFlatArray();
        $n    = count($pred);

        $tp_tn = 0;
        $total = 0;

        for ($i = 0; $i < $n - 1; $i++) {
            for ($j = $i + 1; $j < $n; $j++) {
                $sameCluster = ((int) $pred[$i] === (int) $pred[$j]);
                $sameClass   = ((int) $true[$i] === (int) $true[$j]);
                if ($sameCluster === $sameClass) $tp_tn++;
                $total++;
            }
        }

        return $total > 0 ? $tp_tn / $total : 1.0;
    }

    public function range(): array { return [0.0, 1.0]; }
}
