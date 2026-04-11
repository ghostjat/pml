<?php
declare(strict_types=1);

namespace Pml\Metrics\Clustering;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Homogeneity — each cluster contains only members of a single true class.
 * Range [0, 1]: 1 = each cluster is a pure single-class subset.
 *
 * Homogeneity = 1 - H(K|C) / H(K)
 */
final class Homogeneity implements Metric
{
    public function score(Tensor $predictions, ?Tensor $labels): float
    {
        if ($labels === null) {
            throw new \InvalidArgumentException("Homogeneity requires ground-truth labels.");
        }

        $pred         = $predictions->toFlatArray();
        $true         = $labels->toFlatArray();
        $n            = count($pred);
        $contingency  = [];
        $clusterCount = [];

        foreach ($pred as $i => $k) {
            $c = (int) $true[$i];
            $k = (int) $k;
            $contingency[$k][$c] = ($contingency[$k][$c] ?? 0) + 1;
            $clusterCount[$k]    = ($clusterCount[$k] ?? 0) + 1;
        }

        // H(K)
        $hK = 0.0;
        foreach ($clusterCount as $count) {
            $p   = $count / $n;
            $hK -= $p * log($p);
        }
        if ($hK === 0.0) return 1.0;

        // H(K|C)
        $hKC = 0.0;
        foreach ($contingency as $k => $classes) {
            foreach ($classes as $c => $cnt) {
                $p   = $cnt / $n;
                $pK  = $clusterCount[$k] / $n;
                $hKC -= $p * log($p / $pK);
            }
        }

        return 1.0 - $hKC / $hK;
    }

    public function range(): array { return [0.0, 1.0]; }
}
