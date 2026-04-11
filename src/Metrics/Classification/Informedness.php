<?php
declare(strict_types=1);

namespace Pml\Metrics\Classification;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Informedness (Youden's J statistic) = Recall + Specificity - 1.
 * Range [-1, +1]: 1 = perfect, 0 = random.
 */
final class Informedness implements Metric
{
    public function score(Tensor $predictions, ?Tensor $labels): float
    {
        if ($labels === null) {
            throw new \InvalidArgumentException("Informedness requires ground-truth labels.");
        }

        $tp = $predictions->equal($labels)->mul($labels)->sum();
        $tn = $predictions->equal($labels)->mul($labels->logicalNot())->sum();
        $fp = $predictions->greater($labels)->sum();
        $fn = $labels->greater($predictions)->sum();

        $recall      = ($tp + $fn) > 0 ? $tp / ($tp + $fn) : 0.0;
        $specificity = ($tn + $fp) > 0 ? $tn / ($tn + $fp) : 0.0;

        return (float) ($recall + $specificity - 1.0);
    }

    public function range(): array { return [-1.0, 1.0]; }
}
