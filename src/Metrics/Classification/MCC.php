<?php
declare(strict_types=1);

namespace Pml\Metrics\Classification;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Matthews Correlation Coefficient (MCC).
 * Range [-1, +1]: +1 = perfect, 0 = random, -1 = inverse.
 *
 * JIT & Memory Optimized: all counts computed in C-level masks; PHP does only scalar arithmetic.
 */
final class MCC implements Metric
{
    public function score(Tensor $predictions, ?Tensor $labels): float
    {
        if ($labels === null) {
            throw new \InvalidArgumentException("MCC requires ground-truth labels.");
        }

        $tp = $predictions->equal($labels)->mul($labels)->sum();
        $tn = $predictions->equal($labels)->mul($labels->logicalNot())->sum();
        $fp = $predictions->greater($labels)->sum();
        $fn = $labels->greater($predictions)->sum();

        $denom = sqrt(($tp + $fp) * ($tp + $fn) * ($tn + $fp) * ($tn + $fn));
        if ($denom === 0.0) {
            return 0.0;
        }

        return (float) (($tp * $tn - $fp * $fn) / $denom);
    }

    public function range(): array { return [-1.0, 1.0]; }
}
