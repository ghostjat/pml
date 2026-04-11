<?php
declare(strict_types=1);

namespace Pml\Metrics\Classification;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * F-Beta Score — generalised harmonic mean of Precision and Recall.
 * beta=1 → F1, beta=2 → recall-heavy, beta=0.5 → precision-heavy.
 *
 * JIT & Memory Optimized:
 * - TP/FP/FN computed via C-level boolean index counts.
 * - Single scalar arithmetic in PHP userland.
 */
final class FBeta implements Metric
{
    public function __construct(private readonly float $beta = 1.0) {}

    public function score(Tensor $predictions, ?Tensor $labels): float
    {
        if ($labels === null) {
            throw new \InvalidArgumentException("FBeta requires ground-truth labels.");
        }

        $beta2 = $this->beta ** 2;

        // Integer comparison mask — stays in C
        $tp = $predictions->equal($labels)->mul($labels)->sum();
        $fp = $predictions->greater($labels)->sum();
        $fn = $labels->greater($predictions)->sum();

        $denom = ($beta2 + 1.0) * $tp + $beta2 * $fn + $fp;
        if ($denom === 0.0) {
            return 0.0;
        }

        return (float) (($beta2 + 1.0) * $tp / $denom);
    }

    public function range(): array { return [0.0, 1.0]; }
}
