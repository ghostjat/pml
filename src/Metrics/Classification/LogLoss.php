<?php

declare(strict_types=1);

namespace Pml\Metrics\Classification;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Log-Loss (Cross-Entropy Loss as an evaluation metric).
 *
 * Equivalent to BinaryCrossEntropy::compute() but exposed as a Metric so it
 * can be used in evaluation pipelines alongside Accuracy, F1, ROC-AUC etc.
 *
 * Lower is better.  Perfect model → 0.0.  Random binary classifier → ln(2) ≈ 0.693.
 *
 * Supports binary and multi-class:
 *   Binary:      $probabilities [N] or [N,1], $labels [N] in {0,1}
 *   Multi-class: $probabilities [N,K] softmax, $labels [N] integer class indices
 *
 * Complexity: fully C-accelerated via Tensor ops — no PHP scalar loops.
 */
final class LogLoss implements Metric
{
    private float $eps;

    public function __construct(float $eps = 1e-7)
    {
        $this->eps = $eps;
    }

    public function score(Tensor $probabilities, Tensor $labels): float
    {
        $shape = $probabilities->shape();

        // ── Binary ────────────────────────────────────────────────────────────
        if (\count($shape) === 1 || \count($shape) === 2 && $shape[1] === 1) {
            $p = $probabilities->squeeze()->clip($this->eps, 1.0 - $this->eps);
            $y = $labels->squeeze();

            // -mean( y·log(p) + (1-y)·log(1-p) )
            $term1    = $y->mul($p->log());
            $term2    = $y->mulScalar(-1.0)->addScalarInplace(1.0)
                          ->mul($p->mulScalar(-1.0)->log1p());
            return -$term1->addInplace($term2)->mean();
        }

        // ── Multi-class ───────────────────────────────────────────────────────
        // $probabilities: [N, K],  $labels: [N] integer class indices
        //
        // One-hot trick (fully C-level, zero PHP loops):
        //   yExpanded [N,1] == categories [1,K] → broadcasts to one-hot [N,K]
        //   select true-class log probs via element-wise multiply + row sum
        $k = $shape[1];

        $p          = $probabilities->clip($this->eps, 1.0 - $this->eps);
        $logP       = $p->log();                                // [N, K]
        $yExpanded  = $labels->squeeze()->expandDims(1);        // [N, 1]
        $categories = Tensor::fromArray([range(0, $k - 1)]);    // [1, K]
        $oneHot     = $yExpanded->equal($categories);           // [N, K] — broadcast

        // nll_i = -sum_k( one_hot[i,k] * log(p[i,k]) )  →  mean over N
        return -$oneHot->mul($logP)->sumAxis(1)->mean();
    }
}
