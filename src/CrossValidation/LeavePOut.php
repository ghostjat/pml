<?php
declare(strict_types=1);

namespace Pml\CrossValidation;

use Pml\Interfaces\Learner;
use Pml\Metrics\Metric;
use Pml\Dataset;

/**
 * Leave-P-Out cross-validation.
 * Exhaustively evaluates all C(n, p) combinations of P held-out samples.
 *
 * WARNING: Exponential in N for large P. Suitable only for small datasets.
 *
 * JIT & Memory Optimized:
 * - All splits are zero-copy Tensor views via slice + take.
 */
final class LeavePOut
{
    public function __construct(private readonly int $p = 1) {}

    public function test(Learner $estimator, Dataset $dataset, Metric $metric): float
    {
        $n      = $dataset->numRows();
        $scores = [];

        foreach ($this->combinations(range(0, $n - 1), $this->p) as $testIdx) {
            $trainIdx = array_values(array_diff(range(0, $n - 1), $testIdx));

            $tT    = \Pml\Tensor::fromArray($testIdx,  \Pml\Tensor::DTYPE_INT32);
            $trT   = \Pml\Tensor::fromArray($trainIdx, \Pml\Tensor::DTYPE_INT32);

            $testDs  = new Dataset(
                $dataset->samples()->take($tT, 0),
                $dataset->labels()->take($tT, 0)
            );
            $trainDs = new Dataset(
                $dataset->samples()->take($trT, 0),
                $dataset->labels()->take($trT, 0)
            );

            $clone = clone $estimator;
            $clone->train($trainDs);
            $scores[] = $metric->score($clone->predict($testDs), $testDs->labels());
        }

        return empty($scores) ? 0.0 : array_sum($scores) / count($scores);
    }

    /**
     * Generates all C(n, p) combinations — pure PHP, small N only.
     * @return \Generator<int[]>
     */
    private function combinations(array $set, int $p): \Generator
    {
        $n = count($set);
        if ($p > $n) return;

        $indices = range(0, $p - 1);

        while (true) {
            yield array_map(fn($i) => $set[$i], $indices);

            $i = $p - 1;
            while ($i >= 0 && $indices[$i] === $n - $p + $i) {
                $i--;
            }
            if ($i < 0) break;

            $indices[$i]++;
            for ($j = $i + 1; $j < $p; $j++) {
                $indices[$j] = $indices[$j - 1] + 1;
            }
        }
    }
}
