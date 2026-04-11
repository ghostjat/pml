<?php
declare(strict_types=1);

namespace Pml\CrossValidation;

use Pml\Interfaces\Learner;
use Pml\Metrics\Metric;
use Pml\Dataset;

/**
 * K-Fold cross-validation.
 * Splits the dataset into K equal folds; each fold serves as the test set once.
 * Returns the mean score across all K folds.
 *
 * JIT & Memory Optimized:
 * - Dataset fold() generator yields zero-copy slice views.
 * - Each fold clone of the estimator is discarded immediately after scoring.
 */
final class KFold
{
    public function __construct(private readonly int $k = 5) {}

    public function test(Learner $estimator, Dataset $dataset, Metric $metric): float
    {
        $scores = [];

        foreach ($dataset->fold($this->k) as [$train, $val]) {
            $clone = clone $estimator;
            $clone->train($train);
            $predictions = $clone->predict($val);
            $scores[]    = $metric->score($predictions, $val->labels());
        }

        return array_sum($scores) / count($scores);
    }
}
