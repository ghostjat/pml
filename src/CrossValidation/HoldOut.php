<?php
declare(strict_types=1);

namespace Pml\CrossValidation;

use Pml\Interfaces\Learner;
use Pml\Metrics\Metric;
use Pml\Dataset;

/**
 * Hold-Out validation: split once into train/test, train, then score.
 */
final class HoldOut
{
    public function __construct(
        private readonly float $testRatio = 0.2,
        private readonly bool  $stratify  = false
    ) {}

    public function test(Learner $estimator, Dataset $dataset, Metric $metric): float
    {
        [$train, $test] = $dataset->split(1.0 - $this->testRatio);

        $estimator->train($train);
        $predictions = $estimator->predict($test);

        return $metric->score($predictions, $test->labels());
    }
}
