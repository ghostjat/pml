<?php
declare(strict_types=1);

namespace Pml\CrossValidation;

use Pml\Interfaces\Learner;
use Pml\Metrics\Metric;
use Pml\Dataset;

/**
 * Monte Carlo (Repeated Random Sub-sampling) cross-validation.
 * Randomly splits the dataset $simulations times and averages the score.
 */
final class MonteCarlo
{
    public function __construct(
        private readonly int   $simulations = 10,
        private readonly float $testRatio   = 0.2
    ) {}

    public function test(Learner $estimator, Dataset $dataset, Metric $metric): float
    {
        $scores = [];

        for ($i = 0; $i < $this->simulations; $i++) {
            $shuffled = (clone $dataset)->randomize();
            [$train, $test] = $shuffled->split(1.0 - $this->testRatio);

            $clone = clone $estimator;
            $clone->train($train);
            $scores[] = $metric->score($clone->predict($test), $test->labels());
        }

        return array_sum($scores) / count($scores);
    }
}
