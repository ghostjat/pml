<?php
declare(strict_types=1);

namespace Pml\Metrics\Reports;

use Pml\Interfaces\Learner;
use Pml\Metrics\Metric;
use Pml\Dataset;

/**
 * Aggregate Report — runs multiple metrics at once and returns a keyed array.
 */
final class AggregateReport
{
    /** @var array<string, Metric> */
    private array $metrics;

    public function __construct(array $metrics)
    {
        $this->metrics = $metrics;
    }

    /**
     * @return array<string, float>
     */
    public function generate(Learner $estimator, Dataset $dataset): array
    {
        $predictions = $estimator->predict($dataset);
        $labels      = $dataset->labels();

        $results = [];
        foreach ($this->metrics as $name => $metric) {
            $results[$name] = $metric->score($predictions, $labels);
        }
        return $results;
    }
}
