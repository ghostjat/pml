<?php
declare(strict_types=1);

namespace Pml\Benchmarks;

use Pml\Benchmarks\Metrics;

/**
 * Benchmark runner with warmup, isolation, and high-resolution timing.
 */
class BenchmarkRunner
{
    private int $warmupIterations = 5;
    private int $benchmarkIterations = 100;
    private bool $isolateRuns = true;

    public function setWarmupIterations(int $iterations): self
    {
        $this->warmupIterations = $iterations;
        return $this;
    }

    public function setBenchmarkIterations(int $iterations): self
    {
        $this->benchmarkIterations = $iterations;
        return $this;
    }

    public function setIsolateRuns(bool $isolate): self
    {
        $this->isolateRuns = $isolate;
        return $this;
    }

    /**
     * Run a benchmark function with warmup and measurement.
     *
     * @param callable $benchmarkFn Function to benchmark, receives Metrics instance
     * @param string $name Benchmark name
     * @return Metrics
     */
    public function run(callable $benchmarkFn, string $name): Metrics
    {
        $metrics = new Metrics();

        // Warmup phase
        for ($i = 0; $i < $this->warmupIterations; $i++) {
            $benchmarkFn($metrics, true); // true for warmup
        }

        // Reset metrics after warmup
        $metrics->reset();

        // Benchmark phase
        for ($i = 0; $i < $this->benchmarkIterations; $i++) {
            $metrics->startIteration();

            if ($this->isolateRuns) {
                // Isolate runs by forking or just running in sequence
                $this->runIsolated($benchmarkFn, $metrics);
            } else {
                $start = hrtime(true);
                $benchmarkFn($metrics, false);
                $end = hrtime(true);
                $metrics->recordTiming($end - $start);
            }
        }

        return $metrics;
    }

    private function runIsolated(callable $benchmarkFn, Metrics $metrics): void
    {
        $start = hrtime(true);
        $benchmarkFn($metrics, false);
        $end = hrtime(true);
        $metrics->recordTiming($end - $start);
    }
}