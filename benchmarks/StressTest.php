<?php
declare(strict_types=1);

namespace Pml\Benchmarks;

use Pml\Tensor;

/**
 * Stress testing system for memory, compute, and stability.
 */
class StressTest
{
    private static BenchmarkRunner $runner;
    private static bool $initialized = false;

    public function __construct()
    {
        $this->ensureInit();
    }

    private function ensureInit(): void
    {
        if (self::$initialized) {
            return;
        }
        self::$runner = new BenchmarkRunner();
        self::$initialized = true;
    }

    public function setUp(): void
    {
        $this->ensureInit();
    }

    /**
     * Memory stress test: gradually increase tensor sizes until failure.
     */
    public function memoryStressTest(): array
    {
        $results = [];
        $size = 100;

        while ($size <= 10000) {
            try {
                $metrics = self::$runner->run(function(Metrics $metrics, bool $warmup) use ($size) {
                    if (!$warmup) {
                        $tensor = Tensor::randomNormal([$size, $size]);
                        // Simulate some operation
                        $result = $tensor->add($tensor);
                        unset($result, $tensor);
                    }
                }, "Memory Stress {$size}x{$size}");

                $results[] = [
                    'size' => $size,
                    'success' => true,
                    'memory_peak_mb' => $metrics->getMemoryPeakMB(),
                    'avg_time_ms' => $metrics->getAverageTimeMs()
                ];

                $size *= 2;
            } catch (\Throwable $e) {
                $results[] = [
                    'size' => $size,
                    'success' => false,
                    'error' => $e->getMessage()
                ];
                break;
            }
        }

        return $results;
    }

    /**
     * Compute stress test: large batch sizes and deep networks.
     */
    public function computeStressTest(): array
    {
        $results = [];

        // Large batch test
        $batchSizes = [1000, 5000, 10000];
        foreach ($batchSizes as $batchSize) {
            $metrics = self::$runner->run(function(Metrics $metrics, bool $warmup) use ($batchSize) {
                if (!$warmup) {
                    $input = Tensor::randomNormal([$batchSize, 128]);
                    $weight = Tensor::randomNormal([128, 64]);
                    $bias = Tensor::randomNormal([64]);

                    $output = $input->matmul($weight)->add($bias);
                    unset($output, $input, $weight, $bias);
                }
            }, "Compute Stress Batch {$batchSize}");

            $results[] = [
                'test' => 'large_batch',
                'batch_size' => $batchSize,
                'avg_time_ms' => $metrics->getAverageTimeMs(),
                'throughput' => $metrics->getThroughputOpsPerSec()
            ];
        }

        // Deep network simulation (100+ layers)
        $layers = 100;
        $metrics = self::$runner->run(function(Metrics $metrics, bool $warmup) use ($layers) {
            if (!$warmup) {
                $x = Tensor::randomNormal([32, 64]);
                for ($i = 0; $i < $layers; $i++) {
                    $w = Tensor::randomNormal([64, 64]);
                    $x = $x->matmul($w)->relu();
                    unset($w);
                }
                unset($x);
            }
        }, "Deep Network {$layers} layers");

        $results[] = [
            'test' => 'deep_network',
            'layers' => $layers,
            'avg_time_ms' => $metrics->getAverageTimeMs()
        ];

        return $results;
    }

    /**
     * Long-run stability test.
     */
    public function longRunStabilityTest(int $iterations = 1000): array
    {
        $memoryPeaks = [];
        $times = [];

        for ($i = 0; $i < $iterations; $i++) {
            $metrics = self::$runner->run(function(Metrics $metrics, bool $warmup) {
                if (!$warmup) {
                    $a = Tensor::randomNormal([100, 100]);
                    $b = Tensor::randomNormal([100, 100]);
                    $c = $a->matmul($b);
                    unset($c, $b, $a);
                }
            }, "Stability Iteration {$i}");

            $memoryPeaks[] = $metrics->getMemoryPeakMB();
            $times[] = $metrics->getAverageTimeMs();

            // Check for memory leaks (simple check)
            if ($i > 0 && $memoryPeaks[$i] > $memoryPeaks[0] * 2) {
                // Potential leak
            }
        }

        return [
            'iterations' => $iterations,
            'initial_memory_peak' => $memoryPeaks[0] ?? 0,
            'final_memory_peak' => end($memoryPeaks),
            'avg_time_ms' => array_sum($times) / count($times),
            'memory_drift' => (end($memoryPeaks) - ($memoryPeaks[0] ?? 0))
        ];
    }

    /**
     * Concurrency stress test (multiple processes).
     */
    public function concurrencyStressTest(int $processes = 4): array
    {
        $results = [];

        // Simple fork-based concurrency test
        $pids = [];
        for ($i = 0; $i < $processes; $i++) {
            $pid = pcntl_fork();
            if ($pid == -1) {
                throw new \RuntimeException("Failed to fork");
            } elseif ($pid == 0) {
                // Child process
                $metrics = self::$runner->run(function(Metrics $metrics, bool $warmup) {
                    if (!$warmup) {
                        $tensor = Tensor::randomNormal([500, 500]);
                        $result = $tensor->matmul($tensor);
                        unset($result, $tensor);
                    }
                }, "Concurrency Process {$i}");

                file_put_contents("/tmp/stress_result_{$i}.json", json_encode($metrics->toArray()));
                exit(0);
            } else {
                $pids[] = $pid;
            }
        }

        // Wait for all children
        foreach ($pids as $pid) {
            pcntl_waitpid($pid, $status);
        }

        // Collect results
        for ($i = 0; $i < $processes; $i++) {
            $file = "/tmp/stress_result_{$i}.json";
            if (file_exists($file)) {
                $data = json_decode(file_get_contents($file), true);
                $results[] = $data;
                unlink($file);
            }
        }

        return $results;
    }
}