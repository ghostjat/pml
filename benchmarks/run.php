<?php
declare(strict_types=1);

require_once __DIR__ . '/../vendor/autoload.php';

use Pml\Benchmarks\BenchmarkRunner;
use Pml\Benchmarks\Metrics;
use Pml\Benchmarks\ResultFormatter;
use Pml\Benchmarks\StressTest;
use Pml\Benchmarks\Micro\TensorMicroBench;
use Pml\Benchmarks\Micro\NeuralNetworkMicroBench;
use Pml\Benchmarks\Macro\TrainingMacroBench;

// Parse command line arguments
$options = getopt('', [
    'mode:',      // fast, stress, full
    'iterations:',
    'tensor-size:',
    'affinity:',  // CPU affinity
    'output:',    // json, csv
    'file:'       // output file
]);

$mode = $options['mode'] ?? 'fast';
$iterations = (int)($options['iterations'] ?? 100);
$tensorSize = (int)($options['tensor-size'] ?? 1024);
$outputFormat = $options['output'] ?? 'cli';
$outputFile = $options['file'] ?? null;

// Set CPU affinity if specified
if (isset($options['affinity'])) {
    $cpu = (int)$options['affinity'];
    // Use taskset or similar, but for simplicity, skip
}

// Create benchmark runner
$runner = new BenchmarkRunner();
$runner->setBenchmarkIterations($iterations);

// Run benchmarks based on mode
$allResults = [];

if ($mode === 'fast' || $mode === 'full') {
    echo "Running Micro Benchmarks...\n";

    $tensorBench = new TensorMicroBench($runner);
    $tensorResults = $tensorBench->runAll();
    $allResults['micro_tensor'] = $tensorResults;

    $nnBench = new NeuralNetworkMicroBench($runner);
    $nnResults = $nnBench->runAll();
    $allResults['micro_nn'] = $nnResults;
}

if ($mode === 'full') {
    echo "Running Macro Benchmarks...\n";

    $trainingBench = new TrainingMacroBench($runner);
    $macroResults = $trainingBench->runAll();
    $allResults['macro'] = $macroResults;
}

if ($mode === 'stress' || $mode === 'full') {
    echo "Running Stress Tests...\n";

    $stressTest = new StressTest($runner);

    $memoryResults = $stressTest->memoryStressTest();
    $allResults['stress_memory'] = $memoryResults;

    $computeResults = $stressTest->computeStressTest();
    $allResults['stress_compute'] = $computeResults;

    $stabilityResults = $stressTest->longRunStabilityTest(100); // Shorter for demo
    $allResults['stress_stability'] = $stabilityResults;

    // Concurrency stress (if enabled)
    if (function_exists('pcntl_fork')) {
        $concurrencyResults = $stressTest->concurrencyStressTest(2); // Fewer processes
        $allResults['stress_concurrency'] = $concurrencyResults;
    }
}

// Output results
foreach ($allResults as $category => $results) {
    if (is_array($results)) {
        foreach ($results as $name => $metrics) {
            if ($metrics instanceof Metrics) {
                echo ResultFormatter::formatCli("{$category}_{$name}", $metrics);
            }
        }
    }
}

// Save to file if requested
if ($outputFile) {
    if ($outputFormat === 'json') {
        ResultFormatter::saveResults('json', $allResults, $outputFile);
    } elseif ($outputFormat === 'csv') {
        // Flatten results for CSV
        $flatResults = [];
        foreach ($allResults as $category => $results) {
            if (is_array($results)) {
                foreach ($results as $name => $metrics) {
                    if ($metrics instanceof Metrics) {
                        $flatResults[] = array_merge($metrics->toArray(), [
                            'category' => $category,
                            'benchmark' => $name
                        ]);
                    } else {
                        $flatResults[] = array_merge($metrics, [
                            'category' => $category,
                            'benchmark' => $name
                        ]);
                    }
                }
            }
        }
        ResultFormatter::saveResults('csv', $flatResults, $outputFile);
    }
}

echo "\nBenchmarking complete.\n";