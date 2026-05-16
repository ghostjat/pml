<?php
declare(strict_types=1);

namespace Pml\Benchmarks;

/**
 * Formats benchmark results for CLI, JSON, and CSV output.
 */
class ResultFormatter
{
    public static function formatCli(string $benchmarkName, Metrics $metrics): string
    {
        $output = "\nBenchmark: {$benchmarkName}\n";
        $output .= "Iterations: " . $metrics->getIterations() . "\n";
        $output .= str_repeat("-", 40) . "\n";
        $output .= sprintf("Time (avg):     %.2f ms\n", $metrics->getAverageTimeMs());
        $output .= sprintf("Time (p95):     %.2f ms\n", $metrics->getP95TimeMs());
        $output .= sprintf("Time (min):     %.2f ns\n", $metrics->getMinTimeNs());
        $output .= sprintf("Time (max):     %.2f ns\n", $metrics->getMaxTimeNs());
        $output .= sprintf("Throughput:     %.0f ops/sec\n", $metrics->getThroughputOpsPerSec());
        $output .= sprintf("Memory (peak):  %.2f MB\n", $metrics->getMemoryPeakMB());
        $output .= sprintf("Memory (used):  %.2f MB\n", $metrics->getMemoryUsedMB());

        return $output;
    }

    public static function formatJson(array $results): string
    {
        return json_encode($results, JSON_PRETTY_PRINT);
    }

    public static function formatCsv(array $results): string
    {
        if (empty($results)) return '';

        $header = array_keys($results[0]);
        $csv = implode(',', $header) . "\n";

        foreach ($results as $result) {
            $csv .= implode(',', array_map(fn($v) => is_numeric($v) ? $v : '"' . $v . '"', $result)) . "\n";
        }

        return $csv;
    }

    public static function saveResults(string $format, array $results, string $filename): void
    {
        $content = match ($format) {
            'json' => self::formatJson($results),
            'csv' => self::formatCsv($results),
            default => throw new \InvalidArgumentException("Unsupported format: {$format}")
        };

        file_put_contents($filename, $content);
    }
}