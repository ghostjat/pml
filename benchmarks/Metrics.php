<?php
declare(strict_types=1);

namespace Pml\Benchmarks;

/**
 * Metrics collector for benchmarking.
 *
 * Memory tracking note:
 *   - getMemoryPeakMB()  — PHP heap peak via memory_get_peak_usage(). Does NOT
 *     include C tensor allocations (the majority of PML's memory usage).
 *   - getVmRssKb()       — Total process RSS via /proc/self/status. Includes C
 *     heap, mapped libraries, and PHP heap. This is the correct metric for
 *     cross-framework memory comparisons.
 *   - getVmRssDeltaKb()  — RSS change since object construction. Measures
 *     actual C+PHP allocation impact of the benchmarked code.
 */
class Metrics
{
    private array $timings    = [];
    private int $memoryPeak   = 0;
    private int $memoryStart  = 0;
    private int $vmRssStart   = 0;   // /proc/self/status VmRSS at construction
    private int $vmRssPeak    = 0;   // peak VmRSS observed across iterations
    private int $iterations   = 0;

    public function __construct()
    {
        $this->memoryStart = memory_get_usage(true);
        $this->vmRssStart  = self::readVmRssKb();
        $this->vmRssPeak   = $this->vmRssStart;
    }

    /** Read VmRSS from /proc/self/status (Linux only). Returns 0 on non-Linux. */
    public static function readVmRssKb(): int
    {
        static $path = '/proc/self/status';
        if (!is_readable($path)) {
            return 0;
        }
        foreach (file($path, FILE_IGNORE_NEW_LINES) as $line) {
            if (str_starts_with($line, 'VmRSS:')) {
                return (int) preg_replace('/\D/', '', $line);
            }
        }
        return 0;
    }

    /** Total RSS in KB at measurement time. Includes C heap. */
    public function getVmRssKb(): int
    {
        return self::readVmRssKb();
    }

    /** RSS delta since construction (KB). Measures C+PHP allocation impact. */
    public function getVmRssDeltaKb(): int
    {
        return max(0, self::readVmRssKb() - $this->vmRssStart);
    }

    /** Peak RSS observed across all recorded iterations (KB). */
    public function getVmRssPeakKb(): int
    {
        return $this->vmRssPeak;
    }

    public function startIteration(): void
    {
        $this->iterations++;
    }

    public function recordTiming(float $timeNs): void
    {
        $this->timings[]  = $timeNs;
        $this->memoryPeak = max($this->memoryPeak, memory_get_peak_usage(true));
        $this->vmRssPeak  = max($this->vmRssPeak, self::readVmRssKb());
    }

    public function getAverageTimeNs(): float
    {
        return empty($this->timings) ? 0.0 : array_sum($this->timings) / count($this->timings);
    }

    public function getAverageTimeMs(): float
    {
        return $this->getAverageTimeNs() / 1_000_000;
    }

    public function getP95TimeNs(): float
    {
        if (empty($this->timings)) return 0.0;
        sort($this->timings);
        $index = (int) (0.95 * (count($this->timings) - 1));
        return $this->timings[$index];
    }

    public function getP95TimeMs(): float
    {
        return $this->getP95TimeNs() / 1_000_000;
    }

    /** Median (p50) timing — preferred over mean for skewed distributions. */
    public function getMedianTimeNs(): float
    {
        if (empty($this->timings)) return 0.0;
        $sorted = $this->timings;
        sort($sorted);
        $n = count($sorted);
        return $n % 2 === 1
            ? $sorted[$n >> 1]
            : ($sorted[$n / 2 - 1] + $sorted[$n / 2]) / 2.0;
    }

    public function getMedianTimeMs(): float
    {
        return $this->getMedianTimeNs() / 1_000_000;
    }

    /** Relative standard deviation (coefficient of variation) as percentage. */
    public function getRstdevPct(): float
    {
        if (count($this->timings) < 2) return 0.0;
        $mean = array_sum($this->timings) / count($this->timings);
        if ($mean == 0.0) return 0.0;
        $variance = array_sum(array_map(fn($t) => ($t - $mean) ** 2, $this->timings)) / count($this->timings);
        return (sqrt($variance) / $mean) * 100.0;
    }

    public function getMinTimeNs(): float
    {
        return empty($this->timings) ? 0.0 : min($this->timings);
    }

    public function getMaxTimeNs(): float
    {
        return empty($this->timings) ? 0.0 : max($this->timings);
    }

    public function getThroughputOpsPerSec(): float
    {
        $medianSec = $this->getMedianTimeNs() / 1_000_000_000;
        return $medianSec > 0 ? 1.0 / $medianSec : 0.0;
    }

    /** PHP heap peak only — does NOT include C tensor allocations. */
    public function getMemoryPeakMB(): float
    {
        return $this->memoryPeak / 1_048_576;
    }

    /** PHP heap delta since construction. */
    public function getMemoryUsedMB(): float
    {
        return (memory_get_usage(true) - $this->memoryStart) / 1_048_576;
    }

    public function getIterations(): int
    {
        return $this->iterations;
    }

    public function reset(): void
    {
        $this->timings    = [];
        $this->memoryPeak = 0;
        $this->iterations = 0;
        $this->memoryStart = memory_get_usage(true);
        $this->vmRssStart  = self::readVmRssKb();
        $this->vmRssPeak   = $this->vmRssStart;
    }

    public function toArray(): array
    {
        return [
            'iterations'         => $this->iterations,
            'median_time_ns'     => $this->getMedianTimeNs(),
            'median_time_ms'     => $this->getMedianTimeMs(),
            'avg_time_ns'        => $this->getAverageTimeNs(),
            'avg_time_ms'        => $this->getAverageTimeMs(),
            'p95_time_ns'        => $this->getP95TimeNs(),
            'p95_time_ms'        => $this->getP95TimeMs(),
            'min_time_ns'        => $this->getMinTimeNs(),
            'max_time_ns'        => $this->getMaxTimeNs(),
            'rstdev_pct'         => $this->getRstdevPct(),
            'throughput_ops_sec' => $this->getThroughputOpsPerSec(),
            // PHP heap metrics (incomplete — excludes C tensor memory)
            'php_heap_peak_mb'   => $this->getMemoryPeakMB(),
            'php_heap_delta_mb'  => $this->getMemoryUsedMB(),
            // RSS metrics (complete — includes C heap, requires Linux /proc)
            'vmrss_delta_kb'     => $this->getVmRssDeltaKb(),
            'vmrss_peak_kb'      => $this->getVmRssPeakKb(),
        ];
    }
}