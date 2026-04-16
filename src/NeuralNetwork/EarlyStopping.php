<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork;

/**
 * Zero-overhead Early Stopping monitor.
 *
 * All state fits in scalar zvals — zero heap allocations per update().
 * Integer return constants let the JIT use a fast switch/compare with no
 * string interning or object construction on the hot path.
 *
 * Designed to be used with a `final` class so the JIT can devirtualise
 * the update() call and inline the body directly into Sequential::train().
 */
final class EarlyStopping
{
    /** Metric improved by at least minDelta — caller should snapshot weights. */
    public const int IMPROVED = 1;
    /** No improvement yet, patience not exhausted — continue training. */
    public const int CONTINUE = 0;
    /** Patience exhausted — caller should restore best weights and halt. */
    public const int STOP     = 2;

    private float $bestMetric;
    private int   $counter = 0;

    /**
     * @param int    $patience  Epochs to wait after last improvement before stopping.
     * @param string $mode      'min' (loss) or 'max' (accuracy/score).
     * @param float  $minDelta  Minimum absolute change that counts as improvement.
     */
    public function __construct(
        private readonly int    $patience,
        private readonly string $mode     = 'min',
        private readonly float  $minDelta = 1e-4
    ) {
        $this->bestMetric = $mode === 'min' ? \INF : -\INF;
    }

    /**
     * Feed the epoch metric and receive one of IMPROVED / CONTINUE / STOP.
     *
     * Hot-path properties:
     *   - Single branch on `$improved` (predictable: false after initial epochs).
     *   - No closures, no array allocs, no string ops.
     *   - Two scalar comparisons, two scalar writes in the common (no-improvement) case.
     */
    public function update(float $metric): int
    {
        $improved = $this->mode === 'min'
            ? $metric < $this->bestMetric - $this->minDelta
            : $metric > $this->bestMetric + $this->minDelta;

        if ($improved) {
            $this->bestMetric = $metric;
            $this->counter    = 0;
            return self::IMPROVED;
        }

        if (++$this->counter >= $this->patience) {
            return self::STOP;
        }

        return self::CONTINUE;
    }

    /** Best metric value seen so far. */
    public function getBestMetric(): float { return $this->bestMetric; }

    /** Epochs elapsed since last improvement. */
    public function getCounter(): int { return $this->counter; }

    /** Re-arm for a new training run without reallocating the object. */
    public function reset(): void
    {
        $this->counter    = 0;
        $this->bestMetric = $this->mode === 'min' ? \INF : -\INF;
    }
}
