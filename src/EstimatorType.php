<?php
declare(strict_types=1);

namespace Pml;

/**
 * Estimator type descriptor.
 */
final class EstimatorType
{
    public const CLASSIFIER       = 0;
    public const REGRESSOR        = 1;
    public const CLUSTERER        = 2;
    public const ANOMALY_DETECTOR = 3;
    public const EMBEDDER         = 4;

    private function __construct(private readonly int $code) {}

    public static function classifier(): self      { return new self(self::CLASSIFIER); }
    public static function regressor(): self       { return new self(self::REGRESSOR); }
    public static function clusterer(): self       { return new self(self::CLUSTERER); }
    public static function anomalyDetector(): self { return new self(self::ANOMALY_DETECTOR); }
    public static function embedder(): self        { return new self(self::EMBEDDER); }

    public function isClassifier(): bool      { return $this->code === self::CLASSIFIER; }
    public function isRegressor(): bool       { return $this->code === self::REGRESSOR; }
    public function isClusterer(): bool       { return $this->code === self::CLUSTERER; }
    public function isAnomalyDetector(): bool { return $this->code === self::ANOMALY_DETECTOR; }

    public function code(): int { return $this->code; }

    public function __toString(): string
    {
        return match ($this->code) {
            self::CLASSIFIER       => 'classifier',
            self::REGRESSOR        => 'regressor',
            self::CLUSTERER        => 'clusterer',
            self::ANOMALY_DETECTOR => 'anomaly detector',
            self::EMBEDDER         => 'embedder',
            default                => 'unknown',
        };
    }
}
