<?php
declare(strict_types=1);

namespace Pml;

/**
 * Data type descriptor for dataset columns.
 * JIT optimized: constants resolved at compile time, no heap allocation.
 */
final class DataType
{
    public const CONTINUOUS  = 0; // float64 numeric
    public const CATEGORICAL = 1; // string label
    public const IMAGE       = 2; // pixel tensor
    public const OTHER       = 3; // opaque blob

    private function __construct(private readonly int $code) {}

    public static function continuous(): self  { return new self(self::CONTINUOUS); }
    public static function categorical(): self { return new self(self::CATEGORICAL); }
    public static function image(): self       { return new self(self::IMAGE); }
    public static function other(): self       { return new self(self::OTHER); }

    public function isContinuous(): bool  { return $this->code === self::CONTINUOUS; }
    public function isCategorical(): bool { return $this->code === self::CATEGORICAL; }
    public function isImage(): bool       { return $this->code === self::IMAGE; }

    public function code(): int { return $this->code; }

    public function __toString(): string
    {
        return match ($this->code) {
            self::CONTINUOUS  => 'continuous',
            self::CATEGORICAL => 'categorical',
            self::IMAGE       => 'image',
            default           => 'other',
        };
    }
}
