<?php
declare(strict_types=1);

namespace Pml\Data;

use Pml\Lib\TensorEngine;

/**
 * Intermediate object returned by DataFrame::groupBy().
 *
 * Bridges the fluent API to the C df_groupby_agg / df_groupby_multi_agg
 * functions.  All computation runs in C (OpenMP parallel per-group
 * accumulation).  No PHP loops over rows.
 *
 * Usage:
 *   $df->groupBy('category')->sum(['price', 'volume']);
 *   $df->groupBy('category')->mean(['price']);
 *   $df->groupBy('category')->agg(['price' => 'mean', 'volume' => 'sum']);
 */
final class GroupBy
{
    // ── DFAggType enum values (mirrors C DFAggType) ──────────────────────
    private const AGG_SUM   = 0;
    private const AGG_MEAN  = 1;
    private const AGG_MIN   = 2;
    private const AGG_MAX   = 3;
    private const AGG_COUNT = 4;
    private const AGG_STD   = 5;

    private static ?\FFI $ffi = null;

    public function __construct(
        private readonly DataFrame $df,
        private readonly int       $groupColIdx
    ) {}

    // ── Single-agg-type convenience methods ──────────────────────────────

    /** @param list<string> $cols */
    public function sum(array $cols): DataFrame
    {
        return $this->_agg($cols, self::AGG_SUM);
    }

    /** @param list<string> $cols */
    public function mean(array $cols): DataFrame
    {
        return $this->_agg($cols, self::AGG_MEAN);
    }

    /** @param list<string> $cols */
    public function min(array $cols): DataFrame
    {
        return $this->_agg($cols, self::AGG_MIN);
    }

    /** @param list<string> $cols */
    public function max(array $cols): DataFrame
    {
        return $this->_agg($cols, self::AGG_MAX);
    }

    /** Returns group sizes (count of non-null values) per group. */
    public function count(array $cols): DataFrame
    {
        return $this->_agg($cols, self::AGG_COUNT);
    }

    /** @param list<string> $cols */
    public function std(array $cols): DataFrame
    {
        return $this->_agg($cols, self::AGG_STD);
    }

    /**
     * Per-column aggregation.
     *
     * @param array<string,string> $colOps  e.g. ['price' => 'mean', 'volume' => 'sum']
     */
    public function agg(array $colOps): DataFrame
    {
        $ffi      = self::ffi();
        $colNames = array_keys($colOps);
        $aggNames = array_values($colOps);
        $n        = count($colNames);

        if ($n === 0) {
            throw new \InvalidArgumentException('[GroupBy] agg: no columns specified');
        }

        $cIdxs  = $ffi->new("int[{$n}]");
        $cTypes = $ffi->new("int[{$n}]");
        foreach ($colNames as $k => $col) {
            $cIdxs[$k]  = $this->df->colIdx($col);
            $cTypes[$k] = self::aggType($aggNames[$k]);
        }

        $ptr = $ffi->df_groupby_multi_agg(
            $this->df->ptr(),
            $this->groupColIdx,
            $ffi->cast('int*', $cIdxs),
            $ffi->cast('int*', $cTypes),
            $n
        );
        self::checkError();
        return self::wrap($ptr);
    }

    // ── Private helpers ───────────────────────────────────────────────────

    private function _agg(array $cols, int $aggType): DataFrame
    {
        $ffi = self::ffi();
        $n   = count($cols);

        if ($n === 0) {
            throw new \InvalidArgumentException('[GroupBy] no columns specified');
        }

        $cIdxs = $ffi->new("int[{$n}]");
        foreach ($cols as $k => $col) {
            $cIdxs[$k] = $this->df->colIdx($col);
        }

        $ptr = $ffi->df_groupby_agg(
            $this->df->ptr(),
            $this->groupColIdx,
            $ffi->cast('int*', $cIdxs),
            $n,
            $aggType
        );
        self::checkError();
        return self::wrap($ptr);
    }

    private static function aggType(string $name): int
    {
        return match (strtolower($name)) {
            'sum'   => self::AGG_SUM,
            'mean'  => self::AGG_MEAN,
            'min'   => self::AGG_MIN,
            'max'   => self::AGG_MAX,
            'count' => self::AGG_COUNT,
            'std'   => self::AGG_STD,
            default => throw new \InvalidArgumentException("[GroupBy] unknown agg type: {$name}"),
        };
    }

    private static function wrap(\FFI\CData $ptr): DataFrame
    {
        return DataFrame::_fromPtr($ptr);
    }

    private static function ffi(): \FFI
    {
        return self::$ffi ??= TensorEngine::get();
    }

    private static function checkError(): void
    {
        $ffi = self::ffi();
        if ($ffi->tensor_check_error()) {
            $msg = \FFI::string($ffi->tensor_get_last_error());
            $ffi->tensor_clear_error();
            throw new \RuntimeException('[GroupBy] ' . $msg);
        }
    }
}
