<?php
declare(strict_types=1);

namespace Pml\Data;

use Pml\Lib\TensorEngine;
use Pml\Tensor;

/**
 * Pandas-like DataFrame backed by the C columnar engine.
 *
 * MEMORY MODEL
 *   The DataFrame* C pointer is exclusively owned by this PHP object.
 *   __destruct() calls df_free().  Operations that create new DataFrames
 *   return new PHP objects — the original is never mutated.
 *
 * DESIGN CONSTRAINTS (HPC)
 *   - All heavy work runs in C (OpenMP + SIMD).
 *   - PHP is orchestration only — no per-row PHP loops for data.
 *   - Zero-copy: column accessors return Tensor views.
 *   - String comparison ops use C category hash maps (O(1)).
 *
 * DTYPES
 *   0 = FLOAT32 (NaN = missing)
 *   1 = INT32   (INT32_MIN = missing)
 *   2 = STRING  (stored as category index; -1 = missing)
 *
 * COMPARISON OPERATORS (for where())
 *   '=='  '!='  '>'  '>='  '<'  '<='
 *
 * AGG TYPES (for groupBy())
 *   'sum'  'mean'  'min'  'max'  'count'  'std'
 *
 * JOIN TYPES
 *   'inner'  'left'
 *
 * Usage:
 *   $df = DataFrame::fromCSV('data.csv');
 *   $agg = $df->where('price', '>', 100.0)
 *              ->sortBy('price')
 *              ->groupBy('category')
 *              ->mean(['price', 'volume']);
 */
final class DataFrame
{
    // ── DFCmpOp enum values (mirrors C DFCmpOp) ─────────────────────────
    private const CMP_EQ  = 0;
    private const CMP_NEQ = 1;
    private const CMP_GT  = 2;
    private const CMP_GTE = 3;
    private const CMP_LT  = 4;
    private const CMP_LTE = 5;

    // ── DFJoinType enum values ───────────────────────────────────────────
    private const JOIN_INNER = 0;
    private const JOIN_LEFT  = 1;

    private static ?\FFI $ffi = null;

    /** @var \FFI\CData  DataFrame* */
    private \FFI\CData $ptr;

    /** @var array<string,int>  name → column index cache */
    private array $colIndex = [];

    private function __construct(\FFI\CData $ptr)
    {
        $this->ptr = $ptr;
        $this->_buildIndex();
    }

    /** @internal Package-level named constructor used by GroupBy. */
    public static function _fromPtr(\FFI\CData $ptr): self
    {
        return new self($ptr);
    }

    public function __destruct()
    {
        self::ffi()->df_free($this->ptr);
    }

    // ── FFI singleton ────────────────────────────────────────────────────

    private static function ffi(): \FFI
    {
        return self::$ffi ??= TensorEngine::get();
    }

    // ── Factories ────────────────────────────────────────────────────────

    /** Parse a CSV file into a columnar DataFrame. */
    public static function fromCSV(string $path, bool $hasHeader = true): self
    {
        $ptr = self::ffi()->df_read_csv($path, $hasHeader);
        self::checkError();
        if (\FFI::isNull($ptr)) {
            throw new \RuntimeException("[DataFrame] df_read_csv returned NULL for: {$path}");
        }
        return new self($ptr);
    }

    /**
     * Create a DataFrame from a Pml\Tensor (row-major, all columns FLOAT32).
     *
     * @param Tensor       $tensor   [n_rows × n_cols] FLOAT32 tensor
     * @param list<string> $colNames column names (length must match tensor columns)
     */
    public static function fromTensor(Tensor $tensor, array $colNames): self
    {
        $shape = $tensor->shape();
        if (count($shape) !== 2) {
            throw new \InvalidArgumentException('[DataFrame] fromTensor requires a 2D tensor');
        }
        [$nRows, $nCols] = $shape;
        if (count($colNames) !== $nCols) {
            throw new \InvalidArgumentException('[DataFrame] colNames count must match tensor columns');
        }
        return self::_fromFlatColumns($tensor->toFlatArray(), $nRows, $nCols, $colNames);
    }

    /**
     * @internal Build a DataFrame column-by-column from a flat float array.
     *           Uses df_create(n_rows, 0) as the empty seed, then chains
     *           df_add_f32_column — zero temp-file allocation.
     */
    private static function _fromFlatColumns(
        array  $flat, int $nRows, int $nCols, array $colNames
    ): self {
        $ffi = self::ffi();

        /* Bootstrap: create a 1-column DataFrame from the first column,
         * then append remaining columns one by one via df_add_f32_column.
         * df_create(n,0) is rejected by C, so we seed with df_create(n,1). */
        $seed = $ffi->df_create($nRows, 1);
        if (\FFI::isNull($seed)) {
            throw new \RuntimeException('[DataFrame] df_create failed');
        }
        /* Initialise the first column slot */
        $ffi->df_rename_column($seed, 0, $colNames[0]);

        /* Allocate its data buffer by using df_add_f32_column on a NULL base
         * is not valid — instead, replace via drop+add on the seed once freed.
         * Actually the simplest path: use a single-column tmp CSV for the seed,
         * then add remaining columns via df_add_f32_column. */
        $ffi->df_free($seed);

        /* Write first-column CSV → read → add remaining columns */
        $tmp = tempnam(sys_get_temp_dir(), 'pml_') . '.csv';
        $fh  = fopen($tmp, 'w');
        if (!$fh) throw new \RuntimeException('[DataFrame] cannot create temp file');
        fputcsv($fh, [$colNames[0]]);
        for ($r = 0; $r < $nRows; $r++) {
            fputcsv($fh, [$flat[$r * $nCols]]);
        }
        fclose($fh);
        $cur = $ffi->df_read_csv($tmp, true);
        unlink($tmp);
        self::checkError();
        if (\FFI::isNull($cur)) {
            throw new \RuntimeException('[DataFrame] bootstrap df_read_csv failed');
        }

        /* Append remaining columns */
        for ($c = 1; $c < $nCols; $c++) {
            $buf = $ffi->new("float[{$nRows}]");
            for ($r = 0; $r < $nRows; $r++) {
                $buf[$r] = (float)$flat[$r * $nCols + $c];
            }
            $next = $ffi->df_add_f32_column(
                $cur, $colNames[$c], $ffi->cast('float*', $buf), $nRows
            );
            $ffi->df_free($cur);
            self::checkError();
            if (\FFI::isNull($next)) {
                throw new \RuntimeException("[DataFrame] df_add_f32_column failed for '{$colNames[$c]}'");
            }
            $cur = $next;
        }
        return new self($cur);
    }

    // ── Inspection ───────────────────────────────────────────────────────

    /** [n_rows, n_cols] */
    public function shape(): array
    {
        return [(int)self::ffi()->df_num_rows($this->ptr),
                (int)self::ffi()->df_num_cols($this->ptr)];
    }

    public function numRows(): int { return (int)self::ffi()->df_num_rows($this->ptr); }
    public function numCols(): int { return (int)self::ffi()->df_num_cols($this->ptr); }

    /** @return list<string> */
    public function columns(): array
    {
        $ffi = self::ffi();
        $n   = (int)$ffi->df_num_cols($this->ptr);
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[] = self::cstr($ffi->df_col_name($this->ptr, $i));
        }
        return $out;
    }

    /**
     * @return array<string,string>  name → dtype string (float32|int32|string)
     */
    public function dtypes(): array
    {
        $ffi    = self::ffi();
        $n      = (int)$ffi->df_num_cols($this->ptr);
        $labels = ['float32', 'int32', 'string'];
        $out    = [];
        for ($i = 0; $i < $n; $i++) {
            $name  = self::cstr($ffi->df_col_name($this->ptr, $i));
            $dtype = (int)$ffi->df_col_dtype($this->ptr, $i);
            $out[$name] = $labels[$dtype] ?? 'unknown';
        }
        return $out;
    }

    /**
     * Summary stats for all FLOAT32 columns.
     * Returns Tensor [n_float_cols × 5]: [count, mean, std, min, max].
     */
    public function describe(): Tensor
    {
        $ptr = self::ffi()->df_describe($this->ptr);
        self::checkError();
        return Tensor::wrap($ptr);
    }

    /**
     * Category names for a STRING column.
     * @return list<string>
     */
    public function categories(string $col): array
    {
        $idx = $this->colIdx($col);
        $ffi = self::ffi();
        $n   = (int)$ffi->df_col_n_categories($this->ptr, $idx);
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[] = self::cstr($ffi->df_col_category_name($this->ptr, $idx, $i));
        }
        return $out;
    }

    // ── Selection ────────────────────────────────────────────────────────

    /** Return first $n rows (zero-copy view via df_head_rows). */
    public function head(int $n = 5): self
    {
        $ptr = self::ffi()->df_head_rows($this->ptr, $n);
        self::checkError();
        return new self($ptr);
    }

    /** Return last $n rows. */
    public function tail(int $n = 5): self
    {
        $nRows = $this->numRows();
        $offset = max(0, $nRows - $n);
        return $this->iloc($offset, min($n, $nRows));
    }

    /**
     * Row slice: rows [offset, offset+length).
     */
    public function iloc(int $offset, int $length): self
    {
        $ptr = self::ffi()->df_slice_rows($this->ptr, $offset, $length);
        self::checkError();
        return new self($ptr);
    }

    /**
     * Select columns by name.
     * @param list<string> $cols
     */
    public function select(array $cols): self
    {
        $indices = array_map([$this, 'colIdx'], $cols);
        return $this->_selectByIndices($indices);
    }

    /**
     * Drop columns by name.
     * @param list<string> $cols
     */
    public function drop(array $cols): self
    {
        $drop = array_flip(array_map([$this, 'colIdx'], $cols));
        $keep = [];
        foreach (range(0, $this->numCols() - 1) as $i) {
            if (!isset($drop[$i])) $keep[] = $i;
        }
        return $this->_selectByIndices($keep);
    }

    /**
     * Get a single column as a Tensor (zero-copy via df_to_tensor).
     * Only works for FLOAT32 and INT32 columns.
     */
    public function col(string $name): Tensor
    {
        $idx = $this->colIdx($name);
        return $this->_selectByIndices([$idx])->toTensor();
    }

    // ── Filtering ────────────────────────────────────────────────────────

    /**
     * Filter rows where $col $op $val.
     *
     * For FLOAT32/INT32 columns: $val must be numeric.
     * For STRING columns: only '==' is supported; $val must be string.
     *
     * @param string     $col  column name
     * @param string     $op   '==' | '!=' | '>' | '>=' | '<' | '<='
     * @param float|string $val scalar threshold
     */
    public function where(string $col, string $op, float|string $val): self
    {
        $ffi = self::ffi();
        $idx = $this->colIdx($col);
        $dtype = (int)$ffi->df_col_dtype($this->ptr, $idx);

        if ($dtype === 2) { /* STRING */
            if ($op !== '==') {
                throw new \InvalidArgumentException(
                    "[DataFrame] STRING columns only support '==' in where()"
                );
            }
            $ptr = $ffi->df_where_str($this->ptr, $idx, (string)$val);
        } else {
            $ptr = $ffi->df_where_f32($this->ptr, $idx, self::cmpOp($op), (float)$val);
        }
        self::checkError();
        return new self($ptr);
    }

    /**
     * Drop rows containing nulls (NaN / INT32_MIN / category -1).
     */
    public function dropNulls(): self
    {
        $ptr = self::ffi()->df_drop_nans($this->ptr);
        self::checkError();
        return new self($ptr);
    }

    // ── Mutation (returns new DataFrame) ─────────────────────────────────

    /**
     * Rename columns: ['old_name' => 'new_name', ...].
     * Renames are applied in-place on a deep copy to preserve immutability.
     */
    public function rename(array $mapping): self
    {
        // Deep copy via 100%-row slice
        $nRows = $this->numRows();
        $copy  = $nRows > 0
            ? new self(self::ffi()->df_slice_rows($this->ptr, 0, $nRows))
            : new self(self::ffi()->df_head_rows($this->ptr, 0));
        self::checkError();

        foreach ($mapping as $old => $new) {
            if (isset($copy->colIndex[$old])) {
                self::ffi()->df_rename_column($copy->ptr, $copy->colIndex[$old], (string)$new);
            }
        }
        $copy->_buildIndex();
        return $copy;
    }

    /**
     * Add or replace a FLOAT32 column.
     * $data must be a 1-D Tensor with length == numRows().
     */
    public function withColumn(string $name, Tensor $data): self
    {
        $nRows = $this->numRows();
        if ($data->size() !== $nRows) {
            throw new \InvalidArgumentException(
                "[DataFrame] withColumn: tensor length {$data->size()} != row count {$nRows}"
            );
        }
        // Drop existing column with the same name, then add new one.
        $base = isset($this->colIndex[$name]) ? $this->drop([$name]) : $this;
        $flat = $data->toFlatArray();

        $ffi  = self::ffi();
        $buf  = $ffi->new("float[{$nRows}]");
        for ($i = 0; $i < $nRows; $i++) $buf[$i] = $flat[$i];

        $ptr  = $ffi->df_add_f32_column($base->ptr, $name, $ffi->cast('float*', $buf), $nRows);
        self::checkError();
        return new self($ptr);
    }

    /**
     * Add a pre-computed Tensor as a new FLOAT32 column (zero-copy from C side).
     * Faster than withColumn() when tensor is already contiguous.
     */
    public function withTensorColumn(string $name, Tensor $data): self
    {
        $ffi = self::ffi();
        $ptr = $ffi->df_add_tensor_f32_column($this->ptr, $name, $data->ptr);
        self::checkError();
        return new self($ptr);
    }

    // ── Categorical Encoding ─────────────────────────────────────────────────

    /**
     * Fit target encoding: compute per-category smoothed mean of y [N].
     * Returns [n_cats] tensor of smoothed means.
     * @param float $smoothing James-Stein additive smoothing weight (default 10).
     */
    public function targetEncodeFit(string $col, Tensor $y, float $smoothing = 10.0): Tensor
    {
        $ptr = self::ffi()->df_target_encode_fit(
            $this->ptr, $this->colIdx($col), $y->ptr, $smoothing
        );
        self::checkError();
        return Tensor::wrap($ptr);
    }

    /**
     * Apply target encoding: map each row's category → smoothed mean.
     * @param Tensor $catMeans  [n_cats] tensor from targetEncodeFit()
     * @param float  $globalMean  Fallback for missing/unseen categories
     * Returns [N] FLOAT32 tensor.
     */
    public function targetEncodeTransform(
        string $col, Tensor $catMeans, float $globalMean
    ): Tensor {
        $ptr = self::ffi()->df_target_encode_transform(
            $this->ptr, $this->colIdx($col), $catMeans->ptr, $globalMean
        );
        self::checkError();
        return Tensor::wrap($ptr);
    }

    /**
     * Fit frequency encoding: compute category fractions.
     * Returns [n_cats] tensor of frequencies (0–1).
     */
    public function freqEncodeFit(string $col): Tensor
    {
        $ptr = self::ffi()->df_freq_encode_fit($this->ptr, $this->colIdx($col));
        self::checkError();
        return Tensor::wrap($ptr);
    }

    /**
     * Apply frequency encoding: map each row's category → frequency.
     * @param Tensor $catFreqs  [n_cats] tensor from freqEncodeFit()
     * Returns [N] FLOAT32 tensor.
     */
    public function freqEncodeTransform(string $col, Tensor $catFreqs): Tensor
    {
        $ptr = self::ffi()->df_freq_encode_transform(
            $this->ptr, $this->colIdx($col), $catFreqs->ptr
        );
        self::checkError();
        return Tensor::wrap($ptr);
    }

    /**
     * Cast column to FLOAT32.
     * Applicable to INT32 columns; INT32_MIN → NaN.
     */
    public function castToFloat(string $col): self
    {
        $ptr = self::ffi()->df_cast_to_f32($this->ptr, $this->colIdx($col));
        self::checkError();
        return new self($ptr);
    }

    /**
     * Fill NaN values in a FLOAT32 column with $value.
     */
    public function fillNull(string $col, float $value): self
    {
        $ptr = self::ffi()->df_fill_null_f32($this->ptr, $this->colIdx($col), $value);
        self::checkError();
        return new self($ptr);
    }

    /**
     * One-hot encode a STRING column.
     * The STRING column is replaced by n_categories FLOAT32 columns.
     */
    public function oneHotEncode(string $col): self
    {
        $ptr = self::ffi()->df_one_hot_encode($this->ptr, $this->colIdx($col));
        self::checkError();
        return new self($ptr);
    }

    // ── Aggregation & Sorting ─────────────────────────────────────────────

    /**
     * Sort rows by a column (ascending by default).
     */
    public function sortBy(string $col, bool $ascending = true): self
    {
        $ptr = self::ffi()->df_sort_by_col($this->ptr, $this->colIdx($col), $ascending);
        self::checkError();
        return new self($ptr);
    }

    /**
     * Frequency table for a STRING column.
     * Returns new DataFrame [category(STRING) | count(FLOAT32)], sorted desc.
     */
    public function valueCounts(string $col): self
    {
        $ptr = self::ffi()->df_value_counts($this->ptr, $this->colIdx($col));
        self::checkError();
        return new self($ptr);
    }

    /**
     * Random sample of $n rows.
     *
     * @param bool     $replace  sampling with replacement
     * @param int|null $seed     0 or null → use clock seed
     */
    public function sample(int $n, bool $replace = false, ?int $seed = null): self
    {
        $ptr = self::ffi()->df_sample_rows($this->ptr, $n, $replace, (int)($seed ?? 0));
        self::checkError();
        return new self($ptr);
    }

    /**
     * Begin a GroupBy chain.
     *
     * @param string $col  Must be a STRING (categorical) column.
     */
    public function groupBy(string $col): GroupBy
    {
        return new GroupBy($this, $this->colIdx($col));
    }

    // ── Join / Concat ─────────────────────────────────────────────────────

    /**
     * Equijoin on a single column shared by both DataFrames.
     *
     * @param string $on   column name present in both DataFrames
     * @param string $how  'inner' | 'left'
     */
    public function join(self $right, string $on, string $how = 'inner'): self
    {
        $leftIdx  = $this->colIdx($on);
        $rightIdx = $right->colIdx($on);
        return $this->merge($right, $leftIdx, $rightIdx, $how);
    }

    /**
     * Equijoin on (possibly different) key columns.
     */
    public function merge(self $right, int|string $leftOn, int|string $rightOn,
                           string $how = 'inner'): self
    {
        $li = is_int($leftOn)  ? $leftOn  : $this->colIdx($leftOn);
        $ri = is_int($rightOn) ? $rightOn : $right->colIdx($rightOn);

        $joinType = match (strtolower($how)) {
            'left'  => self::JOIN_LEFT,
            default => self::JOIN_INNER,
        };
        $ptr = self::ffi()->df_join($this->ptr, $right->ptr, $li, $ri, $joinType);
        self::checkError();
        return new self($ptr);
    }

    /**
     * Vertically concatenate DataFrames with matching schemas.
     * @param list<self> $frames
     */
    public static function concat(array $frames): self
    {
        if (empty($frames)) {
            throw new \InvalidArgumentException('[DataFrame] concat: empty array');
        }
        $acc = $frames[0];
        for ($i = 1; $i < count($frames); $i++) {
            $ptr = self::ffi()->df_concat_rows($acc->ptr, $frames[$i]->ptr);
            self::checkError();
            $acc = new self($ptr);
        }
        return $acc;
    }

    // ── Conversion ───────────────────────────────────────────────────────

    /**
     * Pack FLOAT32/INT32 columns into a [n_rows × n_cols] FLOAT32 Tensor.
     * If $cols is null, all numeric columns are packed.
     *
     * @param list<string>|null $cols
     */
    public function toTensor(?array $cols = null): Tensor
    {
        $ffi = self::ffi();
        if ($cols === null) {
            /* Pack all numeric columns */
            $n = $this->numCols();
            $indices = [];
            for ($i = 0; $i < $n; $i++) {
                $dtype = (int)$ffi->df_col_dtype($this->ptr, $i);
                if ($dtype !== 2) $indices[] = $i; /* 0=float, 1=int */
            }
        } else {
            $indices = array_map([$this, 'colIdx'], $cols);
        }

        $nIdx = count($indices);
        if ($nIdx === 0) {
            throw new \RuntimeException('[DataFrame] toTensor: no numeric columns to pack');
        }

        $cIdx = $ffi->new("int[{$nIdx}]");
        foreach ($indices as $k => $v) $cIdx[$k] = $v;

        $ptr = $ffi->df_to_tensor($this->ptr, $ffi->cast('int*', $cIdx), $nIdx);
        self::checkError();
        return Tensor::wrap($ptr);
    }

    /**
     * Export to a PHP array of associative rows.
     * CAUTION: returns PHP arrays — use only for small DataFrames / display.
     */
    public function toArray(): array
    {
        $ffi   = self::ffi();
        $nRows = $this->numRows();
        $nCols = $this->numCols();
        $names = $this->columns();
        $dtypes = [];
        for ($c = 0; $c < $nCols; $c++) $dtypes[$c] = (int)$ffi->df_col_dtype($this->ptr, $c);

        $result = [];
        for ($r = 0; $r < $nRows; $r++) {
            $row = [];
            for ($c = 0; $c < $nCols; $c++) {
                /* Slow path: one element at a time — use only for small frames */
                $tensor = $this->_selectByIndices([$c])
                               ->iloc($r, 1)
                               ->toTensor();
                $flat = $tensor->toFlatArray();
                if ($dtypes[$c] === 2) {
                    /* STRING: look up category name */
                    $catIdx = (int)$flat[0];
                    $catPtr = $ffi->df_col_category_name($this->ptr, $c, $catIdx);
                    $row[$names[$c]] = $catPtr ? \FFI::string($catPtr) : null;
                } else {
                    $row[$names[$c]] = $flat[0];
                }
            }
            $result[] = $row;
        }
        return $result;
    }

    // ── Internal access for GroupBy ───────────────────────────────────────

    /** @internal */
    public function ptr(): \FFI\CData  { return $this->ptr; }

    /** @internal */
    public function colIdx(string $name): int
    {
        if (!isset($this->colIndex[$name])) {
            throw new \InvalidArgumentException(
                "[DataFrame] column '{$name}' not found. Available: "
                . implode(', ', array_keys($this->colIndex))
            );
        }
        return $this->colIndex[$name];
    }

    // ── Private helpers ──────────────────────────────────────────────────

    private function _buildIndex(): void
    {
        $this->colIndex = [];
        $ffi = self::ffi();
        $n   = (int)$ffi->df_num_cols($this->ptr);
        for ($i = 0; $i < $n; $i++) {
            $namePtr = $ffi->df_col_name($this->ptr, $i);
            if ($namePtr === null) continue;
            /* PHP FFI returns const char* as either CData or a plain string */
            $name = $namePtr instanceof \FFI\CData
                ? \FFI::string($namePtr)
                : (string)$namePtr;
            if ($name !== '') $this->colIndex[$name] = $i;
        }
    }

    private function _selectByIndices(array $indices): self
    {
        $ffi = self::ffi();
        $n   = count($indices);
        if ($n === 0) {
            throw new \InvalidArgumentException('[DataFrame] select: no columns specified');
        }
        $cIdx = $ffi->new("int[{$n}]");
        foreach ($indices as $k => $v) $cIdx[$k] = $v;
        $ptr = $ffi->df_select_columns($this->ptr, $ffi->cast('int*', $cIdx), $n);
        self::checkError();
        return new self($ptr);
    }

    private static function cmpOp(string $op): int
    {
        return match ($op) {
            '=='    => self::CMP_EQ,
            '!='    => self::CMP_NEQ,
            '>'     => self::CMP_GT,
            '>='    => self::CMP_GTE,
            '<'     => self::CMP_LT,
            '<='    => self::CMP_LTE,
            default => throw new \InvalidArgumentException("[DataFrame] unknown operator: {$op}"),
        };
    }

    private static function checkError(): void
    {
        $ffi = self::ffi();
        if ($ffi->tensor_check_error()) {
            $errPtr = $ffi->tensor_get_last_error();
            $msg    = $errPtr instanceof \FFI\CData ? \FFI::string($errPtr) : (string)$errPtr;
            $ffi->tensor_clear_error();
            throw new \RuntimeException('[DataFrame] ' . $msg);
        }
    }

    /**
     * PHP FFI returns `const char*` as either a \FFI\CData or a plain PHP string
     * depending on the runtime version. This helper handles both.
     */
    private static function cstr(mixed $ptr): string
    {
        if ($ptr === null) return '';
        return $ptr instanceof \FFI\CData ? \FFI::string($ptr) : (string)$ptr;
    }
}
