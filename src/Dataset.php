<?php
declare(strict_types=1);

namespace Pml;

use Pml\Lib\TensorEngine;
use InvalidArgumentException;
use RuntimeException;

/**
 * High-Performance Dataset — RubixML-style API backed by C-level memory.
 *
 * ─── Two internal modes ──────────────────────────────────────────────────────
 *
 * ETL mode   ($dfPtr is set, $samples is null)
 *   The dataset lives as a columnar C DataFrame — mixed types are preserved,
 *   no PHP arrays are ever allocated.  ETL operations (dropNans, oneHotEncode,
 *   selectColumns) return a new Dataset still in ETL mode.  ETL mode is entered
 *   via Dataset::load() or Dataset::fromCSV() on a CSV with non-numeric columns.
 *
 * Tensor mode  ($samples is set, $dfPtr is null)
 *   The dataset lives as two homogeneous FLOAT32 Tensors: $samples [N×D] and
 *   optional $labels [N].  All ML operations (split, fold, batches, etc.) work
 *   exclusively in this mode.
 *
 * ─── Lazy materialisation ─────────────────────────────────────────────────
 *   Any method that needs Tensor access calls _ensureTensorMode() internally.
 *   That converts the C DataFrame into Tensors via df_to_tensor() then frees
 *   the DataFrame pointer — from that point the object is in Tensor mode and
 *   the C DataFrame no longer exists.
 *
 * ─── Memory contract ─────────────────────────────────────────────────────
 *   - Tensor wrappers call tensor_free() in their own __destruct().
 *   - __destruct() here calls df_free() when in ETL mode.
 *   - No double-free: $dfPtr is nulled immediately after df_free().
 *
 * @final — enables JIT devirtualisation + method inlining.
 */
final class Dataset
{
    // ── Tensor mode ─────────────────────────────────────────────────────────
    private ?Tensor $samples    = null;
    private ?Tensor $labels     = null;

    /** JIT cache: prevents FFI boundary crossing in hot loops */
    private int $numRows        = 0;
    private int $numColumns     = 0;

    // ── Reproducibility ─────────────────────────────────────────────────────
    /** Process-wide seed for randomize(). null = non-deterministic (default). */
    private static ?int $globalSeed = null;

    /**
     * Set a global seed so that all subsequent randomize() calls produce the
     * same shuffle order.  Enables fully reproducible training runs.
     *
     * Pass null to revert to non-deterministic behaviour.
     */
    public static function seed(?int $seed): void
    {
        self::$globalSeed = $seed;
        if ($seed !== null) {
            \mt_srand($seed);
        }
    }

    // ── ETL mode ─────────────────────────────────────────────────────────
    /** Opaque C DataFrame* pointer; null when in Tensor mode. */
    private ?\FFI\CData $dfPtr      = null;
    /** Which df column becomes $labels after materialisation (-1 = none). */
    private int          $dfLabelCol = -1;

    // =========================================================================
    // Construction & destruction
    // =========================================================================

    /**
     * Direct Tensor-mode constructor — preserves backward compatibility.
     * Use Dataset::load() for ETL-mode construction from CSV.
     */
    public function __construct(Tensor $samples, ?Tensor $labels = null)
    {
        $shape = $samples->shape();
        if ($labels !== null && $shape[0] !== $labels->shape()[0]) {
            throw new InvalidArgumentException(
                'Number of samples must match number of labels.'
            );
        }
        $this->samples    = $samples;
        $this->labels     = $labels;
        $this->numRows    = $shape[0];
        $this->numColumns = $shape[1] ?? 1;
    }

    public function __destruct()
    {
        // Tensor objects manage their own C memory via their own __destruct.
        // We only need to handle the ETL-mode C DataFrame pointer.
        if ($this->dfPtr !== null) {
            TensorEngine::get()->df_free($this->dfPtr);
            $this->dfPtr = null;
        }
    }

    // =========================================================================
    // Factories
    // =========================================================================

    /**
     * Load a CSV into ETL mode — the C DataFrame is created but NOT yet
     * converted to Tensors.  Chain ETL operations before ML use:
     *
     *   $ds = Dataset::load('train.csv')
     *       ->dropNans()
     *       ->oneHotEncode(2)        // column 2 is categorical
     *       ->materialize(labelCol: 0);
     *
     * ETL methods return NEW Dataset objects; the original is unchanged.
     * PHP GC / explicit unset() will call df_free() on each intermediate.
     */
    public static function load(string $filepath, bool $hasHeader = true): self
    {
        if (!file_exists($filepath)) {
            throw new RuntimeException("File not found: {$filepath}");
        }
        $ffi = TensorEngine::get();
        $ptr = $ffi->df_read_csv($filepath, $hasHeader);
        self::_checkCError();
        return self::_fromDfPtr($ptr, -1);
    }

    /**
     * Fast numeric-only CSV ingestion — identical to the original behaviour.
     * Uses the legacy fgets-based path (tensor_dataset_from_csv) for pure-float
     * CSVs.  Falls back to the ETL path + auto-materialise when the file
     * contains non-numeric columns.
     *
     * @param int $labelColumn  0-based index of the label column; -1 = no label.
     */
    public static function fromCSV(
        string $filepath,
        int    $labelColumn = -1,
        bool   $hasHeader   = true
    ): self {
        if (!file_exists($filepath)) {
            throw new RuntimeException("Dataset file not found: {$filepath}");
        }

        $ffi = TensorEngine::get();

        /* ── Fast path: pure-numeric CSV → direct Tensor ingestion ─────────
         * tensor_dataset_from_csv() is still the fastest route for fully
         * numeric datasets (one fgets loop, no type detection, no mmap).   */
        $ptrArray = $ffi->tensor_dataset_from_csv(
            $filepath, $labelColumn, $hasHeader ? 1 : 0
        );
        if ($ptrArray !== null) {
            $samples = Tensor::wrap($ptrArray[0]);
            $labels  = ($labelColumn >= 0 && $ptrArray[1] !== null)
                     ? Tensor::wrap($ptrArray[1]) : null;
            $ffi->free($ptrArray);
            return new self($samples, $labels);
        }

        /* ── Fallback: mixed-type CSV → ETL path + immediate materialise ───
         * tensor_dataset_from_csv returns NULL for non-numeric fields.      */
        $ptr = $ffi->df_read_csv($filepath, $hasHeader);
        self::_checkCError();
        return self::_fromDfPtr($ptr, $labelColumn)->materialize($labelColumn);
    }

    /**
     * Build a Dataset from PHP arrays (unchanged from original).
     */
    public static function fromArray(array $samples, ?array $labels = null): self
    {
        return new self(
            Tensor::fromArray($samples),
            $labels !== null ? Tensor::fromArray($labels) : null
        );
    }

    // =========================================================================
    // ETL operations  (ETL mode → new Dataset in ETL mode)
    // =========================================================================

    /**
     * Declare which column index is the label column for this ETL-mode Dataset.
     *
     * Dataset::load() sets dfLabelCol = -1 (no label).  Call this to tell the
     * ETL pipeline (and WordCountVectorizer) which column holds the targets so
     * that label extraction works in transform().
     *
     * @param int $col  0-based column index (-1 to clear).
     * @return $this    Fluent; mutates in-place (same ETL object).
     */
    public function withLabelColumn(int $col): self
    {
        $this->_requireEtlMode(__METHOD__);
        $this->dfLabelCol = $col;
        return $this;
    }

    /**
     * Extract ONLY the label column from the C DataFrame as a 1-D Tensor.
     *
     * Unlike materialize(), this does NOT touch feature columns — so it works
     * even when the DataFrame contains non-numeric columns (e.g. text) that
     * df_to_tensor cannot pack.
     *
     * Returns null if no label column has been set (dfLabelCol < 0).
     */
    public function extractLabelTensor(): ?Tensor
    {
        $this->_requireEtlMode(__METHOD__);
        if ($this->dfLabelCol < 0) {
            return null;
        }
        $ffi  = TensorEngine::get();
        $n    = (int) $ffi->df_num_cols($this->dfPtr);
        if ($this->dfLabelCol >= $n) {
            return null;
        }
        $lIdx    = $ffi->new('int[1]');
        $lIdx[0] = $this->dfLabelCol;
        $lPtr    = $ffi->df_to_tensor($this->dfPtr, $ffi->cast('int*', $lIdx), 1);
        self::_checkCError();
        return Tensor::wrap($lPtr)->flatten(); // [N×1] → [N]
    }

    /**
     * Remove every row that contains at least one missing value.
     * Sentinels: NaN (FLOAT32), INT32_MIN (INT32), category index < 0 (STRING).
     * Requires ETL mode — call on the result of Dataset::load().
     */
    public function dropNans(): self
    {
        $this->_requireEtlMode(__METHOD__);
        $ffi    = TensorEngine::get();
        $newPtr = $ffi->df_drop_nans($this->dfPtr);
        self::_checkCError();
        return self::_fromDfPtr($newPtr, $this->dfLabelCol);
    }

    /**
     * Return a new ETL-mode Dataset containing rows [$offset, $offset+$n).
     * STRING columns are category-compacted to only used entries.
     * Clamps to available rows; preserves dfLabelCol.
     */
    public function sliceRowsEtl(int $offset, int $n): self
    {
        $this->_requireEtlMode(__METHOD__);
        $ffi    = TensorEngine::get();
        $newPtr = $ffi->df_slice_rows($this->dfPtr, $offset, $n);
        self::_checkCError();
        return self::_fromDfPtr($newPtr, $this->dfLabelCol);
    }

    /**
     * Convenience — equivalent to sliceRowsEtl(0, $n).
     */
    public function headRows(int $n): self
    {
        return $this->sliceRowsEtl(0, $n);
    }

    /**
     * One-hot encode the STRING column at $colIdx.
     *
     * Replaces the column with K binary FLOAT32 columns named
     * "{original_name}_{category_value}". The returned Dataset is in ETL mode.
     *
     * If $colIdx is the current label column the label assignment is cleared
     * (you should re-assign a label column in materialize()).
     */
    public function oneHotEncode(int $colIdx): self
    {
        $this->_requireEtlMode(__METHOD__);
        $ffi    = TensorEngine::get();
        $newPtr = $ffi->df_one_hot_encode($this->dfPtr, $colIdx);
        self::_checkCError();
        /* If we just expanded the label column, invalidate the label assignment */
        $newLabelCol = ($colIdx === $this->dfLabelCol) ? -1 : $this->dfLabelCol;
        return self::_fromDfPtr($newPtr, $newLabelCol);
    }

    /**
     * Return a new Dataset keeping only the specified column indices.
     * Operates in ETL mode; result is also ETL mode.
     *
     * @param int[] $colIndices  0-based column indices to keep.
     */
    public function selectColumns(array $colIndices): self
    {
        $this->_requireEtlMode(__METHOD__);
        $ffi  = TensorEngine::get();
        $n    = \count($colIndices);
        $cIdx = $ffi->new("int[$n]");
        foreach ($colIndices as $i => $v) { $cIdx[$i] = (int) $v; }
        $newPtr = $ffi->df_select_columns($this->dfPtr, $ffi->cast('int*', $cIdx), $n);
        self::_checkCError();
        /* Remap label column index after column selection */
        $newLabel = -1;
        foreach ($colIndices as $i => $v) {
            if ((int)$v === $this->dfLabelCol) { $newLabel = $i; break; }
        }
        return self::_fromDfPtr($newPtr, $newLabel);
    }

    /**
     * Convert from ETL mode to Tensor mode.
     *
     * @param int|null $labelCol  Column index that becomes $labels. Pass null
     *                            to inherit the label column set at load time,
     *                            or -1 to produce an unlabelled Dataset.
     */
    public function materialize(?int $labelCol = null): self
    {
        $this->_requireEtlMode(__METHOD__);
        $ffi     = TensorEngine::get();
        $useLabel = $labelCol ?? $this->dfLabelCol;

        $n = (int) $ffi->df_num_cols($this->dfPtr);

        /* ── Feature columns: all columns except the label column ─────────── */
        $featIdx = [];
        for ($c = 0; $c < $n; $c++) {
            if ($c !== $useLabel) $featIdx[] = $c;
        }
        $nf   = \count($featIdx);
        $cIdx = $ffi->new("int[$nf]");
        foreach ($featIdx as $i => $c) { $cIdx[$i] = $c; }

        $sPtr   = $ffi->df_to_tensor($this->dfPtr, $ffi->cast('int*', $cIdx), $nf);
        self::_checkCError();
        $samples = Tensor::wrap($sPtr);

        /* ── Label column ──────────────────────────────────────────────────── */
        $labels = null;
        if ($useLabel >= 0 && $useLabel < $n) {
            $lIdx    = $ffi->new('int[1]');
            $lIdx[0] = $useLabel;
            $lPtr    = $ffi->df_to_tensor($this->dfPtr, $ffi->cast('int*', $lIdx), 1);
            self::_checkCError();
            /* df_to_tensor produces [N×1]; squeeze to [N] */
            $labels = Tensor::wrap($lPtr)->flatten();
        }

        /* Free the C DataFrame — we no longer need it */
        $ffi->df_free($this->dfPtr);
        $this->dfPtr = null;

        return new self($samples, $labels);
    }

    // =========================================================================
    // Introspection — works in BOTH modes
    // =========================================================================

    /**
     * Column schema — only meaningful in ETL mode (before materialisation).
     * Returns [] in Tensor mode where type information has been collapsed.
     *
     * @return array<int, array{name: string, dtype: int, n_categories: int}>
     */
    public function schema(): array
	{
		if ($this->dfPtr === null) return [];
		$ffi  = TensorEngine::get();
		$n    = (int) $ffi->df_num_cols($this->dfPtr);
		$cols = [];
		for ($c = 0; $c < $n; $c++) {
			$raw = $ffi->df_col_name($this->dfPtr, $c);
			$name = ($raw instanceof \FFI\CData) ? \FFI::string($raw) : (string)$raw;
			$cols[] = [
				'name'         => $name ?: "col_{$c}",
				'dtype'        => (int) $ffi->df_col_dtype($this->dfPtr, $c),
				'n_categories' => (int) $ffi->df_col_n_categories($this->dfPtr, $c),
			];
		}
		return $cols;
	}

	public function categories(int $colIdx): array
	{
		$this->_requireEtlMode(__METHOD__);
		$ffi  = TensorEngine::get();
		$n    = (int) $ffi->df_col_n_categories($this->dfPtr, $colIdx);
		$cats = [];
		for ($i = 0; $i < $n; $i++) {
			$raw = $ffi->df_col_category_name($this->dfPtr, $colIdx, $i);
			$cats[] = ($raw instanceof \FFI\CData) ? \FFI::string($raw) : (string)$raw;
		}
		return $cats;
	}

    // =========================================================================
    // Properties  (Tensor mode; auto-materialises if needed)
    // =========================================================================

    public function samples(): Tensor  { $this->_ensureTensorMode(); return $this->samples; }
    public function labels(): ?Tensor  { $this->_ensureTensorMode(); return $this->labels;  }

    /** Return the raw C DataFrame* pointer (ETL mode only). */
    public function rawDfPtr(): ?\FFI\CData { return $this->dfPtr; }

    public function numRows(): int {
        /* Avoid materialising just to count — ask C directly in ETL mode */
        if ($this->dfPtr !== null) {
            return (int) TensorEngine::get()->df_num_rows($this->dfPtr);
        }
        return $this->numRows;
    }

    public function numColumns(): int {
        if ($this->dfPtr !== null) {
            $ffi = TensorEngine::get();
            $n   = (int) $ffi->df_num_cols($this->dfPtr);
            /* Subtract label column from feature count */
            return $this->dfLabelCol >= 0 ? $n - 1 : $n;
        }
        return $this->numColumns;
    }

    public function isLabeled(): bool
    {
        if ($this->dfPtr !== null) return $this->dfLabelCol >= 0;
        return $this->labels !== null;
    }

    // =========================================================================
    // Selecting & Dropping (column operations, Tensor mode)
    // =========================================================================

    /** Return a new Dataset keeping only the specified feature columns. */
    public function select(array $columns): self
    {
        $this->_ensureTensorMode();
        $indices    = Tensor::fromArray($columns);
        $newSamples = $this->samples->take($indices, 1);
        return new self($newSamples, $this->labels);
    }

    /** Return a new Dataset with the specified feature columns removed. */
    public function drop(array $columns): self
    {
        $this->_ensureTensorMode();
        $keep = array_values(array_diff(range(0, $this->numColumns - 1), $columns));
        return $this->select($keep);
    }

    // =========================================================================
    // Slicing & splicing (Tensor mode, all zero-copy)
    // =========================================================================

    /** Zero-copy view of the first N rows. */
    public function head(int $n = 10): self
    {
        $this->_ensureTensorMode();
        return $this->slice(0, min($n, $this->numRows));
    }

    /** Zero-copy view of the last N rows. */
    public function tail(int $n = 10): self
    {
        $this->_ensureTensorMode();
        $n = min($n, $this->numRows);
        return $this->slice($this->numRows - $n, $n);
    }

    /** Zero-copy row slice. */
    public function slice(int $offset, int $length): self
    {
        $this->_ensureTensorMode();
        $s = $this->samples->slice(0, $offset, $length);
        $l = $this->labels  ? $this->labels->slice(0, $offset, $length) : null;
        return new self($s, $l);
    }

    /**
     * Extract the first N rows into a new Dataset and REMOVE them from this one.
     * Mutates this Dataset's internal state.
     */
    public function take(int $n): self
    {
        $this->_ensureTensorMode();
        $n     = min($n, $this->numRows);
        $chunk = $this->head($n);
        $rem   = $this->numRows - $n;
        if ($rem <= 0) {
            throw new RuntimeException('Cannot take all rows — would leave an empty Dataset.');
        }
        $this->samples    = $this->samples->slice(0, $n, $rem)->copy();
        $this->labels     = $this->labels ? $this->labels->slice(0, $n, $rem)->copy() : null;
        $this->numRows    = $rem;
        return $chunk;
    }

    /** Drop the first N rows permanently from this Dataset. */
    public function leave(int $n): self
    {
        $this->take($n);
        return $this;
    }

    // =========================================================================
    // Splitting & folding (Tensor mode)
    // =========================================================================

    /**
     * Split into [train, test] by ratio.
     * @return array{0: self, 1: self}
     */
    public function split(float $ratio = 0.8): array
    {
        $this->_ensureTensorMode();
        if ($ratio <= 0.0 || $ratio >= 1.0) {
            throw new InvalidArgumentException('Ratio must be between 0 and 1.');
        }
        $n = (int) round($this->numRows * $ratio);
        return [$this->slice(0, $n), $this->slice($n, $this->numRows - $n)];
    }

    /**
     * K-fold cross-validation splits.
     * @return \Generator<array{0: self, 1: self}>  Yields [train, validation]
     */
    public function fold(int $k = 10): \Generator
    {
        $this->_ensureTensorMode();
        $foldSize = (int) floor($this->numRows / $k);
        for ($i = 0; $i < $k; $i++) {
            $offset = $i * $foldSize;
            $length = ($i === $k - 1) ? $this->numRows - $offset : $foldSize;
            $val    = $this->slice($offset, $length);

            $trainS = []; $trainL = [];
            if ($offset > 0) {
                $trainS[] = $this->samples->slice(0, 0, $offset);
                if ($this->labels) $trainL[] = $this->labels->slice(0, 0, $offset);
            }
            $end = $offset + $length;
            if ($end < $this->numRows) {
                $rem = $this->numRows - $end;
                $trainS[] = $this->samples->slice(0, $end, $rem);
                if ($this->labels) $trainL[] = $this->labels->slice(0, $end, $rem);
            }
            yield [
                new self(
                    Tensor::concat($trainS, 0),
                    $this->labels ? Tensor::concat($trainL, 0) : null
                ),
                $val,
            ];
        }
    }

    // =========================================================================
    // Batching & randomisation (Tensor mode)
    // =========================================================================

    /**
     * Zero-copy mini-batches for neural network training.
     * @return \Generator<self>
     */
    public function batches(int $batchSize): \Generator
    {
        $this->_ensureTensorMode();
        $total = $this->numRows;    /* JIT-cached — no FFI in loop */
        for ($start = 0; $start < $total; $start += $batchSize) {
            yield $this->slice($start, min($batchSize, $total - $start));
        }
    }

    /**
     * Shuffle row order in-place via C-level argsort on a random uniform vector.
     */
    public function randomize(): self
    {
        $this->_ensureTensorMode();
        if (self::$globalSeed !== null) {
            \mt_srand(self::$globalSeed);
        }
        $idx           = Tensor::randomUniform([$this->numRows], 0, 1)->argsort();
        $this->samples = $this->samples->take($idx, 0);
        if ($this->labels) $this->labels = $this->labels->take($idx, 0);
        return $this;
    }

    // =========================================================================
    // Transformations (Tensor mode)
    // =========================================================================

    /** Standardise features to zero-mean / unit-variance in-place (C-level). */
    public function standardize(): self
    {
        $this->_ensureTensorMode();
        $this->samples->standardizeInplace();
        return $this;
    }

    /** Apply a closure to the underlying Tensors. */
    public function apply(callable $fn): self
    {
        $this->_ensureTensorMode();
        $fn($this->samples, $this->labels);
        return $this;
    }

    /** Filter rows using a binary float mask Tensor (1.0 = keep, 0.0 = drop). */
    public function filterByMask(Tensor $mask): self
    {
        $this->_ensureTensorMode();
        return new self(
            $this->samples->booleanIndex($mask),
            $this->labels ? $this->labels->booleanIndex($mask) : null
        );
    }

    // =========================================================================
    // Stacking & joining (Tensor mode)
    // =========================================================================

    /** Vertical concat — stack another Dataset below this one. */
    public function stack(Dataset $other): self
    {
        $this->_ensureTensorMode();
        $other->_ensureTensorMode();
        if ($this->numColumns !== $other->numColumns()) {
            throw new InvalidArgumentException(
                'Datasets must have the same number of feature columns to stack.'
            );
        }
        return new self(
            Tensor::concat([$this->samples, $other->samples()], 0),
            ($this->isLabeled() && $other->isLabeled())
                ? Tensor::concat([$this->labels, $other->labels()], 0) : null
        );
    }

    /** Horizontal concat — join feature columns of another Dataset. */
    public function join(Dataset $other): self
    {
        $this->_ensureTensorMode();
        $other->_ensureTensorMode();
        if ($this->numRows !== $other->numRows()) {
            throw new InvalidArgumentException(
                'Datasets must have the same number of rows to join.'
            );
        }
        return new self(
            Tensor::concat([$this->samples, $other->samples()], 1),
            $this->labels
        );
    }

    // =========================================================================
    // Statistics & sorting (Tensor mode)
    // =========================================================================

    /** Column-wise descriptive statistics — computed entirely in C. */
    public function describe(): array
    {
        $this->_ensureTensorMode();
        return [
            'mean' => $this->samples->meanAxis(0)->toFlatArray(),
            'max'  => $this->samples->maxAxis(0)->toFlatArray(),
            'min'  => $this->samples->minAxis(0)->toFlatArray(),
            'sum'  => $this->samples->sumAxis(0)->toFlatArray(),
        ];
    }

    /** Sort all rows by values in one feature column. */
    public function sortByColumn(int $column): self
    {
        $this->_ensureTensorMode();
        if ($column < 0 || $column >= $this->numColumns) {
            throw new InvalidArgumentException('Column index out of bounds.');
        }
        $idx = $this->samples->col($column)->argsort();
        return new self(
            $this->samples->take($idx, 0),
            $this->labels ? $this->labels->take($idx, 0) : null
        );
    }

    // =========================================================================
    // Export (Tensor mode)
    // =========================================================================

    /**
     * Pull all data from C memory into a PHP array.
     * WARNING: This is the only method that crosses the C → PHP boundary for
     * bulk data.  Use only for final output, not inside training loops.
     */
    public function toArray(): array
    {
        $this->_ensureTensorMode();
        $rows       = $this->numRows;
        $cols       = $this->numColumns;
        $flatS      = $this->samples->toFlatArray();
        $flatL      = $this->labels ? $this->labels->toFlatArray() : [];
        $data       = [];
        for ($i = 0; $i < $rows; $i++) {
            $row = \array_slice($flatS, $i * $cols, $cols);  /* \-prefix: JIT global-ns optimisation */
            if ($this->labels) $row[] = $flatL[$i];
            $data[] = $row;
        }
        return $data;
    }

    /** Export to CSV file. */
    public function toCSV(string $filepath): void
    {
        $fp = fopen($filepath, 'w');
        if (!$fp) throw new RuntimeException("Could not open file for writing: {$filepath}");
        foreach ($this->toArray() as $row) fputcsv($fp, $row);
        fclose($fp);
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /**
     * Wrap a raw C DataFrame* into a new ETL-mode Dataset.
     * Private — callers are static factory methods and ETL chaining methods.
     */
    private static function _fromDfPtr(\FFI\CData $ptr, int $labelCol): self
    {
        /* Bypass __construct entirely — ETL mode has no Tensors yet.
         * Mirrors the same pattern used by Tensor::wrap(). */
        $obj = (new \ReflectionClass(self::class))->newInstanceWithoutConstructor();
        $obj->samples    = null;
        $obj->labels     = null;
        $obj->numRows    = 0;
        $obj->numColumns = 0;
        $obj->dfPtr      = $ptr;
        $obj->dfLabelCol = $labelCol;
        return $obj;
    }

    /**
     * Lazy materialisation: convert ETL mode → Tensor mode on first Tensor access.
     * After this call, $dfPtr is null and $samples/$labels are set.
     */
    private function _ensureTensorMode(): void
    {
        if ($this->dfPtr === null) return;   /* already in Tensor mode */

        /* Delegate to materialize() which handles the actual conversion */
        $tensor = $this->materialize($this->dfLabelCol);

        $this->samples    = $tensor->samples;
        $this->labels     = $tensor->labels;
        $this->numRows    = $tensor->numRows;
        $this->numColumns = $tensor->numColumns;
        /* $this->dfPtr was set to null inside materialize() */
    }

    /** Assert that the Dataset is in ETL mode; throw otherwise. */
    private function _requireEtlMode(string $method): void
    {
        if ($this->dfPtr === null) {
            throw new RuntimeException(
                "{$method} requires ETL mode. Use Dataset::load() to create an ETL pipeline, " .
                "or chain ETL ops before calling materialize()."
            );
        }
    }

    /** Propagate C-engine errors as PHP RuntimeExceptions. */
    private static function _checkCError(): void
	{
		$ffi = TensorEngine::get();
		if ($ffi->tensor_check_error()) {
			$errPtr = $ffi->tensor_get_last_error();
			// FFI may return const char* as either FFI\CData or plain PHP string
			$err = ($errPtr instanceof \FFI\CData) ? \FFI::string($errPtr) : (string)$errPtr;
			$ffi->tensor_clear_error();
			throw new RuntimeException($err);
		}
	}
    
	/**
	 * Check if the Dataset is in ETL mode (has a C DataFrame pointer).
	 */
	public function isEtlMode(): bool
	{
		return $this->dfPtr !== null;
	}

	/**
	 * Get the opaque C DataFrame pointer.
	 * For internal use by transformers that operate directly on the DataFrame.
	 *
	 * @internal
	 * @return \FFI\CData|null
	 */
	public function getDataFramePointer(): ?\FFI\CData
	{
		return $this->dfPtr;
	}
    
    /**
     * Get the index of a column by name (ETL mode only).
     */
    public function columnIndex(string $name): int
    {
        $schema = $this->schema();
        foreach ($schema as $idx => $col) {
            if ($col['name'] === $name) {
                return $idx;
            }
        }
        throw new RuntimeException("Column '{$name}' not found.");
    }

    /**
     * Check if a column is of STRING type (ETL mode only).
     */
    public function isTextColumn($column): bool
    {
        $idx = is_int($column) ? $column : $this->columnIndex($column);
        $schema = $this->schema();
        return isset($schema[$idx]) && $schema[$idx]['dtype'] === 2;
    }

    /**
     * Apply Bag‑of‑Words transformation to a text column.
     *
     * Convenience wrapper around WordCountVectorizer.
     */
    public function bagOfWords($column, ?int $maxFeatures = null): self
    {
        $colName = \is_int($column) ? $this->schema()[$column]['name'] : $column;
        $vec = new \Pml\Transformers\WordCountVectorizer($maxFeatures, $colName);
        return $vec->fitTransform($this);
    }
}
