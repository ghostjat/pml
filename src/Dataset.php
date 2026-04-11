<?php

declare(strict_types=1);

namespace Pml;

use Pml\Lib\TensorEngine;
use Pml\Tensor; 
use InvalidArgumentException;
use RuntimeException;

/**
 * High-Performance Dataset Object (RubixML Style API).
 * Manages samples and labels natively in continuous C memory via FFI.
 * * JIT OPTIMIZED: Declared `final` to allow method devirtualization and inlining.
 */
final class Dataset
{
    private Tensor $samples;
    private ?Tensor $labels;

    // JIT CACHE: Prevents crossing the FFI boundary on every loop iteration
    private int $numRows;
    private int $numColumns;

    public function __construct(Tensor $samples, ?Tensor $labels = null)
    {
        $sampleShape = $samples->shape();
        
        if ($labels !== null && $sampleShape[0] !== $labels->shape()[0]) {
            throw new InvalidArgumentException("Number of samples must match number of labels.");
        }
        
        $this->samples = $samples;
        $this->labels = $labels;
        
        // Cache the dimensions in PHP userland for lightning-fast loop access
        $this->numRows = $sampleShape[0];
        $this->numColumns = $sampleShape[1] ?? 1;
    }

    // ========================================================================
    // FACTORY METHODS
    // ========================================================================

    /**
     * Ingests a CSV directly into C-memory at disk-bandwidth speeds.
     */
    public static function fromCSV(string $filepath, int $labelColumn = -1, bool $hasHeader = true): self
    {
        if (!file_exists($filepath)) {
            throw new RuntimeException("Dataset file not found: {$filepath}");
        }

        $ffi = TensorEngine::get();
        $ptrArray = $ffi->tensor_dataset_from_csv($filepath, $labelColumn, $hasHeader ? 1 : 0);

        if ($ptrArray === null) {
            throw new RuntimeException("Failed to ingest CSV dataset.");
        }

        $samples = Tensor::wrap($ptrArray[0]);
        $labels = $labelColumn >= 0 ? Tensor::wrap($ptrArray[1]) : null;

        $ffi->free($ptrArray);
        return new self($samples, $labels);
    }

    /**
     * Builds a dataset directly from standard PHP Arrays.
     */
    public static function fromArray(array $samples, ?array $labels = null): self
    {
        $tSamples = Tensor::fromArray($samples);
        $tLabels = $labels !== null ? Tensor::fromArray($labels) : null;
        return new self($tSamples, $tLabels);
    }

    // ========================================================================
    // PROPERTIES
    // ========================================================================

    public function samples(): Tensor { return $this->samples; }
    public function labels(): ?Tensor { return $this->labels; }
    
    // Reads from cached PHP properties instead of FFI
    public function numRows(): int { return $this->numRows; }
    public function numColumns(): int { return $this->numColumns; }
    
    public function isLabeled(): bool { return $this->labels !== null; }

    // ========================================================================
    // SELECTING & DROPPING (Columns)
    // ========================================================================

    /**
     * Returns a new dataset containing only the specified feature columns.
     */
    public function select(array $columns): self
    {
        $indices = Tensor::fromArray($columns);
        // Axis 1 represents the columns. Uses C-level memory gathering.
        $newSamples = $this->samples->take($indices, 1);
        return new self($newSamples, $this->labels);
    }

    /**
     * Returns a new dataset with the specified feature columns removed.
     */
    public function drop(array $columns): self
    {
        $keep = array_values(array_diff(range(0, $this->numColumns - 1), $columns));
        return $this->select($keep);
    }

    // ========================================================================
    // HEAD, TAIL, SLICING & SPLICING (Rows)
    // ========================================================================

    /**
     * Return a zero-copy view of the first N rows.
     */
    public function head(int $n = 10): self
    {
        $n = min($n, $this->numRows);
        return $this->slice(0, $n);
    }

    /**
     * Return a zero-copy view of the last N rows.
     */
    public function tail(int $n = 10): self
    {
        $n = min($n, $this->numRows);
        $offset = $this->numRows - $n;
        return $this->slice($offset, $n);
    }

    /**
     * Returns a specific subset of rows (Zero-Copy View).
     * The underlying C floats are never copied to PHP.
     */
    public function slice(int $offset, int $length): self
    {
        $s = $this->samples->slice(0, $offset, $length);
        $l = $this->labels ? $this->labels->slice(0, $offset, $length) : null;
        return new self($s, $l);
    }

    // ========================================================================
    // TAKING & LEAVING (State Mutators)
    // ========================================================================

    /**
     * Extracts the first N rows into a new dataset, and REMOVES them from this dataset.
     * Mutates the current dataset's state.
     */
    public function take(int $n): self
    {
        $n = min($n, $this->numRows);
        $chunk = $this->head($n);

        $remainder = $this->numRows - $n;
        if ($remainder > 0) {
            // Re-allocate remaining memory to shrink the current dataset safely
            $this->samples = $this->samples->slice(0, $n, $remainder)->copy();
            $this->labels = $this->labels ? $this->labels->slice(0, $n, $remainder)->copy() : null;
            
            // Update the JIT cached values
            $this->numRows = $remainder;
        } else {
            throw new RuntimeException("Cannot take all rows and leave an empty dataset in memory.");
        }

        return $chunk;
    }

    /**
     * Removes the first N rows from the dataset permanently.
     */
    public function leave(int $n): self
    {
        $this->take($n); // We take them and discard the returned chunk
        return $this;
    }

    // ========================================================================
    // SPLITTING & FOLDING (Cross Validation)
    // ========================================================================

    /**
     * Splits the dataset into two disjoint datasets (e.g., Train and Test).
     * @return array{0: self, 1: self} 
     * NOTE: This returns a tuple of Objects, NOT a copied array of floats.
     */
    public function split(float $ratio = 0.8): array
    {
        if ($ratio <= 0.0 || $ratio >= 1.0) throw new InvalidArgumentException("Ratio must be between 0 and 1.");
        
        $n = (int) round($this->numRows * $ratio);
        
        // Zero-copy view creations
        $train = $this->slice(0, $n);
        $test = $this->slice($n, $this->numRows - $n);
        
        return [$train, $test];
    }

    /**
     * Generates K-Fold Cross Validation splits.
     * @return \Generator<array{0: self, 1: self}> Yields [TrainDataset, ValidationDataset]
     */
    public function fold(int $k = 10): \Generator
    {
        $foldSize = (int) floor($this->numRows / $k);
        
        for ($i = 0; $i < $k; $i++) {
            $offset = $i * $foldSize;
            $length = ($i === $k - 1) ? $this->numRows - $offset : $foldSize;
            
            // Validation block is a zero-copy slice
            $val = $this->slice($offset, $length);
            
            // Train block requires concatenating the pieces before and after the validation block
            $trainSamples = [];
            $trainLabels = [];
            
            if ($offset > 0) {
                $trainSamples[] = $this->samples->slice(0, 0, $offset);
                if ($this->labels) $trainLabels[] = $this->labels->slice(0, 0, $offset);
            }
            if ($offset + $length < $this->numRows) {
                $rem = $this->numRows - ($offset + $length);
                $trainSamples[] = $this->samples->slice(0, $offset + $length, $rem);
                if ($this->labels) $trainLabels[] = $this->labels->slice(0, $offset + $length, $rem);
            }
            
            $train = new self(
                Tensor::concat($trainSamples, 0),
                $this->labels ? Tensor::concat($trainLabels, 0) : null
            );
            
            yield [$train, $val];
        }
    }

    // ========================================================================
    // BATCHING & RANDOMIZATION
    // ========================================================================

    /**
     * Generates Zero-Copy mini-batches for Neural Network training.
     * @return \Generator<self>
     */
    public function batches(int $batchSize): \Generator
    {
        // Using the cached $this->numRows makes this loop exceptionally fast for JIT
        $total = $this->numRows;
        for ($start = 0; $start < $total; $start += $batchSize) {
            $length = min($batchSize, $total - $start);
            yield $this->slice($start, $length);
        }
    }

    /**
     * Randomizes the dataset order efficiently at the C-level.
     */
    public function randomize(): self
    {
        $indices = Tensor::randomUniform([$this->numRows], 0, 1)->argsort();
        $this->samples = $this->samples->take($indices, 0);
        if ($this->labels) {
            $this->labels = $this->labels->take($indices, 0);
        }
        return $this;
    }

    // ========================================================================
    // TRANSFORMATIONS & FILTERING
    // ========================================================================

    /**
     * Standardizes the features (Mean=0, Std=1) in-place.
     */
    public function standardize(): self
    {
        $this->samples->standardizeInplace();
        return $this;
    }

    /**
     * Applies a generic closure to the underlying Tensors.
     */
    public function apply(callable $fn): self
    {
        $fn($this->samples, $this->labels);
        return $this;
    }

    /**
     * Filters the dataset using a binary mask (Tensor of 1.0 and 0.0).
     */
    public function filterByMask(Tensor $mask): self
    {
        return new self(
            $this->samples->booleanIndex($mask), 
            $this->labels ? $this->labels->booleanIndex($mask) : null
        );
    }

    // ========================================================================
    // STACKING, MERGING, AND JOINING
    // ========================================================================

    /**
     * Vertical Concatenation: Stacks another dataset below this one.
     */
    public function stack(Dataset $other): self
    {
        if ($this->numColumns !== $other->numColumns()) {
            throw new InvalidArgumentException("Datasets must have the same number of feature columns to stack.");
        }

        $newSamples = Tensor::concat([$this->samples, $other->samples()], 0);
        $newLabels = null;
        
        if ($this->isLabeled() && $other->isLabeled()) {
            $newLabels = Tensor::concat([$this->labels, $other->labels()], 0);
        }

        return new self($newSamples, $newLabels);
    }

    /**
     * Horizontal Concatenation: Joins the feature columns of another dataset to this one.
     */
    public function join(Dataset $other): self
    {
        if ($this->numRows !== $other->numRows()) {
            throw new InvalidArgumentException("Datasets must have the same number of rows to join.");
        }

        $newSamples = Tensor::concat([$this->samples, $other->samples()], 1);
        return new self($newSamples, $this->labels); // Maintains the original labels
    }

    // ========================================================================
    // DESCRIPTIVE STATISTICS & SORTING
    // ========================================================================

    /**
     * Returns a summary of column-wise statistics rapidly computed in C.
     */
    public function describe(): array
    {
        return [
            'mean' => $this->samples->meanAxis(0)->toFlatArray(),
            'max'  => $this->samples->maxAxis(0)->toFlatArray(),
            'min'  => $this->samples->minAxis(0)->toFlatArray(),
            'sum'  => $this->samples->sumAxis(0)->toFlatArray(),
        ];
    }

    /**
     * Sorts the entire dataset based on the values in a specific feature column.
     */
    public function sortByColumn(int $column): self
    {
        if ($column < 0 || $column >= $this->numColumns) {
            throw new InvalidArgumentException("Column index out of bounds.");
        }

        $colData = $this->samples->col($column);
        $indices = $colData->argsort(); // Returns indices sorting ascending

        return new self(
            $this->samples->take($indices, 0),
            $this->labels ? $this->labels->take($indices, 0) : null
        );
    }

    // ========================================================================
    // EXPORTING
    // ========================================================================

    /**
     * Converts the C-Tensor memory back into a standard PHP associative array chunk.
     * WARNING: This is the ONLY method that pulls data out of C and into PHP's memory space.
     */
    public function toArray(): array
    {
        $rows = $this->numRows;
        $cols = $this->numColumns;
        
        // Fast binary dump from C pointer to PHP string, then unpacked
        $flatSamples = $this->samples->toFlatArray();
        $flatLabels = $this->labels ? $this->labels->toFlatArray() : [];
        
        // Optimize PHP array allocation
        $data = [];
        
        for ($i = 0; $i < $rows; $i++) {
            $row = array_slice($flatSamples, $i * $cols, $cols);
            if ($this->labels) {
                $row[] = $flatLabels[$i]; // Append label as the last item
            }
            $data[] = $row;
        }
        
        return $data;
    }

    /**
     * Exports the dataset to a CSV file.
     */
    public function toCSV(string $filepath): void
    {
        $fp = fopen($filepath, 'w');
        if (!$fp) throw new RuntimeException("Could not open file for writing: {$filepath}");

        $data = $this->toArray();
        foreach ($data as $row) {
            fputcsv($fp, $row);
        }
        
        fclose($fp);
    }
}