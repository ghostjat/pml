# Dataset API

`Pml\Dataset` is the entry point for data ingestion, ETL, and tensor-backed training data.

## Overview

`Dataset` operates in two modes:

- **ETL mode**: the data lives in a native C DataFrame. Use this for mixed-type CSV input and feature engineering.
- **Tensor mode**: the data lives in numeric `Pml\Tensor` objects and is ready for training.

Use `Dataset::load()` or `Dataset::fromCSV()` to create a dataset from disk.

## API Signature

```php
final class Dataset
```

### Static factory methods

```php
public static function load(string $filepath, bool $hasHeader = true): self
public static function fromCSV(string $filepath, int $labelColumn = -1, bool $hasHeader = true): self
public static function fromArray(array $samples, ?array $labels = null): self
```

### Mode and introspection

```php
public function isEtlMode(): bool
public function isLabeled(): bool
public function rawDfPtr(): ?\FFI\CData
public function schema(): array
public function categories(int $colIdx): array
public function numRows(): int
public function numColumns(): int
public function columnIndex(string $name): int
public function isTextColumn($column): bool
```

### ETL operations (ETL mode only)

```php
public function withLabelColumn(int $col): self
public function extractLabelTensor(): ?Tensor
public function dropNans(): self
public function sliceRowsEtl(int $offset, int $n): self
public function headRows(int $n): self
public function oneHotEncode(int $colIdx): self
public function selectColumns(array $colIndices): self
public function materialize(?int $labelCol = null): self
```

### Tensor mode operations

```php
public function samples(): Tensor
public function labels(): ?Tensor
public function select(array $columns): self
public function drop(array $columns): self
public function head(int $n = 10): self
public function tail(int $n = 10): self
public function slice(int $offset, int $length): self
public function take(int $n): self
public function leave(int $n): self
public function split(float $ratio = 0.8): array
public function fold(int $k = 10): \Generator
public function batches(int $batchSize): \Generator
public function randomize(): self
public function standardize(): self
public function apply(callable $fn): self
public function filterByMask(Tensor $mask): self
public function stack(Dataset $other): self
public function join(Dataset $other): self
public function describe(): array
public function sortByColumn(int $column): self
public function toArray(): array
public function toCSV(string $filepath): void
public function bagOfWords($column, ?int $maxFeatures = null): self
```

## What it does

`Dataset` reads CSV data into a C-backed DataFrame for ETL. Once you call `materialize()`, it converts the data into two `Tensor` objects:

- `samples` — feature matrix `[N×D]`
- `labels` — target vector `[N]`

The ETL pipeline is lazy: `Dataset::load()` does not allocate tensors until necessary.

## When to use it

- Use ETL mode when the CSV contains strings, categories, or missing values.
- Use tensor mode for model training, batching, and numeric transforms.
- Use `Dataset::fromCSV()` for fast numeric-only CSV ingestion.

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `filepath` | `string` | Path to the CSV file.
| `labelColumn` | `int` | Zero-based label column index, or `-1` if there is no label.
| `hasHeader` | `bool` | Whether the CSV has a header row.
| `samples` | `array` | Nested PHP array of feature rows.
| `labels` | `array|null` | PHP array of labels.
| `colIdx` | `int` | Column index to modify or encode.
| `colIndices` | `array` | Column indices to keep.
| `ratio` | `float` | Train split ratio.
| `batchSize` | `int` | Batch size for `batches()`.

## Return values

- ETL methods return a new `Dataset` object in ETL mode.
- Tensor methods return new `Dataset` objects or views that share underlying C memory where possible.
- `toArray()` returns a PHP array and performs the only full C → PHP copy.

## Example Usage

```php
use Pml\Dataset;

$dataset = Dataset::load('datasets/housing/train.csv', true)
    ->dropNans()
    ->oneHotEncode(2)
    ->materialize(labelCol: 0);

$samples = $dataset->samples();
$labels = $dataset->labels();

[$train, $test] = $dataset->split(0.75);

foreach ($train->batches(32) as $batch) {
    // Each batch is zero-copy.
    $x = $batch->samples();
    $y = $batch->labels();
}
```

## Performance notes

- `Dataset::materialize()` converts the C DataFrame to tensors once and then frees the DataFrame pointer.
- `head()`, `tail()`, `slice()`, and `batches()` are zero-copy views and do not allocate new tensor data.
- `Dataset::randomize()` uses C-level argsort on a random uniform tensor for shuffling.
- `toArray()` is the only bulk export that copies data into PHP arrays.

## Common mistakes

- Calling `dropNans()` after `materialize()` will fail because it requires ETL mode.
- Expecting `selectColumns()` to rename label indexes automatically; always verify `labelCol` after column selection.
- Assuming `Dataset::fromCSV()` always uses the fast numeric path; it falls back to ETL mode for mixed-type CSVs.
- Using `toArray()` in the training loop can cause a large performance penalty.
