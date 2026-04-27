---
layout: default
title: Core Dataset Engine
---

# Core Dataset Engine

The dataset subsystem separates ETL from tensor execution. It keeps CSV ingestion and feature pipeline logic in C until numeric tensors are required.

## ETL mode vs Tensor mode

### ETL mode

- Represents data as a C `DataFrame` pointer.
- Supports mixed types and non-numeric columns.
- Avoids PHP arrays during CSV ingestion.
- Operations like `dropNans()`, `oneHotEncode()`, and `selectColumns()` are executed in C.

### Tensor mode

- Represents features as `Tensor` objects.
- Uses float32 tensors for samples and optional labels.
- Required for training and inference.

## Lifecycle

```text
CSV file
   ├─ tensor_dataset_from_csv() → direct Tensor mode (numeric-only)
   └─ df_read_csv() → ETL DataFrame
                    ├─ ETL transforms
                    └─ df_to_tensor() → Tensor mode
```

### Lazy materialization

`Dataset::load()` creates an ETL dataset without immediate tensor conversion. `materialize()` triggers the conversion on demand.

### Fast path vs fallback

- `Dataset::fromCSV()` uses `tensor_dataset_from_csv()` for pure numeric CSVs.
- If non-numeric values are present, it falls back to `df_read_csv()` and `materialize()`.

## Data flow and memory

`Dataset::fromCSV()` returns either:

- a fully materialized `Dataset` in Tensor mode
- or a mixed-type `Dataset` in ETL mode that can still be transformed

In both cases, the dataset owns only the C pointer or tensor wrappers.

## Key methods

### `Dataset::load(string $filepath, bool $hasHeader = true): self`

- Loads a CSV into ETL mode.
- Does not allocate full datasets in PHP.
- Use for datasets with categorical or text columns.

### `Dataset::fromCSV(string $filepath, int $labelColumn = -1, bool $hasHeader = true): self`

- Fast numeric ingestion for float-only CSVs.
- If mixed columns are detected, fall back to ETL mode and immediate materialization.
- `labelColumn` identifies the target column.

### `Dataset::materialize(int $labelCol = -1): self`

- Converts the C DataFrame into tensors.
- Frees the original DataFrame pointer.
- Returns a new `Dataset` in Tensor mode.

## Code example

```php
use Pml\Dataset;

$dataset = Dataset::load('datasets/housing.csv')
    ->withLabelColumn(0)
    ->dropNans()
    ->oneHotEncode(2)
    ->materialize(labelCol: 0);
```

## C-level behavior

- `df_read_csv()` parses the CSV and creates a columnar `DataFrame`.
- `df_to_tensor()` converts selected columns to `TensorC*` buffers.
- `df_free()` releases the ETL pointer once materialization is complete.

## Performance implications

- ETL mode delays numeric allocation until necessary.
- Mixed-type CSVs incur an ETL-to-tensor conversion cost.
- Numeric-only CSVs bypass this cost entirely with `tensor_dataset_from_csv()`.
- Use `dropNans()` before `materialize()` to keep the tensor memory footprint minimal.

## When to use

- Use ETL mode for ingestion pipelines with text features or categorical columns.
- Use Tensor mode for training, validation, and inference.

## When not to use

- Avoid calling `materialize()` too early if additional ETL transformations remain.
- Avoid repeated conversion of the same dataset in multiple passes.
