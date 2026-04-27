---
layout: default
title: Basic Example
---

# Basic Example

This example shows the data path from CSV ingestion to tensor training.

## Objective

Demonstrate the numeric ingestion path with minimal assumptions about input schema.

## Data flow

```text
CSV file
   ├─ numeric-only fast path -> Tensor mode
   └─ mixed-type fallback -> ETL mode -> materialize -> Tensor mode
```

## Example

```php
use Pml\Dataset;
use Pml\Estimators\Regression\GBDTRegressor;

$dataset = Dataset::fromCSV('datasets/housing/train.csv', labelColumn: 0);
$model = new GBDTRegressor();
$model->train($dataset);
```

## Internals

- `Dataset::fromCSV()` attempts `tensor_dataset_from_csv()` first.
- If the CSV contains mixed types, the framework falls back to `df_read_csv()`.
- `materialize()` converts the `DataFrame` to native tensors once.

## Performance notes

- Numeric-only ingestion avoids ETL overhead.
- If the CSV contains categorical columns, prefer ETL transformations before `materialize()`.
