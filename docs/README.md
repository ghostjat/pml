# PML AI/ML Framework Documentation

A high-performance PHP AI/ML framework built around a custom C-backed tensor engine and zero-copy data movement.

## What this repository is

This project is a full-stack machine learning library written in PHP with a native C backend for tensor math, data loading, feature engineering, and model persistence.

Key components:

- `src/Lib/TensorEngine.php` — PHP FFI bridge to `libtensor.so`, which implements fast tensor operations, matrix algebra, CSV ingestion, and neural network primitives.
- `src/Dataset.php` — high-level dataset abstraction with ETL mode for mixed-type CSVs and Tensor mode for numeric training data.
- `src/NeuralNetwork/Sequential.php` — deep learning container with layer stacking, training, validation, and checkpointing.
- `src/Estimators/` — classical ML models, including GBDT, decision trees, regression, clustering, and anomaly detection.
- `src/Lib/ModelStore.php` and `src/Lib/SafeTensorsIO.php` — zero-serialize persistence for model state and tensor weights.

## Key features

- C-backed tensor algebra via PHP FFI.
- Mixed-type CSV ETL with lazy conversion to tensor mode.
- Zero-copy batching, slicing, and view semantics.
- SafeTensors-compatible model checkpointing and mmap-backed tensor loading.
- Neural network training loop with callbacks, early stopping, and optimizer support.
- Pure PHP data transformations plus C-accelerated matrix operations.

## Why this framework is unique

- **PHP-first API**: Designed for PHP applications that need ML without leaving the PHP runtime.
- **FFI-based performance**: Core math runs in C, while PHP orchestrates data flow and model logic.
- **Zero-copy design**: Most dataset and tensor operations avoid extra memory allocation.
- **Safe persistence**: Models are saved without PHP `serialize()`, which prevents unsafe FFI state from leaking into disk artifacts.

## Installation

```bash
cd /home/ghost/projects/php/lab/ffi
composer install
```

The first call to `Pml\Lib\TensorEngine::get()` will compile `libtensor.so` automatically if it is missing.

### Required system dependencies

- PHP 8.1+ with FFI enabled
- `gcc`
- `libopenblas` and `liblapacke`

## Quick start example

```php
<?php
require 'vendor/autoload.php';

use Pml\Dataset;
use Pml\Estimators\Regression\GBDTRegressor;

// Load CSV into ETL mode and prepare numeric features.
$dataset = Dataset::fromCSV('datasets/housing/train.csv', labelColumn: 0, hasHeader: true)
    ->dropNans()
    ->materialize(labelCol: 0);

$model = new GBDTRegressor(nEstimators: 50, maxDepth: 4, numBins: 128);
$model->train($dataset);

$predictions = $model->predict($dataset);
print_r($predictions->toFlatArray());
```

## Minimal training example

```php
<?php
require 'vendor/autoload.php';

use Pml\Dataset;
use Pml\Estimators\Regression\GBDTRegressor;

$dataset = Dataset::fromArray([
    [0.0, 1.0],
    [1.0, 3.0],
    [2.0, 5.0],
], [0.0, 1.0, 2.0]);

$model = new GBDTRegressor();
$model->train($dataset);
$preds = $model->predict($dataset);
var_export($preds->toFlatArray());
```

## Documentation structure

- `getting-started.md` — environment, setup, first dataset and model.
- `architecture.md` — how the PHP/C layers interact.
- `api/` — reference docs for each major component.
- `examples/` — runnable scenarios for classification, regression, email automation, custom pipelines.
- `advanced/` — performance tuning, memory management, and internals.
