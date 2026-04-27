---
layout: default
title: PML AutoML Framework
---

# PML AutoML Framework

A high-performance machine learning framework in PHP with a native C FFI backend. Designed for speed, low memory use, and production-grade model accuracy.

## Overview

PML combines PHP orchestration with a native C tensor engine to deliver efficient data pipelines, fast training, and scalable inference. The framework is built for developers who need low-level control without sacrificing performance.

## Features

- Zero-copy tensor operations via PHP FFI
- Native C backend for fast matrix math and data loading
- CSV dataset ETL with minimal PHP memory overhead
- Built-in model training, saving, and inference
- Developer-focused API and production-ready architecture

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ghostjat/pml.git
   cd pml
   ```

2. Install PHP dependencies:

   ```bash
   composer install
   ```

3. Build the native backend when needed:

   ```bash
   cd src/Lib
   gcc -O3 -march=native -mtune=native -mfma -fno-math-errno -funsafe-math-optimizations \
       -fopenmp -funroll-loops -fomit-frame-pointer -D_GNU_SOURCE -shared -fPIC \
       -o libtensor.so.7 tensor.c dataset_io.c inference.c autograd.c graph.c \
       -lopenblas -llapacke -lm
   ln -sf libtensor.so.7 libtensor.so
   ```

## Quick Start

1. Install dependencies and prepare the project.
2. Load a dataset from CSV.
3. Train an estimator and save the model.

Example:

```php
require 'vendor/autoload.php';
use Pml\Dataset;

$dataset = Dataset::fromCSV('datasets/housing/train.csv', labelColumn: 0)
    ->dropNans()
    ->materialize(labelCol: 0);

$model = new Pml\Estimators\Regression\GBDTRegressor();
$model->train($dataset);
$model->save('saved_models/gbdt_regressor');
```

## Documentation Links

- [Getting Started](getting-started.md)
- [Architecture](architecture.md)
- [API Reference](api.md)
- [Examples](examples/classification.md)
- [Advanced Topics](advanced/performance.md)
