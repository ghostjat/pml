---
layout: default
title: Getting Started
---

# Getting Started

This page describes the environment setup, build path, and runtime validation for the framework.

## Requirements

- PHP 8.1 or newer
- `php-ffi` enabled
- `gcc`
- `libopenblas-dev`
- `liblapacke-dev`
- `composer`

### Debian/Ubuntu

```bash
sudo apt update
sudo apt install php-cli php-ffi gcc libopenblas-dev liblapacke-dev composer
```

## Install the repository

```bash
cd /home/ghost/projects/php/lab/ffi
composer install
```

## Build and verify the native backend

The native backend is built automatically when `Pml\Lib\TensorEngine::get()` is first called.

Manual build:

```bash
cd src/Lib
gcc -O3 -march=native -mtune=native -mfma -fno-math-errno -funsafe-math-optimizations \
    -fopenmp -funroll-loops -fomit-frame-pointer -D_GNU_SOURCE -shared -fPIC \
    -o libtensor.so.7 tensor.c dataset_io.c inference.c autograd.c graph.c \
    -lopenblas -llapacke -lm
ln -sf libtensor.so.7 libtensor.so
```

## Verify runtime

```bash
php -r 'require "vendor/autoload.php"; echo Pml\Lib\TensorEngine::get() ? "OK\n" : "FAIL\n";'
```

## Validate dataset ingestion

```php
<?php
require 'vendor/autoload.php';
use Pml\Dataset;

$dataset = Dataset::load('datasets/housing/train.csv');
var_dump($dataset->numRows());
```

## Validate tensor operations

```php
<?php
require 'vendor/autoload.php';
use Pml\Tensor;

$t = Tensor::zeros(4, 4);
$t->fill(1.0);
var_export($t->shape());
```

## Runtime checklist

- `TensorEngine::get()` loads `libtensor.so`
- `Dataset::fromCSV()` uses the numeric fast path when available
- `Dataset::load()` preserves ETL mode for mixed-type CSVs
- `Pipeline::save()` and `Pipeline::load()` persist metadata and tensor weights

## What to read next

- [Architecture](architecture.md)
- [Core: Dataset](core/dataset.md)
- [Core: FFI](core/ffi.md)
