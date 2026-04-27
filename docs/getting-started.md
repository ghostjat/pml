# Getting Started

This guide covers the minimum steps to run the PML framework, load data, and train a model.

## Requirements

- PHP 8.1 or newer
- `php-ffi` enabled
- `gcc`
- `libopenblas-dev`
- `liblapacke-dev`
- `composer`

On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install php-cli php-ffi gcc libopenblas-dev liblapacke-dev composer
```

## Install the repository

```bash
cd /home/ghost/projects/php/lab/ffi
composer install
```

## Build the C backend

The C backend compiles automatically when `Pml\Lib\TensorEngine::get()` is called and `src/Lib/libtensor.so` is missing.

If you want to build it manually:

```bash
cd src/Lib
gcc -O3 -march=native -mtune=native -mfma -fno-math-errno -funsafe-math-optimizations \
    -fopenmp -funroll-loops -fomit-frame-pointer -D_GNU_SOURCE -shared -fPIC \
    -o libtensor.so.7 tensor.c dataset_io.c inference.c autograd.c graph.c \
    -lopenblas -llapacke -lm
ln -sf libtensor.so.7 libtensor.so
```

## Run your first script

Create `examples/hello.php`:

```php
<?php
require 'vendor/autoload.php';

use Pml\Tensor;

$t = Tensor::zeros(2, 3)->fill(1.5);
var_export($t->shape());
print_r($t->toFlatArray());
```

Run it:

```bash
php examples/hello.php
```

## Load a dataset from CSV

```php
<?php
use Pml\Dataset;

$dataset = Dataset::load('datasets/housing/train.csv');
print_r($dataset->schema());
```

## Train a model

```php
use Pml\Estimators\Regression\GBDTRegressor;

$dataset = Dataset::fromCSV('datasets/housing/train.csv', labelColumn: 0)
    ->dropNans()
    ->materialize(labelCol: 0);

$model = new GBDTRegressor();
$model->train($dataset);
```

## Save and load a model

```php
$model->save('saved_models/gbdt_regressor');
$loaded = \Pml\Estimators\Regression\GBDTRegressor::load('saved_models/gbdt_regressor');
```

## What to read next

- `architecture.md` for the system design and C/PHP boundary.
- `api/dataset.md` for dataset creation and ETL operations.
- `api/tensor.md` for tensor algebra and zero-copy operations.
