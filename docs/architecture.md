# Architecture

The framework is organized as a PHP orchestration layer backed by a native C tensor engine.

## High-level architecture

```
PHP Application
    ├── src/                  (PHP API layer)
    │     ├── Dataset.php
    │     ├── Tensor.php
    │     ├── Pipeline.php
    │     ├── Training/
    │     ├── Transformers/
    │     ├── Tokenizers/
    │     └── Estimators/
    └── src/Lib/              (FFI and persistence)
          ├── TensorEngine.php
          ├── ModelStore.php
          ├── SafeTensorsIO.php
          └── libtensor.so
```

## Core design principles

- **PHP orchestrates only**: no ML math is implemented in PHP.
- **C executes compute kernels**: tensor math, linear algebra, CSV ingestion, and neural network kernels are written in C and exposed through FFI.
- **Zero-copy whenever possible**: dataset slicing, batching, and SafeTensors loading avoid unnecessary copies.
- **Lazy ETL materialization**: the dataset can remain in ETL mode until tensor access is required.

## Dataset pipeline

1. `Dataset::load()` reads a CSV into a C DataFrame in ETL mode.
2. ETL methods such as `dropNans()`, `oneHotEncode()`, and `selectColumns()` operate on the C DataFrame.
3. `materialize()` converts the DataFrame into numeric tensors.
4. Most training and inference methods work in Tensor mode.

## Tensor engine layers

- `TensorEngine.php` exposes `libtensor.so` via PHP `FFI::cdef()`.
- The `Tensor` class is a PHP wrapper around native `TensorC*` pointers.
- `TensorEngine` declares:
  - tensor creation and views
  - arithmetic and reductions
  - matrix multiplication and BLAS-backed routines
  - convolution / im2col / attention kernels
  - dataset CSV ingestion and SafeTensors I/O

## Persistence and safety

- `ModelStore.php` serializes PHP object configuration and non-tensor state without using `serialize()`.
- `SafeTensorsIO.php` writes tensor bytes to disk in the SafeTensors format.
- `Pipeline::save()` and `Sequential::save()` produce:
  - `config.json` for PHP state
  - `*.safetensors` for tensor weights

## Performance optimization points

- **FFI caching**: both `TensorEngine` and `Tensor` cache the `FFI` instance.
- **Shape and dtype metadata**: tensor metadata is read without copying buffer contents.
- **Arena allocation**: optional arena support allows O(1) allocations for many tensors.
- **Threading control**: `Tensor::configureThreading()` manages OpenMP and BLAS thread pools.
- **Vectorized fused kernels**: matrix + bias, add+ReLU, FMA, and attention kernels are fused in C.

## Typical runtime flow

- PHP creates a `Dataset`.
- PHP constructs a model or pipeline.
- Training or inference calls traverse PHP objects.
- Numeric work happens in `Tensor` methods that call C kernels.
- Checkpoints are written by SafeTensors and JSON metadata.

## How C and PHP interact

- PHP passes pointers to `TensorC` structs into native functions.
- Many C methods return `TensorC*` pointers that are immediately wrapped by `Tensor::wrap()`.
- `Tensor` objects own the C pointer by default and free it in `__destruct()`.
- Zero-copy views retain a reference to the parent tensor to prevent premature free.
