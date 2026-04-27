---
layout: default
title: API Reference
---

# API Reference

This page summarizes the public classes and methods used to build production workloads.

## Public subsystems

- `Pml\Tensor`
- `Pml\Dataset`
- `Pml\Pipeline`
- `Pml\Lib\TensorEngine`
- `Pml\Lib\SafeTensorsIO`
- `Pml\Estimators` families
- `Pml\Transformers`

## Tensor API

### `Tensor::zeros(int ...$shape): self`
Creates a new float32 tensor with the requested shape.

### `Tensor::wrap(?\FFI\CData $ptr, ?self $parent = null): self`
Wraps a native `TensorC*` pointer into a PHP object. Use when a C kernel returns a tensor pointer.

### `Tensor::view(): self`
Creates a zero-copy view. The returned tensor retains the parent reference.

### `Tensor::copy(): self`
Performs an explicit copy of the underlying data buffer.

### `Tensor::contiguous(): self`
Returns a contiguous tensor, copying only when necessary.

## Dataset API

### `Dataset::fromCSV(string $filepath, int $labelColumn = -1, bool $hasHeader = true): self`
Loads a CSV using the fast numeric path when possible. Fallbacks to ETL mode for mixed types.

### `Dataset::load(string $filepath, bool $hasHeader = true): self`
Loads a CSV into native ETL mode without immediate tensor conversion.

### `Dataset::materialize(int $labelCol = -1): self`
Converts a `DataFrame` to tensors and frees the source ETL pointer.

## Pipeline API

### `Pipeline::__construct(array $transformers, Learner $estimator)`
Builds a pipeline that fits transformers before estimator training.

### `Pipeline::train(Dataset $dataset, mixed ...$args): void`
Trains the pipeline end-to-end.

### `Pipeline::predict(Dataset $dataset): Tensor`
Runs inference with the fitted preprocessing path.

### `Pipeline::save(string $dir): void`
Persists pipeline metadata and tensor weights in SafeTensors format.

## FFI and persistence

### `TensorEngine::get(): \FFI`
Returns the cached FFI interface and ensures `libtensor.so` is loaded.

### `SafeTensorsIO::save(string $path, array $tensorDict): void`
Writes tensor weights to disk in a safe binary layout.

### `SafeTensorsIO::load(string $path): array`
Loads tensor weights from a SafeTensors file.

## Recommended references

- [Core: FFI](core/ffi.md)
- [Core: Tensor Engine](core/tensors.md)
- [Core: Dataset](core/dataset.md)
- [Estimators Overview](estimators/overview.md)

## Edge cases

- `Tensor::wrap()` must not be called with a NULL native pointer.
- `Dataset::materialize()` is a one-way transition from ETL mode.
- `Pipeline::predict()` requires the pipeline to be trained.
