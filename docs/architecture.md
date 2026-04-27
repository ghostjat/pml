---
layout: default
title: Architecture
---

# Architecture

The framework is built as a PHP orchestration layer over a native C tensor engine. The design separates control flow from numeric computation.

## High-level architecture

```text
PHP layer                       Native C layer
-----------                     -------------
src/                            src/Lib/
  Dataset.php                     TensorEngine.php
  Tensor.php                      libtensor.so
  Pipeline.php                    SafeTensorsIO.php
  Transformers/                   tensor math kernels
  Estimators/                     dataset ingestion
                                 arena and memory primitives
```

## Design principles

- PHP handles control flow, configuration, and persistence.
- C implements numeric kernels, tensor memory, and dataset ingestion.
- Zero-copy views minimize data movement.
- ETL and tensor execution are kept separate.

## Dataset pipeline

```text
CSV file
   ├─ numeric-only fast path -> Tensor mode
   └─ mixed-type fallback -> ETL mode -> transforms -> materialize -> Tensor mode
```

### Pipeline steps

1. `Dataset::load()` creates an ETL `DataFrame`.
2. ETL transforms execute in native C.
3. `Dataset::materialize()` converts the `DataFrame` to tensors.
4. Estimators consume tensor data for training and inference.

## Tensor engine

- `TensorEngine.php` loads or builds `libtensor.so`.
- `FFI::cdef()` defines the native API once.
- `Tensor` wraps `TensorC*` pointers and exposes a PHP API.

## Persistence

- `ModelStore` serializes object state without PHP `serialize()`.
- `SafeTensorsIO` writes tensor weights to disk.
- `Pipeline::save()` stores metadata and weights separately.

## Performance optimization points

- `TensorEngine::get()` caches the FFI interface.
- `Tensor` caches shape and dtype metadata.
- Arena allocation reduces allocator overhead.
- Fused kernels reduce FFI boundary crossings.

## Runtime flow

- PHP constructs dataset and model objects.
- Training and inference traverse PHP objects.
- Numeric compute runs in native C.
- Model state is persisted with SafeTensors and JSON metadata.

## Cross references

- [Core: Dataset](core/dataset.md)
- [Core: Tensors](core/tensors.md)
- [Core: FFI](core/ffi.md)
- [Estimators Overview](estimators/overview.md)
