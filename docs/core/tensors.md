---
layout: default
title: Core Tensor Engine
---

# Core Tensor Engine

The tensor subsystem is the foundation of runtime performance. It exposes a PHP API while delegating numeric workloads to a native C engine through FFI.

## Concept

Tensors are multi-dimensional arrays backed by a C `TensorC` struct. PHP objects are thin wrappers around opaque native pointers.

Key design decisions:

- tensor metadata is stored in C and exposed through PHP getters
- data buffers are allocated in C, not PHP
- zero-copy views retain parent references to avoid premature deallocation
- operations are implemented as native kernels, not PHP loops

## Internal Flow

1. `Tensor::wrap($ptr)` creates a PHP wrapper for a native `TensorC*`.
2. PHP methods call `TensorEngine::get()` and invoke FFI functions.
3. C kernels return new `TensorC*` pointers, which PHP wraps again.
4. `__destruct()` frees owned tensors unless the tensor is a zero-copy view.

### Data flow diagram

```text
PHP Tensor object
        │
        │ wrap/unwrap
        ▼
Native TensorC* pointer
        │
        │ FFI boundary
        ▼
 C kernel / BLAS call
```

## Memory semantics

- `Tensor::wrap($ptr, $parent)` sets `$parent` when the result is a view.
- Zero-copy views do not allocate a new data buffer.
- `Tensor::copy()` explicitly duplicates data when a contiguous independent tensor is required.
- `Tensor::contiguous()` is a low-cost check that avoids unnecessary copies.

### Copy vs zero-copy

- `slice()`, `row()`, `col()`, `view()` → zero-copy
- `copy()`, `reshape()` on non-contiguous storage → copy
- `Tensor::emptyLike()` / `zeros()` / `ones()` → allocate new buffer

## API exposure

### PHP usage

```php
use Pml\Tensor;

$t = Tensor::zeros(512, 128);
$t = $t->fill(1.0);
$row = $t->row(0);          // zero-copy view
$copy = $row->copy();       // allocate new buffer
```

### Corresponding C-level behavior

- `tensor_zeros()` allocates `TensorC` and fills data with zero.
- `tensor_row_view()` returns a `TensorC*` with shared data pointer.
- `tensor_copy()` allocates a new data buffer and performs `memcpy`.

## Performance implications

- Row-major layout favors dense matrix kernels and BLAS access patterns.
- Zero-copy views minimize pressure on the PHP GC and reduce heap churn.
- Contiguous buffers are essential for batched BLAS and OpenMP kernels.
- Non-contiguous views require explicit materialization before compute-intensive operations.

## When to use

- Use `Tensor::view()` and `slice()` for dataset indexing and feature extraction.
- Use `copy()` when the buffer must outlive the parent or when a contiguous layout is required.
- Avoid repeated slicing inside inner loops; materialise slices first if the same window is reused.

## When not to use

- Avoid `Tensor::view()` for long-lived buffers if the parent tensor is large and temporary.
- Do not rely on PHP GC to free C backing memory in time-critical loops; unset tensors explicitly.
