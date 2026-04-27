---
layout: default
title: Core Memory Model
---

# Core Memory Model

Memory management is a first-class concern. The framework exposes explicit ownership boundaries, arena pooling, and zero-copy views to minimize allocations.

## Memory ownership

- `Tensor` owns its C backing buffer by default.
- Views retain a reference to their parent to keep data alive.
- `Dataset` owns a `DataFrame` pointer in ETL mode and frees it in `__destruct()`.
- Arena allocations are bulk-freed by `arena_destroy()`.

## Arena allocation

The C backend supports arena-backed tensors for high-throughput workloads.

- `tensor_create_arena()` allocates many tensors inside a single block.
- PHP destructors must not free arena tensors individually.
- Arena lifetime is controlled explicitly.

## Zero-copy constraints

Zero-copy is available for:

- views and slices
- label extraction in ETL mode
- SafeTensors memory-mapped weights

Constraints:

- parent tensors must outlive views
- non-contiguous layouts may require a copy before BLAS
- arena tensors cannot be freed by `tensor_free()` individually

## Layout and locality

The underlying layout is row-major.

- Row-major favors matrix multiplication and row-based batch operations.
- Column-wise slicing can produce non-contiguous views.
- Contiguous data is required for the fastest native kernels.

## Garbage collection and destructors

PHP GC works at the object layer. Native memory is freed when:

- `Tensor::__destruct()` calls `tensor_free()` for owned tensors
- `Dataset::__destruct()` calls `df_free()` in ETL mode
- `arenas` are destroyed as a unit

### Best practices

- unset temporary tensors inside hot loops
- avoid deep reference chains across large tensors
- use `copy()` when storing views beyond the parent scope

## Performance trade-offs

- Zero-copy views reduce allocation cost but increase lifetime coupling.
- Explicit arena pools reduce allocator overhead for repeated temporary tensors.
- Contiguous copies cost memory but improve compute throughput significantly.

## C-level primitives

The tensor memory model is built on:

- `safe_malloc`, `safe_memalign`
- `tensor_create_*`
- `tensor_free`
- `arena_create`, `arena_alloc`, `arena_destroy`

These primitives ensure data alignment for SIMD and BLAS.
