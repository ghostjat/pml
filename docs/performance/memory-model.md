---
layout: default
title: Memory Model
---

# Memory Model

The framework's memory model is explicit: the C engine owns tensor buffers and PHP objects manage lifetime.

## Allocation paths

- `Tensor::zeros()` / `Tensor::ones()` allocate new buffers.
- `Tensor::emptyLike()` allocates uninitialized memory.
- `tensor_dataset_from_csv()` can create direct tensor buffers from CSV.
- `arena_create()` allocates a bulk memory pool.

## Data layout

- Row-major layout is used for all tensors.
- `stride` metadata tracks contiguous and non-contiguous views.
- Non-contiguous views are valid but may incur copy costs in BLAS.

## Zero-copy and views

- `tensor_view()` and `tensor_slice()` produce views with shared data.
- Parent references keep the backing memory alive.
- A view becomes invalid if the parent is freed.

## Page and cache behavior

- Contiguous tensors reduce page crossing.
- Large tensors should be accessed in sequential order.
- Strided access patterns on slices will reduce throughput.

## Best practices

- Keep the training dataset contiguous before matrix multiplication.
- Use arena allocation for repeated temporary tensors.
- Use `Tensor::copy()` when a non-contiguous view must be repeatedly reused.
