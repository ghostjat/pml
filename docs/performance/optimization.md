---
layout: default
title: Performance Optimization
---

# Performance Optimization

This page describes the low-level optimizations that make the framework efficient on CPU-bound workloads.

## Core optimizations

- native C kernels for tensor math
- BLAS-backed matrix multiplication and vector operations
- OpenMP parallel loops for reduction and elementwise kernels
- fused operations to reduce boundary overhead

## Fused kernels

The C backend exposes fused primitives such as:

- `tensor_linear()` for `X @ W^T + bias`
- `tensor_add_relu()` for fused activation
- `tensor_fused_adam_step()` for optimizer updates

Fused kernels reduce memory traffic and FFI call count.

## BLAS and LAPACKE

- `tensor_matmul()` uses OpenBLAS when available.
- `tensor_matmul_ex()` supports transposed inputs without extra copies.
- Dense matrix operations are optimized for row-major layout.

## Hot path minimization

- Cache the `FFI` instance in PHP.
- Avoid repeated metadata lookup in inner loops.
- Use native reduction kernels like `tensor_sum_axis()` instead of PHP loops.

## When to use

- Use native fused kernels for training kernels and inference.
- Use BLAS-backed matmul for large matrix multiplication.

## When not to use

- Avoid elementwise operations in PHP if the same result can be computed with a bulk kernel.
- Avoid repeated conversions between contiguous and non-contiguous tensors.
