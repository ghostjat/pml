---
layout: default
title: Core FFI Bridge
---

# Core FFI Bridge

FFI is the bridge between PHP orchestration and native compute. The engine is designed to minimize boundary overhead while preserving safety.

## Engine initialization

`TensorEngine::get()` is the central entry point.

- It lazy-loads `libtensor.so`.
- It compiles the C sources automatically if the shared library is missing.
- It caches the `FFI` instance for repeated use.

## JIT boundary minimization

- The cached `FFI` instance avoids repeated `cdef()` operations.
- PHP method wrappers call native kernels with as few arguments as possible.
- Results are returned as `FFI\CData` pointers and wrapped immediately.

## Error handling

- Native functions do not throw exceptions directly.
- `Tensor::checkError()` queries `tensor_check_error()` and reads `tensor_get_last_error()`.
- The C engine clears the error state after each check.

## Pointer ownership

- `Tensor::wrap()` asserts non-null pointers.
- `Tensor` objects own the pointer unless a parent is provided.
- Parent pointers keep child views valid.

## Example

```php
$ffi = Pml\Lib\TensorEngine::get();
$ptr = $ffi->tensor_zeros(2, $ffi->new('int[2]')); // sample API shape
// wrap into a PHP tensor
$tensor = Pml\Tensor::wrap($ptr);
```

## C definitions exposed

The FFI header includes:

- `TensorC` memory layout
- tensor creation, copy, and view primitives
- BLAS-style matrix multiplication and tensor math
- dataset ingestion and SafeTensors I/O
- arena allocation primitives

## Performance notes

- Keep the number of FFI calls low for hot loops.
- Use composite native kernels such as `tensor_linear()` and `tensor_matmul_ex()`.
- Avoid repeated `shape()` or `dtype()` calls when the metadata is already available.

## When to use

- Use the FFI bridge for all numeric kernels and dataset ingestion.
- Do not implement numeric logic in PHP.
- Use PHP only for control flow, configuration, and persistence.
