# FFI API

This page explains the PHP/C binding layer and memory safety rules for the tensor engine.

## `Pml\Lib\TensorEngine`

`TensorEngine` exposes `libtensor.so` via PHP `FFI::cdef()`.

### `get()`

```php
public static function get(): \FFI
```

- Lazily builds `libtensor.so` if it does not exist.
- Returns a cached `FFI` instance.
- Declares all native functions used by the framework.

## Native C functions exposed

The following groups are available through `TensorEngine`:

- Tensor creation and destruction
- Arithmetic and comparison kernels
- Linear algebra, BLAS, and matrix operations
- Convolution, im2col, and attention operations
- CSV dataset ingestion
- SafeTensors file save/load
- Threading configuration
- KV cache and inference primitives

## Error handling

All native calls should be followed by `tensor_check_error()`.

`TensorEngine::get()` exposes:

```c
bool tensor_check_error(void);
const char* tensor_get_last_error(void);
void tensor_clear_error(void);
```

The PHP wrappers translate C errors into `RuntimeException`.

## Memory safety

### Ownership semantics

- `Tensor` objects own their `TensorC*` pointer by default.
- `Tensor::wrap()` can accept a `parent` tensor to preserve shared ownership.
- When a `Tensor` is a view, the `parent` reference prevents the original storage from being freed.
- Native arena allocations are supported; arena-backed tensors are not freed individually.

### SafeTensors and mmap

- `Pml\Lib\SafeTensorsIO` writes tensor bytes with zero PHP heap copies.
- `Tensor::fromMmap()` maps a file region into memory and returns a tensor backed by the OS page cache.
- `mmapFree()` must be called to release the mapping when the tensor is no longer needed.

### CSV ingestion

- `tensor_dataset_from_csv()` is the fastest path for numeric-only CSV files.
- Mixed-type CSVs fall back to ETL mode via `df_read_csv()`.
- The C dataset loader uses memory-efficient parsing and avoids loading entire files into PHP arrays.

## Threading

Control CPU parallelism using:

```php
Tensor::configureThreading(int $ompThreads, int $blasThreads = 1): void
```

- `ompThreads` controls OpenMP kernels.
- `blasThreads` controls the underlying BLAS thread pool.

## Example Usage

```php
use Pml\Tensor;

Tensor::configureThreading(4, 1);
$t = Tensor::randomNormal([1024, 1024]);
$u = $t->transpose()->matmul($t);
```

## Common mistakes

- Calling native C functions without checking `tensor_check_error()`.
- Releasing mmap-backed tensors by relying solely on PHP GC instead of `mmapFree()`.
- Assuming `Tensor::wrap()` always transfers ownership; shared parent references may keep memory alive.
