# Performance

This framework is built around C-level compute kernels and zero-copy data movement.

## Why it is fast

- **C computation**: All heavy math is implemented in `src/Lib/*.c` and exposed through `FFI`.
- **OpenBLAS / LAPACKE**: Linear algebra operations such as matrix multiplication and decompositions use optimized native libraries.
- **AVX2 and fused kernels**: The backend includes fused kernels for common deep learning patterns such as `linear`, `addRelu`, and `mulAdd`.
- **Zero-copy views**: Tensor slicing and batching create views instead of copying data when possible.
- **SafeTensors mmap**: Model weights can be loaded directly from disk without copying into the PHP heap.

## CPU optimizations

- **OpenMP**: The C runtime is compiled with `-fopenmp` and exposes `tensor_configure_threading()`.
- **BLAS threading**: `configureThreading()` lets you control BLAS and OpenMP independently.
- **Vectorized kernels**: Fused operations use AVX2 instructions for fewer memory passes.

## Zero-copy design

- `Tensor::slice()` and `Dataset::slice()` use C views that share the same underlying buffer.
- Views retain a `parent` reference to prevent the original memory from being freed.
- `SafeTensorsIO::load()` returns mmap-backed tensors that are not copied into the process memory.
- `Dataset::batches()` yields zero-copy tensor slices.

## Data ingestion performance

- `Dataset::fromCSV()` uses `tensor_dataset_from_csv()` for numeric CSVs, bypassing PHP array allocation.
- For mixed-type CSVs, `Dataset::load()` uses the ETL-mode C DataFrame to parse and transform data.
- `Tensor::fromArray()` packs nested PHP arrays into a binary string and copies them in a single FFI boundary crossing.

## Practical tips

- Call `Tensor::configureThreading()` once at startup to avoid oversubscription.
- Prefer `Dataset::randomize()` over manual PHP shuffling.
- Avoid `toArray()` or `toFlatArray()` in hot loops.
- Use `Tensor::copyFrom()` and `matmulInto()` to reuse pre-allocated buffers.

## Example

```php
use Pml\Tensor;

Tensor::configureThreading(8, 2);
$x = Tensor::randomNormal([1024, 1024]);
$y = Tensor::randomNormal([1024, 1024]);
$z = $x->matmul($y);
```
