# Memory

This page explains how the framework manages memory across PHP and C.

## C-backed tensors

`Pml\Tensor` objects wrap native `TensorC*` pointers. The pointer owns the tensor data unless:

- the tensor is a view
- the tensor is arena-backed
- the tensor is mmap-backed

## Ownership rules

- `Tensor::__destruct()` frees the native pointer when `owned` is true.
- View tensors keep a `$parent` reference to the original tensor.
- Arena-backed tensors are created with an arena pointer and are not freed individually.

## Arena allocation

The native API exposes:

- `arena_create(size_t capacity)`
- `arena_alloc(TensorArena* arena, size_t size)`
- `arena_reset(TensorArena* arena)`
- `arena_destroy(TensorArena* arena)`

This allows bulk allocation of many tensors with a single deallocation.

## Mmap-backed tensors

- `Tensor::fromMmap()` creates a tensor whose data points directly into a memory-mapped file region.
- `Tensor::mmapFree()` releases the mapping explicitly.
- Mmap-backed tensors are not automatically freed by the normal destructor semantics.

## SafeTensors I/O

- `Pml\Lib\SafeTensorsIO::save()` writes tensor bytes in the SafeTensors format.
- `SafeTensorsIO::load()` maps tensor regions from disk and returns zero-copy tensors.

## Mixed-mode dataset memory

- ETL mode uses a C DataFrame pointer stored in `Dataset::$dfPtr`.
- After `materialize()`, the DataFrame is freed and the dataset transitions fully to tensor mode.
- `Dataset::toArray()` converts tensor data to PHP arrays and should be used sparingly.

## Common memory mistakes

- Holding references to many intermediate datasets or tensors during pipeline construction.
- Calling `toArray()` repeatedly on large tensors.
- Using `Tensor::fromMmap()` without `mmapFree()`.
- Expecting `Tensor::copy()` to behave like a deep PHP clone; it allocates native tensor memory.
