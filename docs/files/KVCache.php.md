# KVCache.php

## Purpose

The `KVCache.php` file contains the implementation of an interleaved Key-Value (KV) cache for use in autoregressive transformer inference. This cache is designed to store K and V vectors efficiently, allowing for optimized attention mechanisms without excessive memory allocation or computation.

## Key Components

### Classes, Functions, Methods with Signatures

1. **KVCache**
   - **Class**:
     ```php
     final class KVCache
     ```
     - A final class representing the KV cache.
   
2. **Constructor (`__construct`)**:
   ```php
   public function __construct(int $cap, int $headDim)
   ```
   - Initializes a new KV cache with a specified capacity (`$cap`) and head dimension (`$headDim`). It allocates memory using an FFI call to `kvcache_create`.

3. **append**:
   ```php
   public function append(Tensor $k, Tensor $v): void
   ```
   - Appends one or more new K,V token rows to the cache. Uses FFI to call `kvcache_append`.

4. **len**:
   ```php
   public function len(): int
   ```
   - Returns the number of tokens currently stored in the cache.

5. **reset**:
   ```php
   public function reset(): void
   ```
   - Resets the cache to an empty state without deallocating memory by calling `kvcache_reset`.

6. **Destructor (`__destruct`)**:
   ```php
   public function __destruct()
   ```
   - Frees the allocated memory for the KV cache when the object is destroyed.

### Important Variables and Constants

- None explicitly defined in the class. The class relies on FFI calls to manage internal state, which is encapsulated within the `kvcache_create`, `kvcache_append`, etc., functions.

## Inputs / Outputs

### For ML Components (Autoregressive Transformer)

- **Inputs**:
  - **K and V**: Instances of the `Tensor` class representing the Key and Value vectors to be stored in the cache.
  
- **Outputs**:
  - No explicit outputs; the methods modify the internal state of the KV cache.

### For Utility Files

- This is a utility file, but it does not take or return any parameters directly. It operates on an instance of `KVCache`.

## Dependencies

- **Internal Dependencies**:
  - The class relies on FFI (Foreign Function Interface) calls to a C library for managing the KV cache.

- **External Dependencies**:
  - The class depends on the `TensorEngine` class, which is assumed to provide access to FFI functions for tensor operations and error handling.

## Usage Notes

### Integration with the Framework

- This file should be integrated into the larger machine learning framework as a component of an autoregressive transformer model. It facilitates efficient management of the attention cache during inference.
  
- The `append` method is crucial for adding new tokens to the cache, and it must be called in sequence to maintain the correct order of tokens.

### Edge Cases

- If memory allocation fails during construction (`$ffi->kvcache_create` returns NULL), a runtime exception will be thrown.
  
- Calling methods on an uninitialized or already freed `KVCache` instance could lead to undefined behavior. Always ensure the cache is properly initialized and reset before use.

### Performance Considerations

- The KV cache is designed to minimize memory allocations and computations, making it efficient for real-time inference in transformer-based models.

- The `append` method does not perform O(seq²) operations on scores, which could be a bottleneck in traditional attention mechanisms. This makes the `KVCache` particularly useful for large sequence lengths.