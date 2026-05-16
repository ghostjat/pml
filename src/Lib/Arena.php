<?php

declare(strict_types=1);

namespace Pml\Lib;

use Pml\Tensor;

/**
 * PHP wrapper around the C TensorArena bump allocator.
 *
 * An Arena pre-allocates a slab of memory once and hands out sub-regions
 * sequentially (no individual free). arena_reset() rewinds the bump pointer
 * in O(1) — ideal for per-batch scratch tensors that are discarded after
 * each training step (§35).
 *
 * Usage:
 *   $arena = new Arena(64 * 1024 * 1024);  // 64 MiB slab
 *   foreach ($batches as $batch) {
 *       $tmp = $arena->tensor([N, H]);      // zero-copy allocation from slab
 *       // ... use $tmp ...
 *       $arena->reset();                    // O(1) rewind — no malloc/free per step
 *   }
 *   // $arena is freed when it goes out of scope (__destruct)
 */
final class Arena
{
    private \FFI\CData $ptr;  // TensorArena*

    public function __construct(int $capacityBytes = 32 * 1024 * 1024)
    {
        $ffi       = TensorEngine::get();
        $this->ptr = $ffi->arena_create($capacityBytes);
        if (\FFI::isNull($this->ptr)) {
            throw new \RuntimeException("Arena::__construct — arena_create() returned NULL (OOM?).");
        }
    }

    /**
     * Allocate a Tensor from the arena slab.
     * The Tensor is valid until the next reset() or the Arena is destroyed.
     *
     * @param int[]  $shape e.g. [batch, hidden]
     * @param int    $dtype Tensor::DTYPE_FLOAT32 (default)
     */
    public function tensor(array $shape, int $dtype = Tensor::DTYPE_FLOAT32): Tensor
    {
        $ffi    = TensorEngine::get();
        $ndim   = count($shape);
        $cShape = $ffi->new("int[$ndim]");
        foreach ($shape as $i => $s) {
            $cShape[$i] = $s;
        }
        $ptr = $ffi->tensor_create_arena($ndim, $ffi->cast('int*', $cShape), $dtype, $this->ptr);
        if (\FFI::isNull($ptr)) {
            throw new \RuntimeException("Arena::tensor — allocation failed (arena full or OOM).");
        }
        return Tensor::wrap($ptr);
    }

    /**
     * Rewind the arena bump pointer to the start — O(1), no individual frees.
     * All Tensors previously allocated from this arena are invalidated.
     */
    public function reset(): void
    {
        TensorEngine::get()->arena_reset($this->ptr);
    }

    /** Returns the underlying TensorArena* for direct C calls that accept an arena. */
    public function ptr(): \FFI\CData
    {
        return $this->ptr;
    }

    public function __destruct()
    {
        TensorEngine::get()->arena_destroy($this->ptr);
    }
}
