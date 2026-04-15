<?php

declare(strict_types=1);

namespace Pml;

use Pml\Lib\TensorEngine;

/**
 * Interleaved KV cache for autoregressive transformer inference.
 *
 * Stores K and V vectors interleaved as [cap][2*head_dim] in a single
 * 32-byte-aligned C buffer.  append() adds new token(s); attentionKV()
 * runs streaming Milakov attention against the full cache with zero PHP
 * allocations and no O(seq²) scores buffer.
 */
final class KVCache
{
    public ?\FFI\CData $ptr;

    public function __construct(int $cap, int $headDim)
    {
        $ffi       = TensorEngine::get();
        $this->ptr = $ffi->kvcache_create($cap, $headDim);
        if ($ffi->tensor_check_error()) {
            $msg = \FFI::string($ffi->tensor_get_last_error());
            $ffi->tensor_clear_error();
            throw new \RuntimeException($msg);
        }
        if ($this->ptr === null) {
            throw new \RuntimeException('kvcache_create returned NULL.');
        }
    }

    /** Append one or more new K,V token rows to the cache. */
    public function append(Tensor $k, Tensor $v): void
    {
        $ffi = TensorEngine::get();
        $ffi->kvcache_append($this->ptr, $k->ptr, $v->ptr);
        if ($ffi->tensor_check_error()) {
            $msg = \FFI::string($ffi->tensor_get_last_error());
            $ffi->tensor_clear_error();
            throw new \RuntimeException($msg);
        }
    }

    /** Number of tokens currently stored. */
    public function len(): int
    {
        return TensorEngine::get()->kvcache_len($this->ptr);
    }

    /** Reset cache to empty (keeps allocation). */
    public function reset(): void
    {
        TensorEngine::get()->kvcache_reset($this->ptr);
    }

    public function __destruct()
    {
        if ($this->ptr !== null) {
            TensorEngine::get()->kvcache_free($this->ptr);
            $this->ptr = null;
        }
    }
}
