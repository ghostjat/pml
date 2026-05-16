<?php

declare(strict_types=1);

namespace Pml;

use Pml\Lib\TensorEngine;

/**
 * Multi-head KV cache for autoregressive transformer inference.
 *
 * Wraps the C MultiKVCache struct: a flat array of nLayers × nHeads individual
 * per-head KVCaches, each pre-allocated to maxSeqLen tokens.  All three
 * operations (prefill, append, attend) cross the FFI boundary exactly once
 * per layer per token — no PHP-level per-head loop.
 *
 * Workflow:
 *   $kv = new KVCache(nLayers: 32, nHeads: 32, maxSeqLen: 2048, headDim: 128);
 *
 *   // Prefill (one call per attention layer during prompt processing):
 *   $kv->prefill(layerIdx: $i, k: $Kr, v: $Vr);   // K,V [nH, T, hd]
 *
 *   // Decode loop (one new token per step):
 *   $kv->append(layerIdx: $i, k: $Kr, v: $Vr);    // K,V [nH, hd]
 *   $out = $kv->attend(layerIdx: $i, q: $Qr);      // Q [nH,1,hd] → [nH,1,hd]
 *
 *   // Reset between requests — O(nLayers×nHeads), no alloc/free:
 *   $kv->reset();
 *
 * Memory: nLayers × nHeads × maxSeqLen × 2 × headDim × 4 bytes.
 *   e.g. 32 × 32 × 2048 × 2 × 128 × 4 ≈ 1 GB for a 32-layer MHA model.
 *
 * For GQA models (nKvHeads < nHeads), pass nKvHeads as $nHeads and expand
 * the repeat in the attention layer before calling attend().
 */
final class KVCache
{
    /** @var \FFI\CData  MultiKVCache* */
    private \FFI\CData $ptr;

    private int $nLayers;
    private int $nHeads;
    private int $maxSeqLen;
    private int $headDim;

    /**
     * @param int $nLayers   Number of transformer layers
     * @param int $nHeads    Number of KV heads (nKvHeads for GQA, nHeads for MHA)
     * @param int $maxSeqLen Maximum tokens (prompt + generated combined)
     * @param int $headDim   Dimension of each head vector (d_model / nHeads)
     */
    public function __construct(int $nLayers, int $nHeads, int $maxSeqLen, int $headDim)
    {
        $ffi       = TensorEngine::get();
        $this->ptr = $ffi->mkvca_create($nLayers, $nHeads, $maxSeqLen, $headDim);
        if (\FFI::isNull($this->ptr)) {
            $msg = $ffi->tensor_check_error()
                ? \FFI::string($ffi->tensor_get_last_error())
                : 'OOM or invalid dimensions';
            $ffi->tensor_clear_error();
            throw new \RuntimeException("KVCache: mkvca_create failed — {$msg}");
        }
        $this->nLayers   = $nLayers;
        $this->nHeads    = $nHeads;
        $this->maxSeqLen = $maxSeqLen;
        $this->headDim   = $headDim;
    }

    /**
     * Populate the cache for a layer from full prompt K/V (prefill phase).
     * Runs in O(nHeads × T) — one call replaces nH × T individual appends.
     *
     * @param Tensor $k  [nH, T, hd] — key projections for all prompt tokens
     * @param Tensor $v  [nH, T, hd] — value projections for all prompt tokens
     */
    public function prefill(int $layerIdx, Tensor $k, Tensor $v): void
    {
        $ffi = TensorEngine::get();
        $ffi->mkvca_prefill($this->ptr, $layerIdx, $k->ptr, $v->ptr);
        $this->checkError('prefill');
    }

    /**
     * Append one decode-step token's K/V for a layer.
     * 1 FFI crossing regardless of nHeads.
     *
     * @param Tensor $k  [nH, hd] — key projections for the new token
     * @param Tensor $v  [nH, hd] — value projections for the new token
     */
    public function append(int $layerIdx, Tensor $k, Tensor $v): void
    {
        $ffi = TensorEngine::get();
        $ffi->mkvca_append($this->ptr, $layerIdx, $k->ptr, $v->ptr);
        $this->checkError('append');
    }

    /**
     * Run Milakov online-softmax attention against the filled cache.
     * OpenMP-parallel over heads inside C — 1 FFI crossing per layer per token.
     *
     * @param  Tensor $q  [nH, 1, hd] — query for the current decode token
     * @return Tensor     [nH, 1, hd] — attention output (new Tensor allocation)
     */
    public function attend(int $layerIdx, Tensor $q): Tensor
    {
        $ffi = TensorEngine::get();
        $ptr = $ffi->mkvca_attend($this->ptr, $layerIdx, $q->ptr);
        $this->checkError('attend');
        if (\FFI::isNull($ptr)) {
            throw new \RuntimeException('KVCache::attend — mkvca_attend returned NULL');
        }
        return Tensor::wrap($ptr);
    }

    /**
     * Reset all per-head caches to len=0.
     * O(nLayers × nHeads) — no realloc.
     * Call before each new conversation turn or request.
     */
    public function reset(): void
    {
        TensorEngine::get()->mkvca_reset($this->ptr);
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    public function nLayers(): int   { return $this->nLayers;   }
    public function nHeads(): int    { return $this->nHeads;     }
    public function maxSeqLen(): int { return $this->maxSeqLen;  }
    public function headDim(): int   { return $this->headDim;    }

    /**
     * Peak memory usage in bytes.
     * nLayers × nHeads × maxSeqLen × 2 × headDim × sizeof(float).
     */
    public function memoryBytes(): int
    {
        return $this->nLayers * $this->nHeads * $this->maxSeqLen * 2 * $this->headDim * 4;
    }

    public function __destruct()
    {
        TensorEngine::get()->mkvca_free($this->ptr);
    }

    // ── Private ───────────────────────────────────────────────────────────────

    private function checkError(string $op): void
    {
        $ffi = TensorEngine::get();
        if ($ffi->tensor_check_error()) {
            $msg = \FFI::string($ffi->tensor_get_last_error());
            $ffi->tensor_clear_error();
            throw new \RuntimeException("KVCache::{$op} — {$msg}");
        }
    }
}
