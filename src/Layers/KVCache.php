<?php

declare(strict_types=1);

namespace Pml\Layers;

use Pml\{Tensor, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  KV CACHE  (with Sliding Window / Ring Buffer)
//
//  Pre-allocated ring buffer for transformer autoregressive decoding.
//
//  Design:
//    - Capacity is fixed at construction time ($maxSeqLen tokens).
//    - When $totalTokens < $maxSeqLen: behaves identically to a plain cache;
//      getActiveK/V() return a zero-copy view into the pre-allocated buffer.
//    - When $totalTokens >= $maxSeqLen (sliding window kicks in):
//        • New tokens are written into the slot at index
//            writeSlot = totalTokens % maxSeqLen
//          overwriting the oldest token in the ring.
//        • getActiveK/V() reconstruct a chronologically-ordered copy using
//          two BLAS scopy calls (tail of ring → head of output, then head of
//          ring → tail of output).
//        • The reorder is O(maxSeqLen × dModel) floats copied — for a 1024-
//          token window with d_kv=512, that is 2 MB, done in native C via
//          cblas_scopy.  It happens once per generated token.
//
//  Why ring buffer instead of memmove?
//    memmove shifts the entire cache left on every token, which is the same
//    O(N) cost but writes to every cache line, thrashing L3.  The ring
//    buffer writes only one new row per step and defers reordering to the
//    read path (getActive*), keeping the write path cache-friendly.
//
//  GQA support:
//    The buffer dimension $dModel should equal (nKVHeads * headDim), NOT
//    (nHeads * headDim).  MultiHeadAttention is responsible for slicing the
//    correct KV head from the returned tensor.
// ═══════════════════════════════════════════════════════════════════════════

class KVCache
{
    /** Pre-allocated ring buffers — shape [maxSeqLen, dModel] */
    private Tensor $k;
    private Tensor $v;

    /**
     * Total tokens ever appended (monotonically increasing).
     * This is the true position counter used by RoPE.
     */
    private int $totalTokens = 0;

    /**
     * @param int $maxSeqLen  Sliding-window size: maximum tokens kept in RAM.
     *                        Older tokens are silently evicted once this is exceeded.
     * @param int $dModel     KV projection width = nKVHeads * headDim.
     */
    public function __construct(
        private readonly int $maxSeqLen,
        private readonly int $dModel
    ) {
        // Allocate once — no reallocation ever occurs during a session
        $this->k = Tensor::zeros([$maxSeqLen, $dModel]);
        $this->v = Tensor::zeros([$maxSeqLen, $dModel]);
    }

    // ── Write path ────────────────────────────────────────────────────────

    /**
     * Append one or more new K/V rows into the ring buffer.
     *
     * Each row is written to  slot = totalTokens % maxSeqLen,
     * overwriting the slot of the oldest token once the window is full.
     *
     * Uses cblas_scopy — zero PHP-level loops over individual floats.
     *
     * @param Tensor $newK  [nRows, dModel]
     * @param Tensor $newV  [nRows, dModel]
     */
    public function append(Tensor $newK, Tensor $newV): void
    {
        $nRows = $newK->shape[0];
        $ffi   = BlasEngine::get()->ffi;
        $d     = $this->dModel;

        for ($r = 0; $r < $nRows; $r++) {
            // Ring-buffer slot for this token — wraps around silently
            $slot = $this->totalTokens % $this->maxSeqLen;

            // Source: row $r of the incoming K/V tensors
            $kSrc = \FFI::cast('float*', \FFI::addr($newK->buffer[$r * $d]));
            $vSrc = \FFI::cast('float*', \FFI::addr($newV->buffer[$r * $d]));

            // Destination: slot $slot in the ring buffers
            $kDst = \FFI::cast('float*', \FFI::addr($this->k->buffer[$slot * $d]));
            $vDst = \FFI::cast('float*', \FFI::addr($this->v->buffer[$slot * $d]));

            $ffi->cblas_scopy($d, $kSrc, 1, $kDst, 1);
            $ffi->cblas_scopy($d, $vSrc, 1, $vDst, 1);

            $this->totalTokens++;
        }
    }

    // ── Read path ─────────────────────────────────────────────────────────

    /**
     * Return the active key cache as a chronologically-ordered tensor.
     *
     * Before wrap-around: O(1) zero-copy view of the first $totalTokens rows.
     * After wrap-around:  O(window) BLAS copy to restore chronological order.
     *
     * @return Tensor  [min(totalTokens, maxSeqLen), dModel]
     */
    public function getActiveK(): Tensor
    {
        return $this->getOrdered($this->k);
    }

    public function getActiveV(): Tensor
    {
        return $this->getOrdered($this->v);
    }

    // ── Inspection ────────────────────────────────────────────────────────

    /** Number of tokens currently in the active window (capped at maxSeqLen). */
    public function currentLength(): int
    {
        return min($this->totalTokens, $this->maxSeqLen);
    }

    /** Total tokens ever appended — use this as the RoPE position. */
    public function totalTokens(): int
    {
        return $this->totalTokens;
    }

    public function maxLength(): int  { return $this->maxSeqLen; }
    public function isFull(): bool    { return $this->totalTokens >= $this->maxSeqLen; }

    /** Reset to empty state — call between independent generation sessions. */
    public function reset(): void     { $this->totalTokens = 0; }

    // ── Private: ring-buffer reorder ──────────────────────────────────────

    /**
     * Return the cache rows in chronological order.
     *
     *   Before wrap (totalTokens ≤ maxSeqLen):
     *     rows [0 .. totalTokens) are already in order → zero-copy view.
     *
     *   After wrap (totalTokens > maxSeqLen):
     *     The oldest surviving token lives at slot = totalTokens % maxSeqLen.
     *     The ring looks like:
     *
     *       [  newer  |  oldest ... newest  |  newer  ]
     *         0 ..writePtr-1    writePtr .. maxSeqLen-1
     *
     *     Chronological order is: [writePtr .. maxSeqLen) ++ [0 .. writePtr)
     *     We reconstruct it with two cblas_scopy calls (tail then head).
     */
    private function getOrdered(Tensor $ring): Tensor
    {
        $active = $this->currentLength();

        if ($this->totalTokens <= $this->maxSeqLen) {
            // Not yet wrapped — plain zero-copy view into the first $active rows
            return new Tensor([$active, $this->dModel], $ring->buffer);
        }

        // Wrapped: reconstruct chronological order into a fresh tensor
        $out  = Tensor::zeros([$this->maxSeqLen, $this->dModel]);
        $ffi  = BlasEngine::get()->ffi;
        $d    = $this->dModel;

        // Write pointer = position of the NEXT write = oldest slot
        $oldestSlot = $this->totalTokens % $this->maxSeqLen;

        // ── Part A: tail of ring [oldestSlot .. maxSeqLen) → output[0 .. tail) ──
        $tailRows = $this->maxSeqLen - $oldestSlot;
        if ($tailRows > 0) {
            $src = \FFI::cast('float*', \FFI::addr($ring->buffer[$oldestSlot * $d]));
            $ffi->cblas_scopy($tailRows * $d, $src, 1, $out->buffer, 1);
        }

        // ── Part B: head of ring [0 .. oldestSlot) → output[tailRows .. end) ──
        if ($oldestSlot > 0) {
            $dst = \FFI::cast('float*', \FFI::addr($out->buffer[$tailRows * $d]));
            $ffi->cblas_scopy($oldestSlot * $d, $ring->buffer, 1, $dst, 1);
        }

        return $out;
    }
}
