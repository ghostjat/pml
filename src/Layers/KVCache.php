<?php

declare(strict_types=1);

namespace Pml\Layers;

use Pml\{Tensor, Ops, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  KV CACHE
//  Pre-allocated ring buffer for transformer autoregressive decoding.
// ═══════════════════════════════════════════════════════════════════════════

class KVCache
{
    private Tensor $k;
    private Tensor $v;
    private int    $currentPos = 0;

    /**
     * @param int $maxSeqLen  Maximum number of tokens the cache can hold.
     * @param int $dModel     KV projection dimension (n_heads * head_dim).
     */
    public function __construct(
        private readonly int $maxSeqLen,
        private readonly int $dModel
    ) {
        // Pre-allocate maximum capacity — avoids realloc during generation loop
        $this->k = Tensor::zeros([$maxSeqLen, $dModel]);
        $this->v = Tensor::zeros([$maxSeqLen, $dModel]);
    }

    /**
     * Append a new [1, d_model] (or [seq, d_model]) key/value pair.
     * Uses BLAS scopy — zero PHP loops over values.
     */
    public function append(Tensor $newK, Tensor $newV): void
    {
        $nRows = $newK->shape[0];

        if ($this->currentPos + $nRows > $this->maxSeqLen) {
            throw new \RuntimeException(
                "KVCache overflow: capacity={$this->maxSeqLen}, requested position=" . ($this->currentPos + $nRows)
            );
        }

        $ffi  = BlasEngine::get()->ffi;
        $sz   = $nRows * $this->dModel;

        $kDst = \FFI::cast('float*', \FFI::addr($this->k->buffer[$this->currentPos * $this->dModel]));
        $vDst = \FFI::cast('float*', \FFI::addr($this->v->buffer[$this->currentPos * $this->dModel]));

        $ffi->cblas_scopy($sz, $newK->buffer, 1, $kDst, 1);
        $ffi->cblas_scopy($sz, $newV->buffer, 1, $vDst, 1);

        $this->currentPos += $nRows;
    }

    /**
     * Returns a ZERO-COPY view into the cache up to the current position.
     * O(1): just constructs a new Tensor that aliases the existing C buffer.
     */
    public function getActiveK(): Tensor
    {
        return new Tensor([$this->currentPos, $this->dModel], $this->k->buffer);
    }

    public function getActiveV(): Tensor
    {
        return new Tensor([$this->currentPos, $this->dModel], $this->v->buffer);
    }

    public function currentLength(): int { return $this->currentPos; }
    public function maxLength(): int     { return $this->maxSeqLen; }
    public function isFull(): bool       { return $this->currentPos >= $this->maxSeqLen; }

    public function reset(): void        { $this->currentPos = 0; }
}


