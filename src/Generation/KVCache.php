<?php
declare(strict_types=1);

namespace Pml\Generation;

use Pml\Tensor;
use Pml\BlasEngine;

class KVCache
{
    private Tensor $k;
    private Tensor $v;
    private int $maxSeqLen;
    private int $dModel;
    private int $currentPos = 0;

    public function __construct(int $maxSeqLen, int $dModel)
    {
        $this->maxSeqLen = $maxSeqLen;
        $this->dModel = $dModel;
        
        // Pre-allocate the maximum possible size to prevent memory fragmentation
        $this->k = new Tensor([$maxSeqLen, $dModel]);
        $this->v = new Tensor([$maxSeqLen, $dModel]);
    }

    /**
     * Appends a new [1, d_model] token representation to the cache.
     * ZERO PHP loops. Pure FFI pointer arithmetic and BLAS copy.
     */
    public function append(Tensor $newK, Tensor $newV): void
    {
        if ($this->currentPos >= $this->maxSeqLen) {
            throw new \RuntimeException("KV Cache Exceeded Maximum Sequence Length!");
        }

        $ffi = BlasEngine::get()->ffi;
        
        // Calculate the memory address offset for the current position
        $kOffset = \FFI::cast("float*", \FFI::addr($this->k->buffer[$this->currentPos * $this->dModel]));
        $vOffset = \FFI::cast("float*", \FFI::addr($this->v->buffer[$this->currentPos * $this->dModel]));

        // Copy the single row into the pre-allocated matrix buffer
        $ffi->cblas_scopy($this->dModel, $newK->buffer, 1, $kOffset, 1);
        $ffi->cblas_scopy($this->dModel, $newV->buffer, 1, $vOffset, 1);

        $this->currentPos++;
    }

    /**
     * Creates a "View" of the cache up to the current position.
     * This is an O(1) operation. It creates a new PHP Tensor object that 
     * points to the EXACT SAME underlying CData buffer, just with a smaller shape.
     */
    public function getActiveK(): Tensor
    {
        return new Tensor([$this->currentPos, $this->dModel], $this->k->buffer);
    }

    public function getActiveV(): Tensor
    {
        return new Tensor([$this->currentPos, $this->dModel], $this->v->buffer);
    }
}