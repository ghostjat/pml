<?php

declare(strict_types=1);

namespace Pml\Layers;

use Pml\{Tensor, Ops, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  ROTARY POSITIONAL EMBEDDING (RoPE)
//  Used in LLaMA, Mistral, Gemma, Phi, Falcon.
// ═══════════════════════════════════════════════════════════════════════════

final class RoPE
{
    /** @var float[] Pre-computed inverse frequencies */
    private array $invFreq;

    public function __construct(
        private readonly int   $headDim,
        private readonly float $theta    = 10000.0,
        private readonly int   $maxSeqLen = 4096,
    ) {
        // inv_freq[i] = 1 / theta^(2i / headDim)
        $this->invFreq = [];
        for ($i = 0; $i < $headDim / 2; $i++) {
            $this->invFreq[] = 1.0 / pow($theta, (2.0 * $i) / $headDim);
        }
    }

    /**
     * Apply RoPE to query/key tensors.
     *
     * $q: [seq_len, n_heads * head_dim]  or  [seq_len, head_dim] for a single head
     * $startPos: starting position in the sequence (for KV-cache incremental decoding)
     *
     * Modifies $q in-place and returns it.
     */
    public function apply(Tensor $q, int $startPos = 0): Tensor
    {
        $seqLen  = $q->shape[0];
        $dim     = $q->shape[1]; // should be nHeads * headDim
        $halfDim = $this->headDim / 2;

        for ($pos = 0; $pos < $seqLen; $pos++) {
            $absPos = $startPos + $pos;
            $rowOff = $pos * $dim;

            // Process head_dim at a time (each head has its own RoPE rotation)
            for ($h = 0; $h < $dim; $h += $this->headDim) {
                for ($i = 0; $i < $halfDim; $i++) {
                    $angle = $absPos * $this->invFreq[$i];
                    $cos   = cos($angle);
                    $sin   = sin($angle);

                    $x0 = (float)$q->buffer[$rowOff + $h + $i];
                    $x1 = (float)$q->buffer[$rowOff + $h + $i + $halfDim];

                    $q->buffer[$rowOff + $h + $i]           = $x0 * $cos - $x1 * $sin;
                    $q->buffer[$rowOff + $h + $i + $halfDim] = $x0 * $sin + $x1 * $cos;
                }
            }
        }
        return $q;
    }
}


