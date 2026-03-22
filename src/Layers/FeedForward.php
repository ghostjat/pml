<?php

declare(strict_types=1);

namespace Pml\Layers;

use Pml\{Tensor, Ops, BlasEngine};


// ═══════════════════════════════════════════════════════════════════════════
//  FEED-FORWARD NETWORK (SwiGLU variant — used in LLaMA/Mistral)
// ═══════════════════════════════════════════════════════════════════════════

class FeedForward
{
    /**
     * SwiGLU: FFN(x) = (SiLU(x * W1) ⊙ (x * W3)) * W2
     *
     * @param Tensor $w1  [d_model, d_ff]
     * @param Tensor $w2  [d_ff, d_model]
     * @param Tensor $w3  [d_model, d_ff]  (gate projection)
     */
    public function __construct(
        private readonly Tensor $w1,
        private readonly Tensor $w2,
        private readonly Tensor $w3,
    ) {}

    public function forward(Tensor $x): Tensor
    {
        // Gate: SiLU(x * W1)
        $gate = Ops::silu(Ops::matmul($x, $this->w1)); // [seq, d_ff]

        // Value: x * W3
        $val  = Ops::matmul($x, $this->w3);             // [seq, d_ff]

        // Hadamard: gate ⊙ val
        $merged = Ops::mul($gate, $val);                 // [seq, d_ff]

        // Down projection
        return Ops::matmul($merged, $this->w2);          // [seq, d_model]
    }
}

