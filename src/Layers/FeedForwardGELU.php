<?php

declare(strict_types=1);

namespace Pml\Layers;

use Pml\{Tensor, Ops, BlasEngine};

// ── GELU-based FFN (used in BERT, GPT-2) ──────────────────────────────────
class FeedForwardGELU
{
    public function __construct(
        private readonly Tensor $w1,  // [d_model, d_ff]
        private readonly Tensor $w2,  // [d_ff, d_model]
        private readonly ?Tensor $b1 = null,
        private readonly ?Tensor $b2 = null,
    ) {}

    public function forward(Tensor $x): Tensor
    {
        $h = Ops::matmul($x, $this->w1);
        if ($this->b1 !== null) Ops::addBiasInPlace($h, $this->b1);
        $h = Ops::gelu($h);
        $out = Ops::matmul($h, $this->w2);
        if ($this->b2 !== null) Ops::addBiasInPlace($out, $this->b2);
        return $out;
    }
}
