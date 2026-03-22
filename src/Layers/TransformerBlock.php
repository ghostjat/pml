<?php

declare(strict_types=1);

namespace Pml\Layers;

use Pml\{Tensor, Ops, BlasEngine};
use Pml\Layers\{MultiHeadAttention,FeedForward,KVCache};

// ═══════════════════════════════════════════════════════════════════════════
//  TRANSFORMER BLOCK (LLaMA/Mistral-style with RMSNorm + SwiGLU)
// ═══════════════════════════════════════════════════════════════════════════

class TransformerBlock
{
    public function __construct(
        private readonly MultiHeadAttention $attention,
        private readonly FeedForward        $ffn,
        private readonly Tensor             $normAttn,  // RMSNorm weight [d_model]
        private readonly Tensor             $normFFN,   // RMSNorm weight [d_model]
        private readonly float              $rmsEps = 1e-5,
    ) {}

    /**
     * Pre-norm residual block:
     *   x = x + Attention(RMSNorm(x))
     *   x = x + FFN(RMSNorm(x))
     */
    public function forward(Tensor $x, ?KVCache $cache = null, int $pos = 0): Tensor
    {
        // ── Attention sub-layer ─────────────────────────────────────────────
        $residual = $x->clone();
        Ops::rmsNormInPlace($x, $this->normAttn, $this->rmsEps);
        $attnOut = $this->attention->forward($x, $cache, $pos);
        Ops::saxpy($attnOut, $residual, 1.0); // residual += attnOut

        // ── FFN sub-layer ───────────────────────────────────────────────────
        $x2 = $residual->clone();
        Ops::rmsNormInPlace($x2, $this->normFFN, $this->rmsEps);
        $ffnOut = $this->ffn->forward($x2);
        Ops::saxpy($ffnOut, $residual, 1.0); // residual += ffnOut

        return $residual;
    }
}