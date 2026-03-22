<?php

declare(strict_types=1);

namespace Pml\Layers;

use Pml\{Tensor, Ops, BlasEngine};
use Pml\Layers\RoPE;
use Pml\Layers\KVCache;

// ═══════════════════════════════════════════════════════════════════════════
//  MULTI-HEAD SELF-ATTENTION (with optional KV Cache + RoPE)
// ═══════════════════════════════════════════════════════════════════════════

class MultiHeadAttention
{
    private RoPE   $rope;
    private int    $headDim;

    public function __construct(
        private readonly Tensor $wq,          // [d_model, d_model]
        private readonly Tensor $wk,          // [d_model, d_kv]
        private readonly Tensor $wv,          // [d_model, d_kv]
        private readonly Tensor $wo,          // [d_model, d_model]
        private readonly int    $nHeads,
        private readonly int    $nKVHeads,    // For Grouped Query Attention (GQA)
        private readonly float  $ropeTheta = 10000.0,
    ) {
        $this->headDim = $wq->shape[1] / $nHeads;
        $this->rope    = new RoPE($this->headDim, $ropeTheta);
    }

    /**
     * Forward pass.
     *
     * $x:    [seq_len, d_model]
     * $cache: Optional KV cache for autoregressive generation.
     * $pos:   Starting position (for RoPE + KV cache offset).
     * $mask:  If true, applies causal mask (for decoder self-attention).
     *
     * Returns [seq_len, d_model].
     */
    public function forward(
        Tensor    $x,
        ?KVCache  $cache   = null,
        int       $pos     = 0,
        bool      $causal  = true
    ): Tensor {
        $seqLen   = $x->shape[0];
        $dModel   = $x->shape[1];
        $scale    = 1.0 / sqrt($this->headDim);
        $groupSize = intdiv($this->nHeads, $this->nKVHeads); // heads per KV head (GQA)

        // ── 1. QKV Projections ─────────────────────────────────────────────
        $q = Ops::matmul($x, $this->wq); // [seq, nHeads*headDim]
        $k = Ops::matmul($x, $this->wk); // [seq, nKVHeads*headDim]
        $v = Ops::matmul($x, $this->wv); // [seq, nKVHeads*headDim]

        // ── 2. Apply RoPE to Q and K ───────────────────────────────────────
        $this->rope->apply($q, $pos);
        $this->rope->apply($k, $pos);

        // ── 3. KV Cache: append + retrieve full context ────────────────────
        if ($cache !== null) {
            $cache->append($k, $v);
            $k = $cache->getActiveK(); // [ctx, nKVHeads*headDim]
            $v = $cache->getActiveV(); // [ctx, nKVHeads*headDim]
        }

        // ── 4. Per-head scaled dot-product attention (with GQA grouping) ──
        $output = Tensor::zeros([$seqLen, $dModel]);

        for ($h = 0; $h < $this->nHeads; $h++) {
            $kvH  = intdiv($h, $groupSize);
            $qOff = $h   * $this->headDim;
            $kOff = $kvH * $this->headDim;

            $q_h = $q->sliceCols($qOff, $qOff + $this->headDim); // [seq, headDim]
            $k_h = $k->sliceCols($kOff, $kOff + $this->headDim); // [ctx, headDim]
            $v_h = $v->sliceCols($kOff, $kOff + $this->headDim); // [ctx, headDim]

            $scores = Ops::matmul($q_h, $k_h, false, true); // [seq, ctx]
            $scores->scaleInPlace($scale);

            if ($causal && $seqLen > 1) {
                Ops::applyCausalMaskInPlace($scores);
            }

            Ops::softmaxInPlace($scores);

            $out_h = Ops::matmul($scores, $v_h); // [seq, headDim]
            $output->setColSlice($qOff, $out_h);
        }

        // ── 5. Output projection ───────────────────────────────────────────
        return Ops::matmul($output, $this->wo); // [seq, d_model]
    }
}