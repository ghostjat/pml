<?php
declare(strict_types=1);

namespace Pml\Layers;

use Pml\Tensor;
use Pml\Ops;

class SelfAttention 
{
    private Tensor $wq, $wk, $wv, $wo;
    private int $nHeads, $headDim;

    public function __construct(Tensor $wq, Tensor $wk, Tensor $wv, Tensor $wo, int $nHeads) 
    {
        $this->wq = $wq;
        $this->wk = $wk;
        $this->wv = $wv;
        $this->wo = $wo;
        $this->nHeads = $nHeads;
        $this->headDim = $wq->shape[1] / $nHeads;
    }

    public function forward(Tensor $x): Tensor 
    {
        // 1. Project Q, K, V
        $q = Ops::matmul($x, $this->wq);
        $k = Ops::matmul($x, $this->wk);
        $v = Ops::matmul($x, $this->wv);

        // NOTE: In a full implementation, you would apply RoPE (Rotary Embeddings) 
        // to Q and K here, and reshape them for Multi-Head Attention.
        // For this skeletal representation, we perform standard attention.

        // 2. Q * K^T
        $scores = Ops::matmul($q, $k, false, true);

        // 3. Scale by 1/sqrt(d_k)
        $scale = 1.0 / sqrt($this->headDim);
        for ($i = 0; $i < $scores->size; $i++) {
            $scores->buffer[$i] *= $scale;
        }

        // 4. Softmax
        Ops::softmax($scores);

        // 5. Scores * V
        $context = Ops::matmul($scores, $v);

        // 6. Final output projection
        return Ops::matmul($context, $this->wo);
    }
}

