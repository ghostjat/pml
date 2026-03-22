<?php
declare(strict_types=1);

namespace Pml\Layers;

use Pml\Tensor;
use Pml\Ops;
use Pml\Generation\KVCache;

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

    /**
     * Unified forward pass. 
     * If $cache is provided, it processes a single token and updates the cache.
     * If $cache is null, it performs standard full-sequence attention.
     */
    public function forward(Tensor $x, ?KVCache $cache = null): Tensor 
    {
        // 1. Project Q, K, V
        $q = Ops::matmul($x, $this->wq); 
        $k = Ops::matmul($x, $this->wk); 
        $v = Ops::matmul($x, $this->wv); 

        if ($cache !== null) {
            // --- CACHED ROUTING (O(N) single token step) ---
            $cache->append($k, $v);
            $kActive = $cache->getActiveK();
            $vActive = $cache->getActiveV();

            $scores = Ops::matmul($q, $kActive, false, true);
            $this->scaleAndSoftmax($scores);
            $context = Ops::matmul($scores, $vActive);
            
        } else {
            // --- STANDARD ROUTING (O(N^2) full sequence) ---
            $scores = Ops::matmul($q, $k, false, true);
            $this->scaleAndSoftmax($scores);
            $context = Ops::matmul($scores, $v);
        }

        // Final output projection
        return Ops::matmul($context, $this->wo);
    }

    private function scaleAndSoftmax(Tensor $scores): void 
    {
        $scale = 1.0 / sqrt($this->headDim);
        for ($i = 0; $i < $scores->size; $i++) {
            $scores->buffer[$i] *= $scale;
        }
        Ops::softmax($scores);
    }
}