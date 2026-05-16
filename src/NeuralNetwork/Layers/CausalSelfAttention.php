<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Interfaces\Stateful;
use Pml\KVCache;
use Pml\Tensor;

/**
 * Causal Multi-Head Self-Attention.
 *
 * Projects X → Q, K, V, runs causal masked attention per head,
 * concatenates heads, projects back to d_model.
 *
 * Shapes (sequence-level, B=1):
 *   forward input:  [T, D]
 *   forward output: [T, D]
 *
 * Internal projections:
 *   Wq, Wk, Wv: [D, D]    (weight matrices, stored as [D, D] — linear uses W[out,in])
 *   Wo:          [D, D]    output projection
 *   All biases: [D]
 *
 * State-dict keys (with prefix):
 *   "Wq.weight", "Wq.bias", "Wk.weight", "Wk.bias",
 *   "Wv.weight", "Wv.bias", "Wo.weight", "Wo.bias"
 */
final class CausalSelfAttention implements Layer, Stateful
{
    private readonly int $headDim;

    // Q, K, V, O projection layers
    private Dense $Wq;
    private Dense $Wk;
    private Dense $Wv;
    private Dense $Wo;

    // Saved for backward
    private ?Tensor $savedX   = null;  // [T, D]
    private ?Tensor $savedQ   = null;  // [nH, T, hd]
    private ?Tensor $savedK   = null;  // [nH, T, hd]
    private ?Tensor $savedV   = null;  // [nH, T, hd]
    private ?Tensor $savedAttn = null; // [nH, T, T]

    public function __construct(
        private readonly int $dModel,
        private readonly int $nHeads,
    ) {
        if ($dModel % $nHeads !== 0) {
            throw new \InvalidArgumentException("dModel must be divisible by nHeads");
        }
        $this->headDim = $dModel / $nHeads;
        $this->Wq = new Dense($dModel, $dModel);
        $this->Wk = new Dense($dModel, $dModel);
        $this->Wv = new Dense($dModel, $dModel);
        $this->Wo = new Dense($dModel, $dModel);
    }

    /**
     * Forward pass.
     *
     * Training / prefill (kv = null):
     *   Full O(T²) causal attention; saves activations for backward().
     *
     * KV-cached prefill (kv != null, T > 1):
     *   Full causal attention for the prompt, populates cache — no extra cost
     *   vs. the training path, and the cache is ready for decode steps.
     *
     * KV-cached decode (kv != null, T = 1):
     *   Appends the new token's K/V to the cache then runs Milakov O(seq_cache)
     *   streaming attention — 2 FFI crossings total (append + attend).
     *
     * @param Tensor        $input    [T, D]
     * @param KVCache|null  $kv       Multi-head cache; null for training
     * @param int           $layerIdx Layer index passed to the cache
     */
    public function forward(Tensor $input, ?KVCache $kv = null, int $layerIdx = 0): Tensor
    {
        $T  = $input->shape()[0];
        $nH = $this->nHeads;
        $hd = $this->headDim;
        $D  = $this->dModel;

        $Q = $this->Wq->forward($input);  // [T, D]
        $K = $this->Wk->forward($input);
        $V = $this->Wv->forward($input);

        if ($kv === null || $T > 1) {
            // Training path or prefill: full causal attention
            $Qr = $Q->reshape($T, $nH, $hd)->transposeNd([1, 0, 2])->contiguous(); // [nH,T,hd]
            $Kr = $K->reshape($T, $nH, $hd)->transposeNd([1, 0, 2])->contiguous();
            $Vr = $V->reshape($T, $nH, $hd)->transposeNd([1, 0, 2])->contiguous();

            $out  = Tensor::zeros($nH, $T, $hd);
            $attn = Tensor::zeros($nH, $T, $T);
            $out->causalAttention($Qr, $Kr, $Vr, $attn);

            if ($kv !== null) {
                // Populate cache from prompt K/V so decode steps can use it
                $kv->prefill($layerIdx, $Kr, $Vr);  // 1 FFI call, fills all nH heads
            } else {
                // Save activations only when training (backward needed)
                $this->savedX    = $input;
                $this->savedQ    = $Qr;
                $this->savedK    = $Kr;
                $this->savedV    = $Vr;
                $this->savedAttn = $attn;
            }

            $merged = $out->transposeNd([1, 0, 2])->contiguous()->reshape($T, $D);
            return $this->Wo->forward($merged);
        }

        // KV-cached decode: T == 1
        // Squeeze T dim: [1, D] → reshape directly to [nH, hd]
        $Kr = $K->reshape($nH, $hd)->contiguous();  // [nH, hd]
        $Vr = $V->reshape($nH, $hd)->contiguous();
        $Qr = $Q->reshape($nH, 1, $hd)->contiguous();  // [nH, 1, hd]

        $kv->append($layerIdx, $Kr, $Vr);              // 1 FFI call: append to all heads
        $attnOut = $kv->attend($layerIdx, $Qr);        // 1 FFI call: attend [nH,1,hd]

        // Merge heads: [nH, 1, hd] → [1, nH, hd] → [1, D]
        $merged = $attnOut->transposeNd([1, 0, 2])->contiguous()->reshape(1, $D);
        return $this->Wo->forward($merged);
    }

    public function backward(Tensor $dY): Tensor
    {
        $T  = $this->savedX->shape()[0];
        $nH = $this->nHeads;
        $hd = $this->headDim;
        $D  = $this->dModel;

        // Backward through output projection: dY [T, D] → dMerged [T, D]
        $dMerged = $this->Wo->backward($dY);

        // Reshape dMerged [T, D] → [nH, T, hd]
        $dOut = $dMerged->reshape($T, $nH, $hd)->transposeNd([1, 0, 2])->contiguous();

        // Backward through causal attention → dQ, dK, dV [nH, T, hd]
        $dQ = Tensor::zeros($nH, $T, $hd);
        $dK = Tensor::zeros($nH, $T, $hd);
        $dV = Tensor::zeros($nH, $T, $hd);

        $dOut->causalAttentionBackward(
            $this->savedAttn, $this->savedQ, $this->savedK, $this->savedV,
            $dQ, $dK, $dV
        );

        // Reshape dQ/dK/dV: [nH, T, hd] → [T, D]
        $dQ2D = $dQ->transposeNd([1, 0, 2])->contiguous()->reshape($T, $D);
        $dK2D = $dK->transposeNd([1, 0, 2])->contiguous()->reshape($T, $D);
        $dV2D = $dV->transposeNd([1, 0, 2])->contiguous()->reshape($T, $D);

        // Backward through Q/K/V projections: accumulate dX
        $dX  = $this->Wq->backward($dQ2D);
        $dX->addInplace($this->Wk->backward($dK2D));
        $dX->addInplace($this->Wv->backward($dV2D));

        // Release saved tensors
        $this->savedX = $this->savedQ = $this->savedK = $this->savedV = $this->savedAttn = null;

        return $dX;
    }

    public function getParameters(): array
    {
        return array_merge(
            $this->prefixParams('Wq.', $this->Wq->getParameters()),
            $this->prefixParams('Wk.', $this->Wk->getParameters()),
            $this->prefixParams('Wv.', $this->Wv->getParameters()),
            $this->prefixParams('Wo.', $this->Wo->getParameters()),
        );
    }

    public function getGradients(): array
    {
        return array_merge(
            $this->prefixParams('Wq.', $this->Wq->getGradients()),
            $this->prefixParams('Wk.', $this->Wk->getGradients()),
            $this->prefixParams('Wv.', $this->Wv->getGradients()),
            $this->prefixParams('Wo.', $this->Wo->getGradients()),
        );
    }

    public function getStateDict(string $prefix = ''): array
    {
        return array_merge(
            $this->Wq->getStateDict($prefix . 'Wq.'),
            $this->Wk->getStateDict($prefix . 'Wk.'),
            $this->Wv->getStateDict($prefix . 'Wv.'),
            $this->Wo->getStateDict($prefix . 'Wo.'),
        );
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->Wq->loadStateDict($dict, $prefix . 'Wq.');
        $this->Wk->loadStateDict($dict, $prefix . 'Wk.');
        $this->Wv->loadStateDict($dict, $prefix . 'Wv.');
        $this->Wo->loadStateDict($dict, $prefix . 'Wo.');
    }

    public function getConfig(): array
    {
        return ['dModel' => $this->dModel, 'nHeads' => $this->nHeads];
    }

    private function prefixParams(string $prefix, array $params): array
    {
        $out = [];
        foreach ($params as $k => $v) $out[$prefix . $k] = $v;
        return $out;
    }
}
