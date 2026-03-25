<?php
declare(strict_types=1);

use Pml\Tensor;
use Pml\Ops;
use Pml\BlasEngine;

function buildSinusoidalPE(int $maxSeqLen, int $dModel): Tensor {
    $pe = Tensor::zeros([$maxSeqLen, $dModel]);
    for ($pos = 0; $pos < $maxSeqLen; $pos++) {
        for ($i = 0; $i < intdiv($dModel, 2); $i++) {
            $angle = $pos / (10000.0 ** (2.0 * $i / $dModel));
            $pe->buffer[$pos * $dModel + 2 * $i] = (float) sin($angle);
            $pe->buffer[$pos * $dModel + 2 * $i + 1] = (float) cos($angle);
        }
    }
    return $pe->detach(); // requiresGrad = false
}
function embeddingWithGrad(Tensor $weight, array $ids): Tensor {
    $dModel = $weight->shape[1];
    $seqLen = count($ids);
    $out = Tensor::zeros([$seqLen, $dModel]);
    $ffi = BlasEngine::get()->ffi;

    for ($i = 0; $i < $seqLen; $i++) {
        $id = $ids[$i];
        $src = \FFI::cast('float*', \FFI::addr($weight->buffer[$id * $dModel]));
        $dst = \FFI::cast('float*', \FFI::addr($out->buffer[$i * $dModel]));
        $ffi->cblas_scopy($dModel, $src, 1, $dst, 1);
    }

    if ($weight->requiresGrad) {
        $out->requiresGrad = true;
        $out->_prev = [$weight];
        $capturedIds = $ids;

        $out->_backward = static function ()
        use ($weight, $out, $capturedIds, $dModel, $seqLen, $ffi): void {
            $weight->initGrad();
            for ($i = 0; $i < $seqLen; $i++) {
                $id = $capturedIds[$i];
                $gSrc = \FFI::cast('float*', \FFI::addr($out->grad[$i * $dModel]));
                $gDst = \FFI::cast('float*', \FFI::addr($weight->grad[$id * $dModel]));
                $ffi->cblas_saxpy($dModel, 1.0, $gSrc, 1, $gDst, 1);
            }
        };
    }

    return $out;
}

function scaleWithGrad(Tensor $x, float $scale): Tensor {
    $out = $x->clone();
    BlasEngine::get()->ffi->cblas_sscal($out->size, $scale, $out->buffer, 1);

    if ($x->requiresGrad) {
        $out->requiresGrad = true;
        $out->_prev = [$x];

        $out->_backward = static function () use ($x, $out, $scale): void {
            $x->initGrad();
            BlasEngine::get()->ffi->cblas_saxpy($out->size, $scale, $out->grad, 1, $x->grad, 1);
        };
    }

    return $out;
}

function softmaxRowsWithGrad(Tensor $x): Tensor {
    [$M, $N] = $x->shape;
    $out = Tensor::zeros([$M, $N]);
    $ffi = BlasEngine::get()->ffi;

    for ($i = 0; $i < $M; $i++) {
        $off = $i * $N;
        $maxV = (float) $x->buffer[$off];
        for ($j = 1; $j < $N; $j++) {
            $v = (float) $x->buffer[$off + $j];
            if ($v > $maxV)
                $maxV = $v;
        }
        $sum = 0.0;
        for ($j = 0; $j < $N; $j++) {
            $e = exp((float) $x->buffer[$off + $j] - $maxV);
            $out->buffer[$off + $j] = $e;
            $sum += $e;
        }
        $rowPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$off]));
        $ffi->cblas_sscal($N, 1.0 / $sum, $rowPtr, 1);
    }

    if ($x->requiresGrad) {
        $out->requiresGrad = true;
        $out->_prev = [$x];

        $out->_backward = static function () use ($x, $out, $M, $N): void {
            $x->initGrad();
            for ($i = 0; $i < $M; $i++) {
                $off = $i * $N;
                $dot = 0.0;
                for ($j = 0; $j < $N; $j++) {
                    $dot += (float) $out->buffer[$off + $j] * (float) $out->grad[$off + $j];
                }
                for ($j = 0; $j < $N; $j++) {
                    $p = (float) $out->buffer[$off + $j];
                    $dP = (float) $out->grad[$off + $j];
                    $x->grad[$off + $j] = (float) $x->grad[$off + $j] + $p * ($dP - $dot);
                }
            }
        };
    }

    return $out;
}

function reluWithGrad(Tensor $x): Tensor {
    $n = $x->size;
    $out = Tensor::zeros($x->shape);

    for ($i = 0; $i < $n; $i++) {
        $v = (float) $x->buffer[$i];
        if ($v > 0.0)
            $out->buffer[$i] = $v;
    }

    if ($x->requiresGrad) {
        $out->requiresGrad = true;
        $out->_prev = [$x];

        $out->_backward = static function () use ($x, $out, $n): void {
            $x->initGrad();
            for ($i = 0; $i < $n; $i++) {
                if ((float) $out->buffer[$i] > 0.0) {
                    $x->grad[$i] = (float) $x->grad[$i] + (float) $out->grad[$i];
                }
            }
        };
    }

    return $out;
}
function applyCausalMaskInPlace(Tensor $scores): void {
    $seqLen = $scores->shape[0];
    for ($i = 0; $i < $seqLen; $i++) {
        for ($j = $i + 1; $j < $seqLen; $j++) {
            $scores->buffer[$i * $seqLen + $j] = -1.0e9;
        }
    }
}

final class TinyGPT
{
    /** Token embedding table: [vocabSize, dModel] */
    public Tensor $tokenEmb;

    /**
     * Per-layer weight matrices.
     * @var array<int, array{wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor,
     *                        w1: Tensor, w2: Tensor, rms1_w: Tensor, rms2_w: Tensor}>
     */
    public array $layers = [];

    /** LM head (unembedding): [vocabSize, dModel] */
    public Tensor $lmHead;

    /** Final RMSNorm weight applied after the last block, before lmHead: [dModel] */
    public Tensor $finalNorm;

    /** Pre-computed sinusoidal positional encoding [maxSeqLen, dModel] (no grad) */
    private Tensor $posEnc;

    /** Attention head dimension = dModel / nHeads */
    private readonly int $headDim;

    public function __construct(
        private readonly int $vocabSize,
        private readonly int $dModel   = D_MODEL,
        private readonly int $nLayers  = N_LAYERS,
        private readonly int $nHeads   = N_HEADS,
        private readonly int $dFF      = D_FF,
        int                  $maxSeqLen = 128,
    ) {
        if ($dModel % $nHeads !== 0) {
            throw new \InvalidArgumentException("dModel ({$dModel}) must be divisible by nHeads ({$nHeads}).");
        }
        $this->headDim = intdiv($dModel, $nHeads);

        // ── Token embedding ───────────────────────────────────────────────
        $embStd         = sqrt(2.0 / $dModel);
        $this->tokenEmb = Tensor::randn([$vocabSize, $dModel], 0.0, $embStd);
        $this->tokenEmb->requiresGrad = true;

        // ── Sinusoidal positional encoding (fixed, no grad) ───────────────
        $this->posEnc = buildSinusoidalPE($maxSeqLen, $dModel);

        // ── Transformer blocks ────────────────────────────────────────────
        for ($l = 0; $l < $nLayers; $l++) {
            $this->layers[$l] = [
                // Attention projections: [dModel, dModel]
                'wq' => $this->initWeight([$dModel, $dModel]),
                'wk' => $this->initWeight([$dModel, $dModel]),
                'wv' => $this->initWeight([$dModel, $dModel]),
                'wo' => $this->initWeight([$dModel, $dModel]),

                // FFN weights
                'w1' => $this->initWeight([$dFF, $dModel]),
                'w2' => $this->initWeight([$dModel, $dFF]),

                // RMSNorm scale weights (pre-attention and pre-FFN)
                // Initialised to ones — identity norm at startup.
                'rms1_w' => $this->initNormWeight($dModel),
                'rms2_w' => $this->initNormWeight($dModel),
            ];
        }

        // ── LM head ───────────────────────────────────────────────────────
        $this->lmHead    = $this->initWeight([$vocabSize, $dModel]);

        // ── Final RMSNorm (after last block, before lmHead) ───────────────
        $this->finalNorm = $this->initNormWeight($dModel);
    }

    /**
     * Full forward pass: token IDs → logits [seqLen, vocabSize].
     * Builds the computational graph when requiresGrad is active.
     *
     * @param int[] $tokenIds Token ID sequence [seqLen].
     */
    public function forward(array $tokenIds): Tensor
    {
        $seqLen = count($tokenIds);

        // ── 1. Token embedding + positional encoding ──────────────────────
        $x        = embeddingWithGrad($this->tokenEmb, $tokenIds);
        $posSlice = new Tensor([$seqLen, $this->dModel], $this->posEnc->buffer);
        $x        = Ops::add($x, $posSlice);

        // ── 2. Transformer blocks ─────────────────────────────────────────
        foreach ($this->layers as $layer) {
            $x = $this->block($x, $layer, $seqLen);
        }

        // ── 3. Final RMSNorm + LM head ────────────────────────────────────
        $x = Ops::rmsNorm($x, $this->finalNorm);
        return Ops::matmul($x, $this->lmHead, false, true); // [seqLen, vocabSize]
    }

    /** Collect all trainable parameter tensors for the optimizer. */
    public function getParams(): array
    {
        $params = [$this->tokenEmb];
        foreach ($this->layers as $layer) {
            foreach ($layer as $w) {
                $params[] = $w;
            }
        }
        $params[] = $this->finalNorm;
        $params[] = $this->lmHead;
        return $params;
    }

    /**
     * Return a named parameter map suitable for SafetensorsWriter::write().
     * @return array<string, Tensor>
     */
    public function namedParams(): array
    {
        $named = ['token_emb' => $this->tokenEmb];
        foreach ($this->layers as $l => $layer) {
            foreach ($layer as $key => $w) {
                $named["layer.{$l}.{$key}"] = $w;
            }
        }
        $named['final_norm'] = $this->finalNorm;
        $named['lm_head']    = $this->lmHead;
        return $named;
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /** He (Kaiming) normal init, requiresGrad=true. */
    private function initWeight(array $shape): Tensor
    {
        $fanIn = $shape[count($shape) - 1];
        $std   = sqrt(2.0 / $fanIn);
        $w     = Tensor::randn($shape, 0.0, $std);
        $w->requiresGrad = true;
        return $w;
    }

    /** Ones-initialised RMSNorm scale weight [d], requiresGrad=true. */
    private function initNormWeight(int $d): Tensor
    {
        $w = Tensor::ones([$d]);
        $w->requiresGrad = true;
        return $w;
    }

    /**
     * One transformer block: pre-norm → MHA → residual → pre-norm → FFN → residual.
     *
     * @param Tensor $x     [seqLen, dModel]
     * @param array  $layer Weight tensors for this block
     * @param int    $seqLen Current sequence length
     * @return Tensor        [seqLen, dModel]
     */
    private function block(Tensor $x, array $layer, int $seqLen): Tensor
    {
        // ── Pre-norm (RMSNorm before attention) ───────────────────────────
        $xNorm = Ops::rmsNorm($x, $layer['rms1_w']);

        // ── QKV projections from normalised input ─────────────────────────
        // All are [seqLen, dModel]; we will slice into per-head pieces below.
        $q = Ops::matmul($xNorm, $layer['wq'], false, true);
        $k = Ops::matmul($xNorm, $layer['wk'], false, true);
        $v = Ops::matmul($xNorm, $layer['wv'], false, true);

        // ── Multi-Head Attention ──────────────────────────────────────────
        //
        // For each head h:
        //   Q_h = Q[:, h*headDim : (h+1)*headDim]   [seqLen, headDim]
        //   K_h = K[:, ...]
        //   V_h = V[:, ...]
        //   S_h = Q_h @ K_h^T / sqrt(headDim)        [seqLen, seqLen]
        //   S_h = causal_mask(S_h)                   [in-place]
        //   A_h = softmax(S_h)                       [seqLen, seqLen]
        //   O_h = A_h @ V_h                           [seqLen, headDim]
        //
        // Ops::sliceCols / concatCols are differentiable:
        //   sliceCols backward → scatter grad to the right column range of Q/K/V grad
        //   concatCols backward → split $out.grad back into per-head grad slices

        $headOutputs = [];
        $scaleFactor = 1.0 / sqrt((float) $this->headDim);

        for ($h = 0; $h < $this->nHeads; $h++) {
            $colStart = $h * $this->headDim;
            $colEnd   = $colStart + $this->headDim;

            $q_h = Ops::sliceCols($q, $colStart, $colEnd); // [seqLen, headDim]
            $k_h = Ops::sliceCols($k, $colStart, $colEnd);
            $v_h = Ops::sliceCols($v, $colStart, $colEnd);

            // Scaled dot-product attention
            $scores_h = Ops::matmul($q_h, $k_h, false, true); // [seqLen, seqLen]
            $scores_h = scaleWithGrad($scores_h, $scaleFactor);
            applyCausalMaskInPlace($scores_h); // in-place, no grad needed
            $attn_h = softmaxRowsWithGrad($scores_h);

            $headOutputs[] = Ops::matmul($attn_h, $v_h); // [seqLen, headDim]
        }

        // Merge heads: [seqLen, headDim × nHeads] = [seqLen, dModel]
        $attnOut = Ops::concatCols($headOutputs);

        // Output projection + residual connection
        $o = Ops::matmul($attnOut, $layer['wo'], false, true); // [seqLen, dModel]
        $x = Ops::add($x, $o);                                 // residual

        // ── Pre-norm (RMSNorm before FFN) ─────────────────────────────────
        $xNorm = Ops::rmsNorm($x, $layer['rms2_w']);

        // ── FFN sub-layer ──────────────────────────────────────────────────
        $h_ff = Ops::matmul($xNorm, $layer['w1'], false, true); // [seqLen, dFF]
        $h_ff = reluWithGrad($h_ff);
        $ff   = Ops::matmul($h_ff, $layer['w2'], false, true);  // [seqLen, dModel]

        return Ops::add($x, $ff); // residual
    }
}
