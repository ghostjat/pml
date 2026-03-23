<?php

/**
 * train.php — NanoGPT-style training script (v2: production-ready)
 *
 * Trains a tiny causal language model from scratch (or resumes from checkpoint)
 * on a career Q&A dataset.
 *
 * Upgrades over v1:
 *   • True Multi-Head Attention (MHA) via Ops::sliceCols + Ops::concatCols
 *   • RMSNorm (pre-norm, differentiable) via Ops::rmsNorm in each block + final norm
 *   • Global gradient norm clipping (AdamW::clipGradNorm) before every step
 *   • Train / validation split (90/10) with Top-1 token accuracy measurement
 *   • Checkpoint resumption: loads weights + AdamW m/v state if file exists
 *
 * Usage:
 *   php train.php [--steps=500] [--seqlen=64] [--lr=3e-4]
 *                 [--data=datasets/career_counselling_10000.csv]
 *                 [--batchsize=4] [--evalsteps=50] [--checkpoint=career-nano.safetensors]
 *
 * Architecture (TinyGPT v2):
 *   Character-level tokeniser (byte-level, vocabSize ≤ 256)
 *   Token embedding table [vocabSize, dModel]
 *   N transformer blocks, each with:
 *     • Pre-block RMSNorm (learnable [dModel] weight)
 *     • True Multi-Head Attention (nHeads heads, headDim = dModel/nHeads)
 *       Wq, Wk, Wv [dModel×dModel], Wo [dModel×dModel]
 *     • Residual connection
 *     • Pre-FFN RMSNorm (learnable [dModel] weight)
 *     • Two-layer FFN: W1 [dFF×dModel], W2 [dModel×dFF] with ReLU
 *     • Residual connection
 *   Final RMSNorm [dModel] before the LM head
 *   LM head [vocabSize, dModel]
 */

declare(strict_types=1);

require_once __DIR__ . '/vendor/autoload.php';

use Pml\{Tensor, Ops, BlasEngine};
use Pml\Training\{DataLoader, CrossEntropyLoss, AdamW};
use Pml\IO\{SafetensorsLoader, SafetensorsWriter};

// ═══════════════════════════════════════════════════════════════════════════
//  CLI argument parsing
// ═══════════════════════════════════════════════════════════════════════════

$opts = getopt('', ['steps::', 'seqlen::', 'lr::', 'data::', 'batchsize::', 'evalsteps::', 'checkpoint::']);

$nSteps     = (int)   ($opts['steps']      ?? 5000);
$seqLen     = (int)   ($opts['seqlen']     ?? 256);
$lr         = (float) ($opts['lr']         ?? 1e-4);
$dataFile   = (string)($opts['data']       ?? __DIR__ . '/datasets/career_counselling_10000.csv');
$batchSize  = (int)   ($opts['batchsize']  ?? 4);
$evalEvery  = (int)   ($opts['evalsteps']  ?? 50);  // run val eval every N steps
$ckptPath   = (string)($opts['checkpoint'] ?? __DIR__ . '/career-nano.safetensors');

// ── Model hyper-parameters ────────────────────────────────────────────────
const D_MODEL    = 256;
const N_LAYERS   = 6;
const N_HEADS    = 4;          // headDim = D_MODEL / N_HEADS = 32
const D_FF       = 4 * D_MODEL; // 256 — FFN hidden dimension
const SPLIT_RATIO = 0.9;        // 90% train, 10% val

if (D_MODEL % N_HEADS !== 0) {
    fwrite(STDERR, "Error: D_MODEL must be divisible by N_HEADS.\n");
    exit(1);
}

// ═══════════════════════════════════════════════════════════════════════════
//  STEP 0: Byte-level character tokeniser
// ═══════════════════════════════════════════════════════════════════════════

echo "[train] Loading dataset from: {$dataFile}\n";

if (!file_exists($dataFile)) {
    echo "[train] Dataset not found.  Generating a small synthetic corpus for demo...\n";

    $topics  = ['engineering', 'data science', 'software', 'design', 'finance', 'law', 'medicine'];
    $lines   = [];
    for ($i = 0; $i < 200; $i++) {
        $t       = $topics[$i % count($topics)];
        $lines[] = "Q: What career suits me if I like {$t}?\nA: Consider {$t} as a strong career path. Build skills step by step.\n";
    }
    $corpus = implode('', $lines);
} else {
    $fp   = fopen($dataFile, 'r');
    $hdr  = fgetcsv($fp); // skip header
    $rows = [];
    while (($row = fgetcsv($fp)) !== false) {
        if (count($row) >= 2) {
            $rows[] = trim($row[0]) . "\n" . trim($row[1]) . "\n";
        }
    }
    fclose($fp);

    $rows   = array_slice($rows, 0, 1000);
    $corpus = implode('', $rows);

    echo '[train] Loaded ' . count($rows) . " Q&A pairs (" . strlen($corpus) . " chars).\n";
}

// Byte-level tokenisation: char → ord(char) → contiguous ID
$rawBytes  = array_values(unpack('C*', $corpus));
$byteToId  = [];
$idToByte  = [];
foreach ($rawBytes as $b) {
    if (!isset($byteToId[$b])) {
        $id            = count($byteToId);
        $byteToId[$b]  = $id;
        $idToByte[$id] = $b;
    }
}
ksort($byteToId);

$vocabSize = count($byteToId);
$tokens    = array_map(fn(int $b) => $byteToId[$b], $rawBytes);

echo "[train] Vocabulary size: {$vocabSize} unique bytes.\n";
echo "[train] Corpus tokens:   " . count($tokens) . "\n";

// ═══════════════════════════════════════════════════════════════════════════
//  Sinusoidal positional encoding (fixed — no learnable parameters)
// ═══════════════════════════════════════════════════════════════════════════

function buildSinusoidalPE(int $maxSeqLen, int $dModel): Tensor
{
    $pe = Tensor::zeros([$maxSeqLen, $dModel]);
    for ($pos = 0; $pos < $maxSeqLen; $pos++) {
        for ($i = 0; $i < intdiv($dModel, 2); $i++) {
            $angle = $pos / (10000.0 ** (2.0 * $i / $dModel));
            $pe->buffer[$pos * $dModel + 2 * $i]     = (float) sin($angle);
            $pe->buffer[$pos * $dModel + 2 * $i + 1] = (float) cos($angle);
        }
    }
    return $pe->detach(); // requiresGrad = false
}

// ═══════════════════════════════════════════════════════════════════════════
//  Differentiable helper operations
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Embedding lookup with scatter-based backward pass.
 *
 * Forward:  out[i] = weight[ids[i]]   (cblas_scopy per row)
 * Backward: weight.grad[ids[i]] += out.grad[i]   (cblas_saxpy per row)
 *           BLAS has no scatter-add — PHP loop permitted.
 *
 * @param Tensor $weight  [vocabSize, dModel]  requiresGrad=true
 * @param int[]  $ids     Token ID sequence [seqLen]
 * @return Tensor         [seqLen, dModel]
 */
function embeddingWithGrad(Tensor $weight, array $ids): Tensor
{
    $dModel = $weight->shape[1];
    $seqLen = count($ids);
    $out    = Tensor::zeros([$seqLen, $dModel]);
    $ffi    = BlasEngine::get()->ffi;

    for ($i = 0; $i < $seqLen; $i++) {
        $id  = $ids[$i];
        $src = \FFI::cast('float*', \FFI::addr($weight->buffer[$id * $dModel]));
        $dst = \FFI::cast('float*', \FFI::addr($out->buffer[$i * $dModel]));
        $ffi->cblas_scopy($dModel, $src, 1, $dst, 1);
    }

    if ($weight->requiresGrad) {
        $out->requiresGrad = true;
        $out->_prev        = [$weight];
        $capturedIds       = $ids;

        $out->_backward = static function ()
            use ($weight, $out, $capturedIds, $dModel, $seqLen, $ffi): void
        {
            $weight->initGrad();
            for ($i = 0; $i < $seqLen; $i++) {
                $id   = $capturedIds[$i];
                $gSrc = \FFI::cast('float*', \FFI::addr($out->grad[$i * $dModel]));
                $gDst = \FFI::cast('float*', \FFI::addr($weight->grad[$id * $dModel]));
                $ffi->cblas_saxpy($dModel, 1.0, $gSrc, 1, $gDst, 1);
            }
        };
    }

    return $out;
}

/**
 * Element-wise scaling with backward.
 *
 * Forward:  out = x * scale   (cblas_sscal on a copy)
 * Backward: x.grad += out.grad * scale   (cblas_saxpy)
 */
function scaleWithGrad(Tensor $x, float $scale): Tensor
{
    $out = $x->clone();
    BlasEngine::get()->ffi->cblas_sscal($out->size, $scale, $out->buffer, 1);

    if ($x->requiresGrad) {
        $out->requiresGrad = true;
        $out->_prev        = [$x];

        $out->_backward = static function () use ($x, $out, $scale): void {
            $x->initGrad();
            BlasEngine::get()->ffi->cblas_saxpy($out->size, $scale, $out->grad, 1, $x->grad, 1);
        };
    }

    return $out;
}

/**
 * Row-wise softmax with Jacobian-correct backward.
 *
 * Forward (numerically stable):
 *   For each row i: P[i,j] = exp(x[i,j] − max_i) / Σ_k exp(x[i,k] − max_i)
 *
 * Backward (Jacobian multiplication — PHP loop permitted; no BLAS primitive):
 *   dL/dX[i,j] = P[i,j] * (dP[i,j] − Σ_k P[i,k]*dP[i,k])
 */
function softmaxRowsWithGrad(Tensor $x): Tensor
{
    [$M, $N] = $x->shape;
    $out     = Tensor::zeros([$M, $N]);
    $ffi     = BlasEngine::get()->ffi;

    for ($i = 0; $i < $M; $i++) {
        $off  = $i * $N;
        $maxV = (float) $x->buffer[$off];
        for ($j = 1; $j < $N; $j++) {
            $v = (float) $x->buffer[$off + $j];
            if ($v > $maxV) $maxV = $v;
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
        $out->_prev        = [$x];

        $out->_backward = static function () use ($x, $out, $M, $N): void {
            $x->initGrad();
            for ($i = 0; $i < $M; $i++) {
                $off = $i * $N;
                $dot = 0.0;
                for ($j = 0; $j < $N; $j++) {
                    $dot += (float) $out->buffer[$off + $j] * (float) $out->grad[$off + $j];
                }
                for ($j = 0; $j < $N; $j++) {
                    $p  = (float) $out->buffer[$off + $j];
                    $dP = (float) $out->grad[$off + $j];
                    $x->grad[$off + $j] = (float) $x->grad[$off + $j] + $p * ($dP - $dot);
                }
            }
        };
    }

    return $out;
}

/**
 * Element-wise ReLU with gate-mask backward.
 *
 * Forward:  out[i] = max(0, x[i])
 * Backward: x.grad[i] += out.grad[i]  if out.buffer[i] > 0, else 0
 *           (detects gate from forward output — no separate mask needed)
 */
function reluWithGrad(Tensor $x): Tensor
{
    $n   = $x->size;
    $out = Tensor::zeros($x->shape);

    for ($i = 0; $i < $n; $i++) {
        $v = (float) $x->buffer[$i];
        if ($v > 0.0) $out->buffer[$i] = $v;
    }

    if ($x->requiresGrad) {
        $out->requiresGrad = true;
        $out->_prev        = [$x];

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

/**
 * Apply a causal (lower-triangular) mask to a [seqLen, seqLen] score matrix.
 * Sets score[i, j] = -1e9 for j > i (future tokens).
 * In-place, no parameters → no backward needed.
 */
function applyCausalMaskInPlace(Tensor $scores): void
{
    $seqLen = $scores->shape[0];
    for ($i = 0; $i < $seqLen; $i++) {
        for ($j = $i + 1; $j < $seqLen; $j++) {
            $scores->buffer[$i * $seqLen + $j] = -1.0e9;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  TinyGPT v2 — trainable causal language model
//
//  Key changes from v1:
//    • N_HEADS multi-head attention (headDim = D_MODEL / N_HEADS)
//    • RMSNorm before each sub-layer (pre-norm, like GPT-2 / LLaMA)
//    • Final RMSNorm before the LM head
//
//  Per transformer block:
//    x → RMSNorm → [Multi-Head Self-Attention] → residual → x
//    x → RMSNorm → [FFN]                       → residual → x
//
//  Multi-Head Self-Attention (nHeads heads):
//    Q = RMSnorm(x) @ Wq^T  [seqLen, dModel]
//    K = RMSnorm(x) @ Wk^T
//    V = RMSnorm(x) @ Wv^T
//    For head h (columns [h*headDim .. (h+1)*headDim)):
//      Q_h, K_h, V_h = sliceCols(Q|K|V, h*headDim, (h+1)*headDim)
//      S_h = Q_h @ K_h^T / sqrt(headDim)  [seqLen, seqLen]
//      S_h = causal_mask(S_h)              [in-place]
//      A_h = softmax(S_h)                  [seqLen, seqLen]
//      O_h = A_h @ V_h                     [seqLen, headDim]
//    O = concatCols([O_0, O_1, ...])       [seqLen, dModel]
//    x = x + O @ Wo^T
//
//  FFN (ReLU activation):
//    x = x + relu(RMSnorm(x) @ W1^T) @ W2^T
//
//  All weight matrices: He init (std = sqrt(2/fan_in)), requiresGrad=true.
//  RMSNorm weights: ones init, requiresGrad=true.
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
//  Validation / accuracy evaluation
//
//  Runs the model over $nBatches validation batches without building a
//  gradient graph (requiresGrad is temporarily toggled off).
//
//  Returns:
//    'loss' — average cross-entropy loss over all evaluated sequences
//    'acc'  — Top-1 token accuracy: fraction of positions where
//             argmax(logits[t]) == target[t]
//
//  Top-1 accuracy for a language model:
//    At each position t, the model predicts the next token.
//    The prediction is the token with the highest logit (argmax).
//    Accuracy = (# correct predictions) / (# total predictions)
//    A random baseline for vocabSize=90 is ~1.1%; a well-trained model
//    should reach >50% on the training split for a small overfit dataset.
// ═══════════════════════════════════════════════════════════════════════════

function evaluate(
    TinyGPT       $model,
    DataLoader    $loader,
    CrossEntropyLoss $criterion,
    int           $nBatches = 20
): array {
    // ── Disable gradient tracking for evaluation ──────────────────────────
    // Setting requiresGrad=false on all parameters prevents forward() from
    // building a backward graph, saving significant memory and compute.
    // We restore it immediately after.
    $params = $model->getParams();
    foreach ($params as $p) {
        $p->requiresGrad = false;
    }

    $totalLoss    = 0.0;
    $totalCorrect = 0;
    $totalTokens  = 0;
    $totalSeqs    = 0;

    for ($b = 0; $b < $nBatches; $b++) {
        [$xBatch, $yBatch] = $loader->getBatch('val');

        foreach ($xBatch as $idx => $xSeq) {
            $ySeq   = $yBatch[$idx];
            $logits = $model->forward($xSeq);  // [seqLen, vocabSize] — no grad

            $loss         = $criterion->forward($logits, $ySeq);
            $totalLoss   += (float) $loss->buffer[0];
            $totalSeqs++;

            // ── Top-1 accuracy ────────────────────────────────────────────
            //
            // For each time step t in [0, seqLen):
            //   pred = argmax(logits[t, :])   (the most likely next token)
            //   correct += (pred == ySeq[t])
            //
            // We search linearly: vocabSize is typically small (≤256) and
            // this only runs during evaluation, not the training hot path.
            $seqLen    = count($ySeq);
            $vocabSize = $logits->shape[1];

            for ($t = 0; $t < $seqLen; $t++) {
                $off    = $t * $vocabSize;
                $argmax = 0;
                $maxVal = (float) $logits->buffer[$off];

                for ($v = 1; $v < $vocabSize; $v++) {
                    $val = (float) $logits->buffer[$off + $v];
                    if ($val > $maxVal) {
                        $maxVal = $val;
                        $argmax = $v;
                    }
                }

                if ($argmax === $ySeq[$t]) {
                    $totalCorrect++;
                }
                $totalTokens++;
            }
        }
    }

    // ── Restore gradient tracking ─────────────────────────────────────────
    foreach ($params as $p) {
        $p->requiresGrad = true;
    }

    return [
        'loss' => $totalSeqs > 0 ? $totalLoss / $totalSeqs : 0.0,
        'acc'  => $totalTokens > 0 ? $totalCorrect / $totalTokens : 0.0,
    ];
}

// ═══════════════════════════════════════════════════════════════════════════
//  STEP 1: Instantiate model, data loader, loss, optimizer
// ═══════════════════════════════════════════════════════════════════════════

echo "\n[train] Initialising TinyGPT v2 "
     . "(vocabSize={$vocabSize}, dModel=" . D_MODEL
     . ", nLayers=" . N_LAYERS
     . ", nHeads=" . N_HEADS
     . ", headDim=" . intdiv(D_MODEL, N_HEADS)
     . ", dFF=" . D_FF . ")...\n";

$model = new TinyGPT(
    vocabSize:  $vocabSize,
    maxSeqLen:  $seqLen + 16,
);

$totalParams = 0;
foreach ($model->getParams() as $p) {
    $totalParams += $p->size;
}
$totalMiB = number_format($totalParams * 4 / 1_048_576, 2);
echo "[train] Total parameters: " . number_format($totalParams) . "  ({$totalMiB} MiB F32)\n";

$loader    = new DataLoader($tokens, $seqLen, $batchSize, SPLIT_RATIO);
$criterion = new CrossEntropyLoss();
$optimizer = new AdamW(
    params:      $model->getParams(),
    lr:          $lr,
    beta1:       0.9,
    beta2:       0.999,
    eps:         1e-8,
    weightDecay: 0.1,
);

echo "[train] Train windows: " . $loader->datasetSize()
     . "  Val windows: " . $loader->valSize()
     . "  (seqLen={$seqLen}, splitRatio=" . SPLIT_RATIO . ")\n";

// ═══════════════════════════════════════════════════════════════════════════
//  STEP 2: Checkpoint resumption (Task 5)
//
//  If career-nano.safetensors exists, load the weights into the freshly
//  initialised model AND restore the AdamW moment buffers and step counter.
//
//  Without restoring m/v buffers, AdamW would re-start its warm-up phase
//  (large effective lr at t=1 due to bias correction) and the training
//  curve would stall or spike after the resume.
// ═══════════════════════════════════════════════════════════════════════════

$resumeStep = 0;

if (file_exists($ckptPath)) {
    echo "\n[train] Found checkpoint at {$ckptPath} — resuming.\n";

    $loadedTensors = SafetensorsLoader::load($ckptPath, verbose: false);
    $namedParams   = $model->namedParams();
    $ffi           = BlasEngine::get()->ffi;

    // ── Load model weights ────────────────────────────────────────────────
    $loadedCount = 0;
    foreach ($namedParams as $name => $param) {
        if (isset($loadedTensors[$name])) {
            // cblas_scopy: element-wise copy from loaded buffer → param buffer
            $ffi->cblas_scopy($param->size, $loadedTensors[$name]->buffer, 1, $param->buffer, 1);
            $loadedCount++;
        }
    }
    echo "[train] Restored {$loadedCount}/" . count($namedParams) . " weight tensors.\n";

    // ── Load optimizer m/v state ──────────────────────────────────────────
    $optimizer->loadNamedState($loadedTensors, $namedParams);

    // ── Restore step counter ──────────────────────────────────────────────
    // Saved as a 1-element Tensor named '__opt_step__'.
    if (isset($loadedTensors['__opt_step__'])) {
        $resumeStep = (int) (float) $loadedTensors['__opt_step__']->buffer[0];
        $optimizer->setStep($resumeStep);
        echo "[train] Resuming from step {$resumeStep} (AdamW bias correction restored).\n";
    }
} else {
    echo "\n[train] No checkpoint found — training from scratch.\n";
}

echo "[train] Starting training for {$nSteps} steps"
     . ($resumeStep > 0 ? " (steps {$resumeStep}+1 .. " . ($resumeStep + $nSteps) . ")" : "")
     . "...\n\n";

// ═══════════════════════════════════════════════════════════════════════════
//  STEP 3: Training loop
//
//  Per-step sequence:
//   1. $optimizer->zeroGrad()           — clear previous step's accumulated grads
//   2. $model->forward($X)             — build the computational graph
//   3. $criterion->forward($logits, $Y) — fused softmax + NLL loss
//   4. $loss->backward()               — reverse-mode autodiff (accumulates)
//   5. AdamW::clipGradNorm($params)    — clip global grad norm to 1.0 (Task 4)
//   6. $optimizer->step()              — AdamW weight update
//
//  Gradient accumulation: forward+backward are called $batchSize times before
//  a single step(), so each step sees the summed gradient from the full batch.
// ═══════════════════════════════════════════════════════════════════════════

$startTime    = microtime(true);
$logFrequency = 10;
$avgLoss      = 0.0; // kept in scope for the final checkpoint metadata

for ($step = 1; $step <= $nSteps; $step++) {

    // ── 1. Zero gradients for this step ───────────────────────────────────
    $optimizer->zeroGrad();

    // ── 2–4. Forward + loss + backward (one sequence per batch item) ──────
    [$xBatch, $yBatch] = $loader->getBatch('train');

    $totalLoss = 0.0;

    foreach ($xBatch as $b => $xSeq) {
        $ySeq = $yBatch[$b];

        // Build graph and compute logits
        $logits = $model->forward($xSeq);

        // Fused softmax + NLL loss
        $loss = $criterion->forward($logits, $ySeq);

        // Backward: accumulates grad into all param->grad buffers
        $loss->backward();

        $totalLoss += (float) $loss->buffer[0];
    }

    $avgLoss = $totalLoss / $batchSize;

    // ── 5. Gradient norm clipping (global, before AdamW update) ──────────
    //
    // Clips the entire gradient vector to have L2 norm ≤ 1.0.
    // Prevents gradient explosions that can derail training, especially
    // during the early steps when the model makes large random predictions.
    $gradNorm = AdamW::clipGradNorm($model->getParams(), maxNorm: 1.0);

    // ── 6. AdamW parameter update ─────────────────────────────────────────
    $optimizer->step();

    // ── Logging ───────────────────────────────────────────────────────────
    if ($step % $logFrequency === 0 || $step === 1) {
        $elapsed   = microtime(true) - $startTime;
        $stepsPerS = $step / max($elapsed, 1e-9);
        $eta       = ($nSteps - $step) / max($stepsPerS, 1e-9);

        printf(
            "step %4d/%d  loss=%.4f  gnorm=%.3f  %.1f steps/s  ETA %.0fs\n",
            $step, $nSteps, $avgLoss, $gradNorm, $stepsPerS, $eta
        );
    }

    // ── Periodic validation evaluation ────────────────────────────────────
    //
    // Runs evaluate() on the val split every $evalEvery steps.
    // requiresGrad is toggled off inside evaluate() to save memory.
    // Top-1 accuracy is measured here: a char-level model memorising
    // the training set should eventually reach >60% accuracy on train data.
    if ($loader->valSize() > 0 && $step % $evalEvery === 0) {
        $metrics = evaluate($model, $loader, $criterion, nBatches: 20);
        printf(
            "  [eval]  step %d  val_loss=%.4f  val_acc=%.2f%%\n",
            $step,
            $metrics['loss'],
            $metrics['acc'] * 100.0
        );
    }
}

$elapsed = microtime(true) - $startTime;
echo "\n[train] Training complete. {$nSteps} steps in " . number_format($elapsed, 1) . "s.\n";

// ═══════════════════════════════════════════════════════════════════════════
//  STEP 4: Final evaluation before checkpoint save
// ═══════════════════════════════════════════════════════════════════════════

if ($loader->valSize() > 0) {
    echo "\n[train] Running final validation evaluation...\n";
    $finalMetrics = evaluate($model, $loader, $criterion, nBatches: 40);
    printf(
        "[eval]  FINAL  val_loss=%.4f  val_acc=%.2f%%\n",
        $finalMetrics['loss'],
        $finalMetrics['acc'] * 100.0
    );
} else {
    $finalMetrics = ['loss' => 0.0, 'acc' => 0.0];
}

// ═══════════════════════════════════════════════════════════════════════════
//  STEP 5: Save checkpoint (weights + optimizer state)
//
//  We save:
//    - All model weight tensors (namedParams())
//    - Optimizer m/v moment buffers (getNamedState()) — Task 5
//    - A 1-element '__opt_step__' Tensor holding the current step count,
//      used to restore the AdamW bias-correction state on the next resume
//
//  On the NEXT run, if this file exists, all of the above are loaded back
//  in STEP 2 above.
// ═══════════════════════════════════════════════════════════════════════════

echo "\n[train] Saving checkpoint to: {$ckptPath}\n";

$namedParams  = $model->namedParams();
$optState     = $optimizer->getNamedState($namedParams);
$optStepTensor = Tensor::full([1], (float) ($resumeStep + $nSteps));

$allTensors = array_merge(
    $namedParams,
    $optState,
    ['__opt_step__' => $optStepTensor]
);

SafetensorsWriter::write(
    $ckptPath,
    $allTensors,
    [
        'arch'       => 'TinyGPT-v2',
        'dModel'     => (string) D_MODEL,
        'nLayers'    => (string) N_LAYERS,
        'nHeads'     => (string) N_HEADS,
        'dFF'        => (string) D_FF,
        'vocabSize'  => (string) $vocabSize,
        'seqLen'     => (string) $seqLen,
        'totalSteps' => (string) ($resumeStep + $nSteps),
        'lr'         => (string) $lr,
        'finalLoss'  => number_format($avgLoss, 6),
        'valLoss'    => number_format($finalMetrics['loss'], 6),
        'valAcc'     => number_format($finalMetrics['acc'], 6),
    ]
);

$ckptBytes = file_exists($ckptPath) ? filesize($ckptPath) : 0;
$ckptMiB   = number_format($ckptBytes / 1_048_576, 2);
echo "[train] Checkpoint saved ({$ckptMiB} MiB).  Done.\n";
