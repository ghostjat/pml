<?php
declare(strict_types=1);

namespace Pml\Benchmarks\NeuralNetwork;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\NeuralNetwork\Layers\CausalSelfAttention;
use Pml\KVCache;

/**
 * Attention kernel benchmarks — measures the cost of the three forward paths
 * in CausalSelfAttention: training (full O(T²)), prefill (full+cache write),
 * and decode (T=1, KV-cache attend only).
 *
 * Also benchmarks raw tensor operations that underlie attention:
 *   - scaled dot-product attention
 *   - softmax on attention scores
 *   - matrix-vector product (decode step)
 *
 * Architecture parameters are chosen to match realistic LLM sizes:
 *   - smollm2_135m:  nHeads=9,  nKvHeads=3,  headDim=64,   dModel=576
 *   - llama3_1b:     nHeads=32, nKvHeads=8,  headDim=64,   dModel=2048
 *   - mistral_7b:    nHeads=32, nKvHeads=8,  headDim=128,  dModel=4096
 *
 * Groups:
 *   attention    — all attention benchmarks
 *   prefill      — prompt processing (full sequence, O(T²))
 *   decode       — single token decode (KV-cache attend, O(T))
 *   training     — training forward pass (no cache)
 *   kernels      — individual attention sub-operations
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['attention', 'nn'])]
final class AttentionBench
{
    // smollm2_135m config — small enough to run many revs
    private const SMALL_HEADS    = 9;
    private const SMALL_KVHEADS  = 3;
    private const SMALL_HEAD_DIM = 64;
    private const SMALL_DMODEL   = 576;   // SMALL_HEADS * SMALL_HEAD_DIM

    // mistral-7b-like config — large, realistic
    private const LARGE_HEADS    = 32;
    private const LARGE_KVHEADS  = 8;
    private const LARGE_HEAD_DIM = 128;
    private const LARGE_DMODEL   = 4096;

    // Sequence lengths for prefill benchmarks
    private const SEQ_128  = 128;
    private const SEQ_512  = 512;
    private const SEQ_1024 = 1024;
    private const SEQ_2048 = 2048;

    // Batch sizes for training
    private const BATCH_TRAIN = 8;

    private static CausalSelfAttention $smallAttn;
    private static CausalSelfAttention $largeAttn;

    // Pre-built input tensors [batch_or_1, seq, dmodel]
    private static Tensor $inputSmallSeq128;    // [1, 128, 576]
    private static Tensor $inputSmallSeq512;    // [1, 512, 576]
    private static Tensor $inputSmallSeq1024;   // [1, 1024, 576]
    private static Tensor $inputSmallSeq2048;   // [1, 2048, 576]
    private static Tensor $inputSmallDecode;    // [1, 1, 576]

    private static Tensor $inputLargeSeq128;    // [1, 128, 4096]
    private static Tensor $inputLargeSeq512;    // [1, 512, 4096]
    private static Tensor $inputLargeSeq1024;   // [1, 1024, 4096]
    private static Tensor $inputLargeSeq2048;   // [1, 2048, 4096]
    private static Tensor $inputLargeDecode;    // [1, 1, 4096]

    private static Tensor $inputTrainBatch;     // [8, 128, 576]

    // KV caches — pre-warmed with 128-token context
    private static KVCache $smallKvCache;
    private static KVCache $largeKvCache;

    // Raw attention kernel tensors
    private static Tensor $qSmall128;           // Q for sdpa: [1, nHeads, 128, headDim]
    private static Tensor $kSmall128;           // K: [1, nKvHeads, 128, headDim]
    private static Tensor $vSmall128;           // V: [1, nKvHeads, 128, headDim]

    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }

        // Build attention layers
        self::$smallAttn = new CausalSelfAttention(
            dModel:   self::SMALL_DMODEL,
            nHeads:   self::SMALL_HEADS,
            nKvHeads: self::SMALL_KVHEADS,
        );
        self::$largeAttn = new CausalSelfAttention(
            dModel:   self::LARGE_DMODEL,
            nHeads:   self::LARGE_HEADS,
            nKvHeads: self::LARGE_KVHEADS,
        );

        // Input tensors
        self::$inputSmallSeq128  = Tensor::randomNormal([1, self::SEQ_128,  self::SMALL_DMODEL]);
        self::$inputSmallSeq512  = Tensor::randomNormal([1, self::SEQ_512,  self::SMALL_DMODEL]);
        self::$inputSmallSeq1024 = Tensor::randomNormal([1, self::SEQ_1024, self::SMALL_DMODEL]);
        self::$inputSmallSeq2048 = Tensor::randomNormal([1, self::SEQ_2048, self::SMALL_DMODEL]);
        self::$inputSmallDecode  = Tensor::randomNormal([1, 1,              self::SMALL_DMODEL]);

        self::$inputLargeSeq128  = Tensor::randomNormal([1, self::SEQ_128,  self::LARGE_DMODEL]);
        self::$inputLargeSeq512  = Tensor::randomNormal([1, self::SEQ_512,  self::LARGE_DMODEL]);
        self::$inputLargeSeq1024 = Tensor::randomNormal([1, self::SEQ_1024, self::LARGE_DMODEL]);
        self::$inputLargeSeq2048 = Tensor::randomNormal([1, self::SEQ_2048, self::LARGE_DMODEL]);
        self::$inputLargeDecode  = Tensor::randomNormal([1, 1,              self::LARGE_DMODEL]);

        self::$inputTrainBatch   = Tensor::randomNormal([self::BATCH_TRAIN, self::SEQ_128, self::SMALL_DMODEL]);

        // Raw QKV tensors for kernel-level benchmarks
        self::$qSmall128 = Tensor::randomNormal([1, self::SMALL_HEADS,   self::SEQ_128, self::SMALL_HEAD_DIM]);
        self::$kSmall128 = Tensor::randomNormal([1, self::SMALL_KVHEADS, self::SEQ_128, self::SMALL_HEAD_DIM]);
        self::$vSmall128 = Tensor::randomNormal([1, self::SMALL_KVHEADS, self::SEQ_128, self::SMALL_HEAD_DIM]);

        // Build KV caches and prefill with 128-token context
        self::$smallKvCache = new KVCache(
            nLayers:   1,
            nHeads:    self::SMALL_KVHEADS,
            maxSeqLen: self::SEQ_2048,
            headDim:   self::SMALL_HEAD_DIM,
        );
        self::$largeKvCache = new KVCache(
            nLayers:   1,
            nHeads:    self::LARGE_KVHEADS,
            maxSeqLen: self::SEQ_2048,
            headDim:   self::LARGE_HEAD_DIM,
        );

        // Prefill both caches with 128-token context
        self::$smallAttn->forward(self::$inputSmallSeq128, self::$smallKvCache, layerIdx: 0);
        self::$largeAttn->forward(self::$inputLargeSeq128, self::$largeKvCache, layerIdx: 0);

        self::$initialized = true;
    }

    // =========================================================================
    // PREFILL (training-style, no cache) — measures full O(T²) attention cost
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['attention', 'prefill'])]
    public function benchPrefillSmall_128(): void
    {
        $out = self::$smallAttn->forward(self::$inputSmallSeq128);
        unset($out);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['attention', 'prefill'])]
    public function benchPrefillSmall_512(): void
    {
        $out = self::$smallAttn->forward(self::$inputSmallSeq512);
        unset($out);
    }

    #[Bench\Iterations(3), Bench\Revs(2)]
    #[Bench\Groups(['attention', 'prefill'])]
    public function benchPrefillSmall_1024(): void
    {
        $out = self::$smallAttn->forward(self::$inputSmallSeq1024);
        unset($out);
    }

    #[Bench\Iterations(3), Bench\Revs(1)]
    #[Bench\Groups(['attention', 'prefill'])]
    public function benchPrefillSmall_2048(): void
    {
        $out = self::$smallAttn->forward(self::$inputSmallSeq2048);
        unset($out);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['attention', 'prefill'])]
    public function benchPrefillLarge_128(): void
    {
        $out = self::$largeAttn->forward(self::$inputLargeSeq128);
        unset($out);
    }

    #[Bench\Iterations(3), Bench\Revs(1)]
    #[Bench\Groups(['attention', 'prefill'])]
    public function benchPrefillLarge_512(): void
    {
        $out = self::$largeAttn->forward(self::$inputLargeSeq512);
        unset($out);
    }

    #[Bench\Iterations(3), Bench\Revs(1)]
    #[Bench\Groups(['attention', 'prefill'])]
    public function benchPrefillLarge_1024(): void
    {
        $out = self::$largeAttn->forward(self::$inputLargeSeq1024);
        unset($out);
    }

    // =========================================================================
    // DECODE (T=1, KV-cache attend) — measures single-token generation speed
    //
    // The cache is pre-warmed with 128 tokens. Each decode step appends
    // one token. Cache grows by 1 per call — benchmark should be run with
    // few revs to avoid filling the cache.
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['attention', 'decode'])]
    public function benchDecodeSmall_ctx128(): void
    {
        // Each call appends 1 token to the 128-token cache
        $out = self::$smallAttn->forward(self::$inputSmallDecode, self::$smallKvCache, layerIdx: 0);
        unset($out);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['attention', 'decode'])]
    public function benchDecodeLarge_ctx128(): void
    {
        $out = self::$largeAttn->forward(self::$inputLargeDecode, self::$largeKvCache, layerIdx: 0);
        unset($out);
    }

    // =========================================================================
    // TRAINING FORWARD (batched, no cache)
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['attention', 'training'])]
    public function benchTrainingBatch8_seq128_small(): void
    {
        $out = self::$smallAttn->forward(self::$inputTrainBatch);
        unset($out);
    }

    // =========================================================================
    // KV-CACHE MEMORY FOOTPRINT
    //
    // Documents memory cost for different context lengths and model sizes.
    // Formula: nLayers × nKvHeads × maxSeqLen × 2 × headDim × 4 bytes
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(1)]
    #[Bench\Groups(['attention', 'kvmem'])]
    public function benchKvCacheAlloc_smollm2_135m_ctx2048(): void
    {
        // 1 layer × 3 kv-heads × 2048 ctx × 2 (K+V) × 64 head_dim × 4 bytes = 3 MB
        $kv = new KVCache(nLayers: 28, nHeads: 3, maxSeqLen: 2048, headDim: 64);
        $mem = $kv->memoryBytes();
        unset($kv);
    }

    #[Bench\Iterations(3), Bench\Revs(1)]
    #[Bench\Groups(['attention', 'kvmem'])]
    public function benchKvCacheAlloc_mistral7b_ctx4096(): void
    {
        // 32 layers × 8 kv-heads × 4096 ctx × 2 × 128 head_dim × 4 bytes = 1 GB
        $kv = new KVCache(nLayers: 32, nHeads: 8, maxSeqLen: 4096, headDim: 128);
        $mem = $kv->memoryBytes();
        unset($kv);
    }
}
