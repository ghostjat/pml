<?php
declare(strict_types=1);

namespace Pml\Inference;

/**
 * PHP-side model configuration mirror of the C ModelConfig struct.
 *
 * Pass to InferenceSession::load() / InferenceSession::loadFile() when
 * config.json is not present or you want to override detected values.
 *
 * Usage:
 *   $cfg = ModelConfig::llama3(n_layers: 32, d_model: 4096, d_ff: 14336,
 *                               n_heads: 32, n_kv_heads: 8, vocab_size: 128256);
 */
final class ModelConfig
{
    public const ARCH_LLAMA = 0;
    public const ARCH_GPT2  = 1;

    public function __construct(
        public int   $arch        = self::ARCH_LLAMA,
        public int   $vocabSize   = 32000,
        public int   $nLayers     = 32,
        public int   $nHeads      = 32,
        public int   $nKvHeads    = 32,
        public int   $dModel      = 4096,
        public int   $dFf         = 11008,
        public int   $maxSeqLen   = 4096,
        public float $rmsEps      = 1e-5,
        public float $ropeBase    = 10000.0,
        public float $ropeScale   = 1.0,
        public float $attnScale   = 0.0,      // 0 = auto (1/sqrt(head_dim))
        public bool  $tieEmbeddings = false,
        public int   $bosId       = 1,
        public int   $eosId       = 2,
    ) {}

    /** Preset for LLaMA-3 8B */
    public static function llama3_8b(): self
    {
        return new self(
            arch: self::ARCH_LLAMA,
            vocabSize: 128256,
            nLayers: 32,
            nHeads: 32,
            nKvHeads: 8,
            dModel: 4096,
            dFf: 14336,
            maxSeqLen: 8192,
            rmsEps: 1e-5,
            ropeBase: 500000.0,
            bosId: 128000,
            eosId: 128001,
        );
    }

    /** Preset for SmolLM2-135M */
    public static function smollm2_135m(): self
    {
        return new self(
            arch: self::ARCH_LLAMA,
            vocabSize: 49152,
            nLayers: 30,
            nHeads: 9,
            nKvHeads: 3,
            dModel: 576,
            dFf: 1536,
            maxSeqLen: 2048,
            rmsEps: 1e-5,
            ropeBase: 10000.0,
            bosId: 0,
            eosId: 0,
        );
    }

    /** Preset for Mistral-7B */
    public static function mistral_7b(): self
    {
        return new self(
            arch: self::ARCH_LLAMA,
            vocabSize: 32000,
            nLayers: 32,
            nHeads: 32,
            nKvHeads: 8,
            dModel: 4096,
            dFf: 14336,
            maxSeqLen: 8192,
            rmsEps: 1e-5,
            ropeBase: 10000.0,
        );
    }

    /**
     * Build a C ModelConfig struct from this object.
     * @internal used by InferenceSession
     */
    public function toCStruct(\FFI $ffi): \FFI\CData
    {
        $cfg                 = $ffi->new('ModelConfig');
        $cfg->arch           = $this->arch;
        $cfg->vocab_size     = $this->vocabSize;
        $cfg->n_layers       = $this->nLayers;
        $cfg->n_heads        = $this->nHeads;
        $cfg->n_kv_heads     = $this->nKvHeads;
        $cfg->d_model        = $this->dModel;
        $cfg->d_ff           = $this->dFf;
        $cfg->max_seq_len    = $this->maxSeqLen;
        $cfg->rms_eps        = $this->rmsEps;
        $cfg->rope_base      = $this->ropeBase;
        $cfg->rope_scale     = $this->ropeScale;
        $cfg->attn_scale     = $this->attnScale;
        $cfg->tie_embeddings = $this->tieEmbeddings;
        $cfg->bos_id         = $this->bosId;
        $cfg->eos_id         = $this->eosId;
        return $cfg;
    }
}
