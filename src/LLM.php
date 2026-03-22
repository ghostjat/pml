<?php

declare(strict_types=1);

namespace Pml;

use Pml\Layers\{
    Embedding, Linear, MultiHeadAttention, FeedForward,
    FeedForwardGELU, TransformerBlock, KVCache
};
use Pml\Generation\{Sampler, GenerationConfig, SimpleTokenizer};
use Pml\IO\SafetensorsLoader;

// ═══════════════════════════════════════════════════════════════════════════
//  MODEL CONFIG
// ═══════════════════════════════════════════════════════════════════════════

final class ModelConfig
{
    public function __construct(
        public readonly int   $vocabSize   = 32000,
        public readonly int   $dModel      = 4096,   // hidden_size
        public readonly int   $nLayers     = 32,
        public readonly int   $nHeads      = 32,
        public readonly int   $nKVHeads    = 8,      // GQA heads (MistralAI/LLaMA-3)
        public readonly int   $dFF         = 14336,  // intermediate_size (SwiGLU)
        public readonly int   $maxSeqLen   = 4096,
        public readonly float $rmsEps      = 1e-5,
        public readonly float $ropeTheta   = 10000.0,
    ) {}

    /** Factory: build from a HuggingFace config.json */
    public static function fromJson(string $jsonPath): self
    {
        $cfg = json_decode(file_get_contents($jsonPath), true, flags: JSON_THROW_ON_ERROR);
        return new self(
            vocabSize:  $cfg['vocab_size']         ?? 32000,
            dModel:     $cfg['hidden_size']         ?? 4096,
            nLayers:    $cfg['num_hidden_layers']   ?? 32,
            nHeads:     $cfg['num_attention_heads'] ?? 32,
            nKVHeads:   $cfg['num_key_value_heads'] ?? ($cfg['num_attention_heads'] ?? 32),
            dFF:        $cfg['intermediate_size']   ?? 14336,
            maxSeqLen:  $cfg['max_position_embeddings'] ?? 4096,
            rmsEps:     (float)($cfg['rms_norm_eps']     ?? 1e-5),
            ropeTheta:  (float)($cfg['rope_theta']       ?? 10000.0),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  LLM — LLaMA / Mistral-style Causal Language Model
// ═══════════════════════════════════════════════════════════════════════════

class LLM
{
    /** @var TransformerBlock[] */
    private array          $blocks;
    private Embedding      $tokenEmbedding;
    private Tensor         $normFinal;      // RMSNorm weight [d_model]
    private Linear         $lmHead;         // [vocab_size, d_model]
    private ModelConfig    $cfg;
    private SimpleTokenizer $tokenizer;

    /** @var KVCache[] One per layer */
    private array $caches = [];

    /**
     * Preferred constructor: load from a directory containing
     * config.json, tokenizer.json, and *.safetensors weight files.
     */
    public static function fromPretrained(string $modelDir): self
    {
        $cfg       = ModelConfig::fromJson("{$modelDir}/config.json");
        $tokenizer = new SimpleTokenizer("{$modelDir}/tokenizer.json");

        // Load weight shards — supports single or multi-shard
        $weights = [];
        foreach (glob("{$modelDir}/*.safetensors") as $shard) {
            $weights = array_merge($weights, SafetensorsLoader::load($shard, verbose: true));
        }

        return self::fromWeights($cfg, $tokenizer, $weights);
    }

    /**
     * Build the model graph from a flat weight dictionary.
     * Keys follow the HuggingFace LLaMA naming convention.
     *
     * @param array<string, Tensor> $weights
     */
    public static function fromWeights(ModelConfig $cfg, SimpleTokenizer $tokenizer, array $weights): self
    {
        $llm            = new self();
        $llm->cfg       = $cfg;
        $llm->tokenizer = $tokenizer;

        // Token embeddings
        $llm->tokenEmbedding = new Embedding($weights['model.embed_tokens.weight']);

        // Final norm
        $llm->normFinal = $weights['model.norm.weight'];

        // LM head (may be tied with embeddings in some models)
        $llmHeadW = $weights['lm_head.weight'] ?? $weights['model.embed_tokens.weight'];
        $llm->lmHead = new Linear($llmHeadW);

        // Transformer layers
        $llm->blocks = [];
        for ($i = 0; $i < $cfg->nLayers; $i++) {
            $p = "model.layers.{$i}";

            $attn = new MultiHeadAttention(
                wq:         $weights["{$p}.self_attn.q_proj.weight"],
                wk:         $weights["{$p}.self_attn.k_proj.weight"],
                wv:         $weights["{$p}.self_attn.v_proj.weight"],
                wo:         $weights["{$p}.self_attn.o_proj.weight"],
                nHeads:     $cfg->nHeads,
                nKVHeads:   $cfg->nKVHeads,
                ropeTheta:  $cfg->ropeTheta,
            );

            $ffn = new FeedForward(
                w1: $weights["{$p}.mlp.gate_proj.weight"],
                w2: $weights["{$p}.mlp.down_proj.weight"],
                w3: $weights["{$p}.mlp.up_proj.weight"],
            );

            $llm->blocks[$i] = new TransformerBlock(
                attention: $attn,
                ffn:       $ffn,
                normAttn:  $weights["{$p}.input_layernorm.weight"],
                normFFN:   $weights["{$p}.post_attention_layernorm.weight"],
                rmsEps:    $cfg->rmsEps,
            );
        }

        return $llm;
    }

    // ── Forward Pass ──────────────────────────────────────────────────────

    /**
     * Full forward pass (prefill mode).
     *
     * @param int[] $tokens
     * @return Tensor  [seq_len, vocab_size] logits
     */
    public function forward(array $tokens): Tensor
    {
        $pos = 0;
        $x   = $this->tokenEmbedding->forward($tokens); // [seq, d_model]

        foreach ($this->blocks as $i => $block) {
            $cache = $this->caches[$i] ?? null;
            $x     = $block->forward($x, $cache, $pos);
        }

        // Final norm
        Ops::rmsNormInPlace($x, $this->normFinal, $this->cfg->rmsEps);

        // LM head: [seq, d_model] → [seq, vocab_size]
        return $this->lmHead->forward($x);
    }

    // ── Generation ────────────────────────────────────────────────────────

    /**
     * Autoregressive text generation with streaming output.
     *
     * @param string           $prompt
     * @param GenerationConfig $config
     * @param callable|null    $onToken  Callback for streaming: fn(string $text, int $tokenId)
     */
    public function generate(
        string           $prompt,
        GenerationConfig $config  = new GenerationConfig(),
        ?callable        $onToken = null
    ): string {
        // Initialise per-layer KV caches
        $this->caches = [];
        for ($i = 0; $i < $this->cfg->nLayers; $i++) {
            $this->caches[$i] = new KVCache($this->cfg->maxSeqLen, $this->cfg->dModel);
        }

        $inputTokens = $this->tokenizer->encode($prompt, addBos: true);
        $output      = [];
        $pos         = 0;

        // ── Prefill: process the prompt in one forward pass ────────────────
        $logits   = $this->forward($inputTokens);
        $seqLen   = count($inputTokens);
        $lastRow  = $logits->getRow($seqLen - 1); // [vocab_size]
        $nextToken = Sampler::sample($lastRow, $config->temperature, $config->topK, $config->topP);

        $pos += $seqLen;
        $output[] = $nextToken;

        if ($config->stream) {
            $text = $this->tokenizer->decodeToken($nextToken);
            echo $text;
            flush();
            if ($onToken !== null) $onToken($text, $nextToken);
        }

        // ── Decode: one token at a time using KV cache ────────────────────
        for ($step = 1; $step < $config->maxNewTokens; $step++) {
            if ($nextToken === $config->eosTokenId) break;

            // Single-token forward pass — KV cache handles the context
            $logits    = $this->forward([$nextToken]);
            $lastRow   = $logits->getRow(0);
            $nextToken = Sampler::sample($lastRow, $config->temperature, $config->topK, $config->topP);

            $output[] = $nextToken;
            $pos++;

            if ($config->stream) {
                $text = $this->tokenizer->decodeToken($nextToken);
                echo $text;
                flush();
                if ($onToken !== null) $onToken($text, $nextToken);
            }
        }

        if ($config->stream) echo "\n";

        return $this->tokenizer->decode($output);
    }

    /**
     * Encode a prompt and return its token embedding matrix.
     * Useful for probing, classification heads, or RAG similarity.
     *
     * @return Tensor [seq_len, d_model]
     */
    public function encode(string $text): Tensor
    {
        $tokens  = $this->tokenizer->encode($text, addBos: true);
        $logits  = $this->forward($tokens);
        // Return the last hidden state (before lmHead)
        // For a pooled embedding, mean over seq dimension:
        $seqLen  = count($tokens);
        $dModel  = $this->cfg->dModel;
        $pooled  = Tensor::zeros([$dModel]);

        for ($i = 0; $i < $seqLen; $i++) {
            $row = $logits->getRow($i);
            Ops::saxpy($row, $pooled, 1.0 / $seqLen);
        }

        return $pooled;
    }
}