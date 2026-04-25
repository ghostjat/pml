<?php
declare(strict_types=1);

namespace Pml\Inference;

use Pml\Tensor;
use Pml\Lib\TensorEngine;

/**
 * LLaMA / Mistral / Phi transformer inference session.
 *
 * Wraps the C InferenceSession* handle.  Weights are mmap'd on load (zero-copy).
 * KV caches and workspace buffers are pre-allocated; no heap alloc per token.
 *
 * Usage:
 *   $tok  = Tokenizer::fromJson('/model/tokenizer.json');
 *   $sess = InferenceSession::load('/model', tok: $tok);
 *
 *   // --- streaming generation ---
 *   $ids = $tok->encode('The capital of France is', addBos: true);
 *   foreach ($sess->generate($ids, maxNewTokens: 50) as $tokenId) {
 *       echo $tok->decode([$tokenId], skipSpecial: false);
 *       flush();
 *   }
 *
 *   // --- batch decode (non-streaming) ---
 *   $outIds = $sess->generateIds($ids, maxNewTokens: 50);
 *   echo $tok->decode($outIds);
 */
final class InferenceSession
{
    /** @var \FFI\CData  InferenceSession* */
    private \FFI\CData $ptr;

    /** Borrowed tokenizer reference (not freed here). */
    private ?Tokenizer $tokenizer;

    /** PHP-side position counter, kept in sync with C sess->pos. */
    private int $pos = 0;

    private function __construct(\FFI\CData $ptr, ?Tokenizer $tok = null)
    {
        $this->ptr       = $ptr;
        $this->tokenizer = $tok;
        $this->pos       = 0;
    }

    public function __destruct()
    {
        self::ffi()->inf_free($this->ptr);
    }

    // ── Factories ─────────────────────────────────────────────────────────────

    /**
     * Load a model from a directory containing *.safetensors and config.json.
     * If $cfg is null, config.json is parsed automatically.
     */
    public static function load(
        string       $modelDir,
        ?ModelConfig $cfg = null,
        ?Tokenizer   $tok = null,
    ): self {
        $ffi    = self::ffi();
        $cCfg   = $cfg?->toCStruct($ffi);
        $cTok   = $tok?->cptr();
        $ptr    = $ffi->inf_load(
            $modelDir,
            $cCfg !== null ? \FFI::addr($cCfg) : null,
            $cTok  ?? null
        );
        self::checkError($ffi);
        if (\FFI::isNull($ptr)) {
            throw new \RuntimeException('[InferenceSession] inf_load returned NULL');
        }
        return new self($ptr, $tok);
    }

    /**
     * Load a model from a single .safetensors file.
     * $cfg is required (no config.json to auto-parse alongside a bare weights file).
     */
    public static function loadFile(
        string      $weightsPath,
        ModelConfig $cfg,
        ?Tokenizer  $tok = null,
    ): self {
        $ffi  = self::ffi();
        $cCfg = $cfg->toCStruct($ffi);
        $cTok = $tok?->cptr();
        $ptr  = $ffi->inf_load_file($weightsPath, \FFI::addr($cCfg), $cTok ?? null);
        self::checkError($ffi);
        if (\FFI::isNull($ptr)) {
            throw new \RuntimeException('[InferenceSession] inf_load_file returned NULL');
        }
        return new self($ptr, $tok);
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    /**
     * Run a single token through the model (incremental / KV-cached).
     * Returns the logit Tensor [vocab_size] — owned by the session,
     * valid until the next step() call.  Do NOT tensor_free() it.
     */
    public function step(int $tokenId, int $pos): Tensor
    {
        $ffi = self::ffi();
        $ptr = $ffi->inf_step($this->ptr, $tokenId, $pos);
        self::checkError($ffi);
        if (\FFI::isNull($ptr)) {
            throw new \RuntimeException('[InferenceSession] inf_step returned NULL');
        }
        $this->pos++;
        return Tensor::wrap($ptr);
    }

    /**
     * Run all prompt tokens through the model (fills KV cache).
     * Returns the logit Tensor for the next token — owned by the session.
     *
     * @param int[] $tokens
     */
    public function forward(array $tokens): Tensor
    {
        $ffi = self::ffi();
        $n   = count($tokens);
        if ($n === 0) throw new \InvalidArgumentException('[InferenceSession] empty token array');
        $buf = $ffi->new("int32_t[{$n}]");
        foreach ($tokens as $i => $t) {
            $buf[$i] = $t;
        }
        $ptr = $ffi->inf_forward($this->ptr, $ffi->cast('const int32_t*', $buf), $n);
        self::checkError($ffi);
        if (\FFI::isNull($ptr)) {
            throw new \RuntimeException('[InferenceSession] inf_forward returned NULL');
        }
        /* C inf_forward calls inf_step N times, each incrementing sess->pos. */
        $this->pos += $n;
        return Tensor::wrap($ptr);
    }

    /** Reset KV caches and position counter — call before a new conversation turn. */
    public function resetKv(): void
    {
        self::ffi()->inf_reset_kv($this->ptr);
        $this->pos = 0;
    }

    // ── Sampling ──────────────────────────────────────────────────────────────

    /** Greedy argmax. */
    public function sampleGreedy(Tensor $logits): int
    {
        return self::ffi()->inf_sample_greedy($logits->ptr);
    }

    /**
     * Temperature + top-p nucleus sampling.
     * temperature = 0.0 → greedy.  top_p = 1.0 → no nucleus filtering.
     */
    public function sample(Tensor $logits, float $temperature = 1.0, float $topP = 0.9): int
    {
        return self::ffi()->inf_sample($this->ptr, $logits->ptr, $temperature, $topP);
    }

    // ── Generation ────────────────────────────────────────────────────────────

    /**
     * Autoregressive generation, yielding one token id per iteration.
     * Stops at EOS or maxNewTokens.
     *
     * @param int[]  $promptIds
     * @return \Generator<int, int, void, void>
     */
    public function generate(
        array $promptIds,
        int   $maxNewTokens = 256,
        float $temperature  = 0.0,
        float $topP         = 0.9,
    ): \Generator {
        $logits = $this->forward($promptIds);
        $ffi    = self::ffi();

        for ($i = 0; $i < $maxNewTokens; $i++) {
            $nextToken = ($temperature <= 0.0)
                ? $ffi->inf_sample_greedy($logits->ptr)
                : $ffi->inf_sample($this->ptr, $logits->ptr, $temperature, $topP);

            yield $nextToken;

            /* Pass the tracked position so C uses the correct RoPE offset. */
            $logits = $this->step($nextToken, $this->pos);
        }
    }

    /**
     * Generate all tokens at once and return them as an array.
     * Uses the optimised C loop (inf_generate_ids) — lower PHP overhead.
     *
     * @param int[] $promptIds
     * @return int[]
     */
    public function generateIds(
        array $promptIds,
        int   $maxNewTokens = 256,
        float $temperature  = 0.0,
        float $topP         = 0.9,
        int   $seed         = 0,
    ): array {
        $ffi = self::ffi();
        $n   = count($promptIds);
        $buf = $ffi->new("int32_t[{$n}]");
        foreach ($promptIds as $i => $t) { $buf[$i] = $t; }

        $out   = $ffi->new("int32_t[{$maxNewTokens}]");
        $nGen  = $ffi->inf_generate_ids(
            $this->ptr,
            $ffi->cast('const int32_t*', $buf),
            $n,
            $maxNewTokens,
            $temperature,
            $topP,
            $seed,
            $ffi->cast('int32_t*', $out)
        );
        self::checkError($ffi);

        $ids = [];
        for ($i = 0; $i < $nGen; $i++) {
            $ids[] = $out[$i];
        }
        return $ids;
    }

    /**
     * High-level text generation: encode → generate → decode.
     *
     * @param string|int[] $prompt  Raw text or pre-encoded token ids.
     */
    public function chat(
        string|array $prompt,
        int   $maxNewTokens = 256,
        float $temperature  = 0.6,
        float $topP         = 0.9,
        int   $seed         = 0,
        bool  $addBos       = true,
    ): string {
        if (!$this->tokenizer) {
            throw new \LogicException('[InferenceSession] no tokenizer attached — pass tok: to load()');
        }
        $promptIds = is_string($prompt)
            ? $this->tokenizer->encode($prompt, $addBos)
            : $prompt;

        $this->resetKv();
        $outIds = $this->generateIds($promptIds, $maxNewTokens, $temperature, $topP, $seed);
        return $this->tokenizer->decode($outIds);
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    /**
     * Retrieve a weight tensor by its safetensors name.
     * Pointer is valid for the session lifetime — do NOT tensor_free().
     */
    public function getWeight(string $name): ?Tensor
    {
        $ffi = self::ffi();
        $ptr = $ffi->inf_get_weight($this->ptr, $name);
        if ($ptr === null || \FFI::isNull($ptr)) return null;
        return Tensor::wrap($ptr);
    }

    /**
     * Parse config.json from a model directory into a ModelConfig object.
     */
    public static function parseConfig(string $configJsonPath): ModelConfig
    {
        $ffi  = self::ffi();
        $cCfg = $ffi->new('ModelConfig');
        $ok   = $ffi->inf_parse_config($configJsonPath, \FFI::addr($cCfg));
        self::checkError($ffi);
        if (!$ok) {
            throw new \RuntimeException("[InferenceSession] Failed to parse {$configJsonPath}");
        }
        return new ModelConfig(
            arch:          $cCfg->arch,
            vocabSize:     $cCfg->vocab_size,
            nLayers:       $cCfg->n_layers,
            nHeads:        $cCfg->n_heads,
            nKvHeads:      $cCfg->n_kv_heads,
            dModel:        $cCfg->d_model,
            dFf:           $cCfg->d_ff,
            maxSeqLen:     $cCfg->max_seq_len,
            rmsEps:        $cCfg->rms_eps,
            ropeBase:      $cCfg->rope_base,
            ropeScale:     $cCfg->rope_scale,
            attnScale:     $cCfg->attn_scale,
            tieEmbeddings: $cCfg->tie_embeddings,
            bosId:         $cCfg->bos_id,
            eosId:         $cCfg->eos_id,
        );
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private static function ffi(): \FFI
    {
        return TensorEngine::get();
    }

    private static function checkError(\FFI $ffi): void
    {
        if ($ffi->tensor_check_error()) {
            $msg = \FFI::string($ffi->tensor_get_last_error());
            $ffi->tensor_clear_error();
            throw new \RuntimeException('[InferenceSession] ' . $msg);
        }
    }
}
