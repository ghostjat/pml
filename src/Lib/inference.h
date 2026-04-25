#ifndef INFERENCE_H
#define INFERENCE_H

/*
 * inference.h  —  LLaMA / GPT-2 style transformer inference engine
 *
 * Design goals:
 *   - Zero heap alloc per forward pass (pre-allocated workspace buffers).
 *   - Zero-copy weight loading via tensor_from_mmap() on safetensors files.
 *   - KV cache for autoregressive generation (reuses existing kvcache_* API).
 *   - Supports LLaMA-2/3/3.x, Mistral, Phi-3 (RMSNorm + RoPE + SwiGLU + GQA).
 *   - Streaming generation via token callback.
 *   - Thread-safe for concurrent InferenceSession instances.
 */

#include "tensor.h"
#include "tokenizer.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Model architecture configuration
 * ========================================================================== */
typedef enum {
    INF_ARCH_LLAMA  = 0,  /* LLaMA-2/3, Mistral, Qwen-2, Phi-3 */
    INF_ARCH_GPT2   = 1,  /* GPT-2 (LayerNorm + learned pos emb) */
} InfArch;

typedef struct {
    InfArch arch;
    int vocab_size;
    int n_layers;
    int n_heads;         /* query heads */
    int n_kv_heads;      /* key/value heads (= n_heads for MHA; < n_heads for GQA) */
    int d_model;         /* embedding dimension (= n_heads * head_dim) */
    int d_ff;            /* feed-forward hidden dimension */
    int max_seq_len;
    float rms_eps;       /* RMSNorm epsilon (e.g. 1e-5) */
    float rope_base;     /* RoPE base frequency (10000.0 LLaMA-2, 500000.0 LLaMA-3) */
    float rope_scale;    /* RoPE scaling factor (1.0 = standard) */
    float attn_scale;    /* softmax scale; 0.0 = 1/sqrt(head_dim) */
    bool  tie_embeddings; /* lm_head shares embedding weights */
    int   bos_id;
    int   eos_id;
} ModelConfig;

/* ============================================================================
 * Model weights (owned by InferenceSession; loaded via mmap)
 * ========================================================================== */
typedef struct {
    /* Token + position embeddings */
    Tensor *tok_emb;      /* [vocab, d_model] */
    Tensor *pos_emb;      /* [max_seq_len, d_model] — GPT-2 only, NULL for LLaMA */

    /* Per-layer weights ([n_layers] arrays) */
    Tensor **rms_attn;    /* [n_layers] × [d_model] */
    Tensor **rms_ffn;     /* [n_layers] × [d_model] */
    Tensor **wq;          /* [n_layers] × [n_heads*head_dim, d_model] */
    Tensor **wk;          /* [n_layers] × [n_kv_heads*head_dim, d_model] */
    Tensor **wv;          /* [n_layers] × [n_kv_heads*head_dim, d_model] */
    Tensor **wo;          /* [n_layers] × [d_model, n_heads*head_dim] */
    Tensor **w_gate;      /* [n_layers] × [d_ff, d_model]  — LLaMA SwiGLU gate */
    Tensor **w_up;        /* [n_layers] × [d_ff, d_model]  — LLaMA SwiGLU up */
    Tensor **w_down;      /* [n_layers] × [d_model, d_ff] */

    /* Output */
    Tensor *rms_final;    /* [d_model] — LLaMA final RMSNorm */
    Tensor *lm_head;      /* [vocab, d_model] (may == tok_emb if tie_embeddings) */

    /* Bias arrays (GPT-2; NULL for LLaMA) */
    Tensor **bq, **bk, **bv, **bo;
    Tensor **b_gate, **b_up, **b_down;

    /* GPT-2 LayerNorm gains + biases */
    Tensor **ln_attn_g, **ln_attn_b;
    Tensor **ln_ffn_g,  **ln_ffn_b;
    Tensor *ln_final_g, *ln_final_b;
} ModelWeights;

/* ============================================================================
 * Workspace: pre-allocated buffers reused across forward passes
 * ========================================================================== */
typedef struct {
    Tensor *x;          /* [1, d_model] current hidden state (single token) */
    Tensor *x_norm;     /* [1, d_model] normed hidden state */
    Tensor *q;          /* [1, head_dim] per-head query */
    Tensor *k_cur;      /* [1, head_dim] per-head key for current token */
    Tensor *v_cur;      /* [1, head_dim] per-head value for current token */
    Tensor *attn_out;   /* [1, head_dim] per-head attention output */
    Tensor *attn_full;  /* [1, n_heads*head_dim] concatenated attn output */
    Tensor *ffn_gate;   /* [1, d_ff] */
    Tensor *ffn_up;     /* [1, d_ff] */
    Tensor *logits;     /* [1, vocab_size] */
} InfWorkspace;

/* ============================================================================
 * Inference session (opaque to PHP)
 * ========================================================================== */
typedef struct InferenceSession InferenceSession;

/* ============================================================================
 * LIFECYCLE
 * ========================================================================== */

/**
 * Load a model from a directory containing:
 *   model.safetensors (or model-00001-of-NNNNN.safetensors, etc.)
 *   config.json (optional — config can be passed directly)
 *
 * @param model_dir  Directory with weight files.
 * @param cfg        Model configuration. If NULL, attempts to read config.json.
 * @param tok        Optional pre-loaded tokenizer (borrowed; not freed by inf).
 * @return Heap-allocated InferenceSession*, or NULL on error.
 */
InferenceSession *inf_load(const char *model_dir,
                            const ModelConfig *cfg,
                            Tokenizer *tok);

/**
 * Load from a single safetensors weight file.
 * Useful for single-file models (e.g. Phi-3-mini, SmolLM, etc.).
 */
InferenceSession *inf_load_file(const char *weights_path,
                                 const ModelConfig *cfg,
                                 Tokenizer *tok);

/**
 * Release all resources (workspace, KV caches).
 * Does NOT free borrowed tensors loaded via tensor_from_mmap (caller calls
 * tensor_mmap_free on each weight if they own them).
 */
void inf_free(InferenceSession *sess);

/* ============================================================================
 * FORWARD PASS
 * ========================================================================== */

/**
 * Run the transformer for a single new token (cached / incremental).
 *
 * @param token_id  The token to process.
 * @param pos       Absolute position index (0-based).
 * @return Tensor [vocab_size] FLOAT32 logits. Owned by the session —
 *         valid until next inf_step() call. Do NOT tensor_free().
 */
Tensor *inf_step(InferenceSession *sess, int32_t token_id, int pos);

/**
 * Run a full prompt through the model (fills KV cache, returns last logits).
 * Equivalent to calling inf_step() for each token in sequence.
 *
 * @param tokens   Array of prompt token ids.
 * @param n_tokens Number of tokens.
 * @return Tensor [vocab_size] logits for next token. Owned by session.
 */
Tensor *inf_forward(InferenceSession *sess,
                     const int32_t *tokens, int n_tokens);

/**
 * Reset KV caches and position counter (start a new context).
 */
void inf_reset_kv(InferenceSession *sess);

/* ============================================================================
 * SAMPLING
 * ========================================================================== */

/**
 * Greedy sampling: argmax over logits.
 */
int32_t inf_sample_greedy(const Tensor *logits);

/**
 * Temperature + top-p (nucleus) sampling.
 * temperature: 1.0 = unscaled; < 1.0 = sharper; 0.0 = greedy.
 * top_p:       0.0 or 1.0 = no nucleus filter; 0.9 = typical.
 */
int32_t inf_sample_top_p(const Tensor *logits, float temperature,
                          float top_p, uint64_t *rng_state);

/* ============================================================================
 * GENERATION
 * ========================================================================== */

/**
 * Sample from logits using the session's internal RNG state.
 * Combines temperature scaling + top-p nucleus sampling.
 * temperature <= 0.0 → greedy argmax.
 */
int32_t inf_sample(InferenceSession *sess, const Tensor *logits,
                    float temperature, float top_p);

/**
 * Convenience: generate token ids into a caller-supplied buffer.
 * Returns the number of tokens generated (may be < max_new_tokens if EOS hit).
 * out_ids must hold at least max_new_tokens int32_t values.
 */
int inf_generate_ids(InferenceSession *sess,
                      const int32_t *prompt_ids, int n_prompt,
                      int max_new_tokens,
                      float temperature, float top_p,
                      uint64_t seed,
                      int32_t *out_ids);

/**
 * Token callback: return true to continue, false to stop.
 */
typedef bool (*InfTokenCallback)(int32_t token_id, const char *token_str,
                                  void *userdata);

/**
 * Full autoregressive generation loop.
 *
 * @param prompt_ids     Array of prompt token ids.
 * @param n_prompt       Number of prompt tokens.
 * @param max_new_tokens Maximum new tokens to generate.
 * @param temperature    Sampling temperature (0.0 = greedy).
 * @param top_p          Nucleus sampling threshold (1.0 = disabled).
 * @param seed           RNG seed (0 = time-based).
 * @param callback       Called for each new token; return false to stop early.
 * @param userdata       Passed through to callback.
 */
void inf_generate(InferenceSession *sess,
                   const int32_t *prompt_ids, int n_prompt,
                   int max_new_tokens,
                   float temperature, float top_p,
                   uint64_t seed,
                   InfTokenCallback callback, void *userdata);

/* ============================================================================
 * CONFIG PARSING
 * ========================================================================== */

/**
 * Parse config.json from a HuggingFace model directory into ModelConfig.
 * Handles LLaMA, Mistral, Phi, Qwen model_type values.
 * Returns false on error.
 */
bool inf_parse_config(const char *config_json_path, ModelConfig *cfg);

/* ============================================================================
 * WEIGHT NAME REGISTRY
 * ========================================================================== */

/**
 * Look up a weight tensor by its safetensors name (e.g. "model.layers.0.self_attn.q_proj.weight").
 * Returns NULL if not found. Pointer valid for session lifetime.
 */
Tensor *inf_get_weight(const InferenceSession *sess, const char *name);

#ifdef __cplusplus
}
#endif
#endif /* INFERENCE_H */
