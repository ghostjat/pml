#ifndef TOKENIZER_H
#define TOKENIZER_H

/*
 * tokenizer.h  —  Byte-level BPE tokenizer (GPT-2 / LLaMA / Mistral compatible)
 *
 * Design goals:
 *   - Zero heap alloc in the BPE hot path (stack work buffer per segment).
 *   - Open-addressing hash tables for O(1) vocab and merge lookups.
 *   - Contiguous string pool: one allocation for all token strings.
 *   - Thread-safe encode/decode (no mutable state after construction).
 *   - HuggingFace tokenizer.json or vocab.json + merges.txt loading.
 *   - OpenMP parallel batch encoding.
 */

#include "tensor.h"   /* Tensor*, tensor error reporting */
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Opaque tokenizer handle.
 * ========================================================================== */
typedef struct Tokenizer Tokenizer;

/* ============================================================================
 * LIFECYCLE
 * ========================================================================== */

/**
 * Load from HuggingFace tokenizer.json (BPE byte-level models).
 * Handles GPT-2, LLaMA-2/3, Mistral, Phi, Qwen, etc.
 * Returns NULL on error (error set via tensor error system).
 */
Tokenizer *tok_load_json(const char *tokenizer_json_path);

/**
 * Load from separate vocab.json + merges.txt (legacy GPT-2 style).
 * vocab.json: flat {"<token_string>": id, ...}
 * merges.txt: lines of "token_a token_b" in priority order.
 * Returns NULL on error.
 */
Tokenizer *tok_load(const char *vocab_path, const char *merges_path);

/**
 * Release all resources. Safe to call with NULL.
 */
void tok_free(Tokenizer *tok);

/* ============================================================================
 * ENCODING
 * ========================================================================== */

/**
 * Encode UTF-8 text to token ids.
 *
 * @param add_bos  Prepend BOS token if bos_id >= 0.
 * @param n_out    Receives token count.
 * @return Heap-allocated int32_t[*n_out]; caller must free().
 *         Returns NULL on error.
 */
int32_t *tok_encode(const Tokenizer *tok, const char *text,
                    bool add_bos, int *n_out);

/**
 * Batch encode n_texts strings using OpenMP.
 *
 * @param max_len  Pad/truncate to this length. If <= 0, uses the max token
 *                 count across all texts (natural length, still padded to max).
 * @return Newly allocated Tensor [n_texts × effective_max_len] INT32,
 *         left-aligned, right-padded with pad_id. NULL on error.
 */
Tensor *tok_encode_batch(const Tokenizer *tok,
                          const char **texts, int n_texts,
                          bool add_bos, int max_len);

/* ============================================================================
 * DECODING
 * ========================================================================== */

/**
 * Decode token ids back to UTF-8 text.
 *
 * @param skip_special  If true, skip tokens with the special-token flag.
 * @return Heap-allocated NUL-terminated UTF-8 string; caller must free().
 *         Returns NULL on error.
 */
char *tok_decode(const Tokenizer *tok, const int32_t *ids, int n,
                 bool skip_special);

/* ============================================================================
 * SINGLE-TOKEN ACCESSORS
 * ========================================================================== */

/** id → token string. Pointer valid for tokenizer lifetime. NULL if out of range. */
const char *tok_id_to_str(const Tokenizer *tok, int id);

/** token string → id. Returns -1 if not found. */
int tok_str_to_id(const Tokenizer *tok, const char *str);

/** True if the token is a special token (BOS, EOS, PAD, etc.). */
bool tok_is_special(const Tokenizer *tok, int id);

/* ============================================================================
 * ACCESSORS
 * ========================================================================== */
int tok_vocab_size(const Tokenizer *tok);
int tok_bos_id(const Tokenizer *tok);
int tok_eos_id(const Tokenizer *tok);
int tok_pad_id(const Tokenizer *tok);
int tok_unk_id(const Tokenizer *tok);

#ifdef __cplusplus
}
#endif
#endif /* TOKENIZER_H */
