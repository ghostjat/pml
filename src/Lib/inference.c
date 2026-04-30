/*
 * inference.c  —  LLaMA-style transformer inference engine
 *
 * Supports: LLaMA-2/3, Mistral, Qwen-2, Phi-3, SmolLM (RMSNorm+RoPE+SwiGLU+GQA)
 * Weight format: safetensors (single-file or sharded; F32 only)
 *
 * Allocation profile per inf_step() call:
 *   - 0 allocations in embedding, projection, residual, and attention paths
 *     (all workspace tensors are pre-allocated and reused via data pointer swaps).
 *   - 0 allocations in SwiGLU (implemented as inline fused kernel on gate buffer).
 *   - Safetensors weights are mmap'd (zero-copy; kernel pages in on first access).
 */

#include "inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * Error helpers
 * ========================================================================== */
#define INF_ERR(msg)        tensor_set_error("[inference] " msg)
#define INF_ERRF(fmt, ...) do {                          \
    char _eb[512];                                        \
    snprintf(_eb, sizeof(_eb), "[inference] " fmt,        \
             ##__VA_ARGS__);                              \
    tensor_set_error(_eb);                                \
} while (0)

/* ============================================================================
 * 1.  WEIGHT REGISTRY
 * ========================================================================== */
#define WREG_LOAD_FACTOR  0.60

typedef struct {
    char    *name;    /* heap-allocated name string */
    Tensor  *tensor;  /* mmap'd tensor (tensor_mmap_free on cleanup) */
    bool     mmap;    /* true = owned via tensor_mmap_free(); false = plain tensor_free() */
} WEntry;

typedef struct {
    WEntry *slots;
    int     cap;
    int     size;
} WRegistry;

static uint32_t _wreg_hash(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) { h ^= (uint8_t)*s++; h *= 16777619u; }
    return h;
}

static bool wreg_insert(WRegistry *r, const char *name, Tensor *t, bool mmap_owned) {
    if (!r->slots || r->size >= (int)(r->cap * WREG_LOAD_FACTOR)) {
        int ncap = r->cap ? r->cap * 2 : 256;
        WEntry *ns = calloc((size_t)ncap, sizeof(WEntry));
        if (!ns) return false;
        /* Rehash */
        for (int i = 0; i < r->cap; i++) {
            if (!r->slots[i].name) continue;
            uint32_t h = _wreg_hash(r->slots[i].name);
            int idx = (int)(h & (uint32_t)(ncap - 1));
            while (ns[idx].name) idx = (idx + 1) & (ncap - 1);
            ns[idx] = r->slots[i];
        }
        free(r->slots);
        r->slots = ns; r->cap = ncap;
    }
    uint32_t h = _wreg_hash(name);
    int mask = r->cap - 1;
    int idx  = (int)(h & (uint32_t)mask);
    for (;;) {
        if (!r->slots[idx].name) {
            r->slots[idx].name   = strdup(name);
            r->slots[idx].tensor = t;
            r->slots[idx].mmap   = mmap_owned;
            r->size++;
            return true;
        }
        if (strcmp(r->slots[idx].name, name) == 0) {
            /* Update existing */
            if (r->slots[idx].mmap) tensor_mmap_free(r->slots[idx].tensor);
            else                     tensor_free(r->slots[idx].tensor);
            r->slots[idx].tensor = t;
            r->slots[idx].mmap   = mmap_owned;
            return true;
        }
        idx = (idx + 1) & mask;
    }
}

static Tensor *wreg_get(const WRegistry *r, const char *name) {
    if (!r->slots || !name) return NULL;
    uint32_t h    = _wreg_hash(name);
    int      mask = r->cap - 1;
    int      idx  = (int)(h & (uint32_t)mask);
    for (int i = 0; i < r->cap; i++) {
        if (!r->slots[idx].name) return NULL;
        if (strcmp(r->slots[idx].name, name) == 0) return r->slots[idx].tensor;
        idx = (idx + 1) & mask;
    }
    return NULL;
}

static void wreg_free(WRegistry *r) {
    if (!r->slots) return;
    for (int i = 0; i < r->cap; i++) {
        if (!r->slots[i].name) continue;
        free(r->slots[i].name);
        if (r->slots[i].mmap) tensor_mmap_free(r->slots[i].tensor);
        else                   tensor_free(r->slots[i].tensor);
    }
    free(r->slots);
    r->slots = NULL; r->cap = r->size = 0;
}

/* ============================================================================
 * 2.  SAFETENSORS PARSER
 *     8-byte LE header_len, then header_len bytes of JSON, then raw data.
 *     Per tensor: {"dtype":"F32","shape":[...],"data_offsets":[s,e]}
 * ========================================================================== */

/* Map safetensors dtype string to TensorDType. Returns -1 for unsupported. */
static int _st_dtype(const char *s) {
    if (strcmp(s, "F32") == 0) return DTYPE_FLOAT32;
    if (strcmp(s, "I32") == 0) return DTYPE_INT32;
    if (strcmp(s, "I64") == 0) return DTYPE_INT64;
    return -1;
}

/* Minimal JSON helpers (inlined here for the safetensors header) */
static void _stj_skip_ws(const char **p) {
    while (**p == ' '||**p == '\t'||**p == '\n'||**p == '\r') (*p)++;
}
static int _stj_str(const char **p, char *buf, int cap) {
    _stj_skip_ws(p);
    if (**p != '"') return -1;
    (*p)++;
    int n = 0;
    while (**p && **p != '"') {
        if (**p == '\\') {
            (*p)++;
            if (buf && n < cap-1) {
                switch (**p) {
                case 'n': buf[n++]='\n'; break;
                case 't': buf[n++]='\t'; break;
                case '"': buf[n++]='"'; break;
                case '\\': buf[n++]='\\'; break;
                default: buf[n++]='\\'; if (n<cap-1) buf[n++]=**p; break;
                }
            } else n++;
        } else {
            if (buf && n < cap-1) buf[n++] = **p;
            else n++;
        }
        (*p)++;
    }
    if (**p == '"') (*p)++;
    if (buf && n < cap) buf[n] = '\0';
    return n;
}

/*
 * Parse the safetensors JSON header and register all tensors via wreg_insert().
 * header: NUL-terminated JSON string.
 * filepath: the .safetensors file (for tensor_from_mmap).
 * data_base: byte offset of the data region within the file (= 8 + header_len).
 */
static bool _st_parse_header(const char *filepath, const char *header,
                               size_t data_base, WRegistry *reg) {
    const char *p = header;
    _stj_skip_ws(&p);
    if (*p != '{') { INF_ERR("safetensors: header not a JSON object"); return false; }
    p++;

    while (*p) {
        _stj_skip_ws(&p);
        if (*p == '}') break;
        if (*p != '"') { p++; continue; }

        /* Parse key (tensor name) */
        char name[256];
        _stj_str(&p, name, sizeof(name));
        _stj_skip_ws(&p);
        if (*p != ':') { while (*p && *p != ',' && *p != '}') p++; continue; }
        p++;
        _stj_skip_ws(&p);

        /* Skip __metadata__ */
        if (strcmp(name, "__metadata__") == 0) {
            int depth = 0;
            while (*p) {
                if (*p == '{' || *p == '[') depth++;
                else if (*p == '}' || *p == ']') { if (--depth <= 0) { p++; break; } }
                p++;
            }
            _stj_skip_ws(&p);
            if (*p == ',') p++;
            continue;
        }

        /* Parse tensor descriptor {"dtype":..., "shape":[...], "data_offsets":[s,e]} */
        if (*p != '{') { while (*p && *p != ',' && *p != '}') p++; continue; }
        p++;

        char   dtype_str[16] = {0};
        int    shape[8]      = {0};
        int    ndim          = 0;
        size_t off_start     = 0;

        while (*p && *p != '}') {
            _stj_skip_ws(&p);
            if (*p != '"') { p++; continue; }
            char fk[32];
            _stj_str(&p, fk, sizeof(fk));
            _stj_skip_ws(&p);
            if (*p != ':') continue;
            p++;
            _stj_skip_ws(&p);

            if (strcmp(fk, "dtype") == 0) {
                _stj_str(&p, dtype_str, sizeof(dtype_str));
            } else if (strcmp(fk, "shape") == 0) {
                if (*p == '[') {
                    p++;
                    while (*p && *p != ']') {
                        _stj_skip_ws(&p);
                        if (*p == ']') break;
                        if (ndim < 8) shape[ndim++] = (int)strtol(p, (char**)&p, 10);
                        while (*p && *p != ',' && *p != ']') p++;
                        if (*p == ',') p++;
                    }
                    if (*p == ']') p++;
                }
            } else if (strcmp(fk, "data_offsets") == 0) {
                if (*p == '[') {
                    p++;
                    _stj_skip_ws(&p);
                    off_start = (size_t)strtoull(p, (char**)&p, 10);
                    /* Skip past end offset */
                    while (*p && *p != ',' && *p != ']') p++;
                }
                /* skip rest of array */
                while (*p && *p != ']') p++;
                if (*p == ']') p++;
            } else {
                /* skip value */
                int dep = 0;
                while (*p && (dep > 0 || (*p != ',' && *p != '}'))) {
                    if (*p=='{'||*p=='['||*p=='"') {
                        if (*p == '"') { p++; while (*p && *p != '"') { if (*p=='\\') p++; p++; } }
                        else dep++;
                    } else if (*p=='}'||*p==']') dep--;
                    if (*p) p++;
                }
            }
            _stj_skip_ws(&p);
            if (*p == ',') p++;
        }
        if (*p == '}') p++;

        int dtype = _st_dtype(dtype_str);
        if (dtype < 0) {
            /* Skip unsupported dtypes (F16, BF16, etc.) silently */
            _stj_skip_ws(&p);
            if (*p == ',') p++;
            continue;
        }

        /* Handle scalar tensors (ndim == 0) as [1] */
        if (ndim == 0) { ndim = 1; shape[0] = 1; }

        size_t byte_offset = data_base + off_start;
        Tensor *t = tensor_from_mmap(filepath, byte_offset, ndim, shape, dtype);
        if (!t) {
            INF_ERRF("tensor_from_mmap failed for '%s'", name);
            /* Non-fatal: skip this tensor */
        } else {
            wreg_insert(reg, name, t, true);
        }

        _stj_skip_ws(&p);
        if (*p == ',') p++;
    }
    return true;
}

/* Load all tensors from a single .safetensors file into the registry. */
static bool _st_load_file(const char *filepath, WRegistry *reg) {
    FILE *f = fopen(filepath, "rb");
    if (!f) { INF_ERRF("cannot open %s", filepath); return false; }

    /* Read 8-byte header length (little-endian uint64) */
    uint64_t header_len = 0;
    if (fread(&header_len, 1, 8, f) != 8) { fclose(f); INF_ERR("short read on header_len"); return false; }
    /* LE byte order (x86 native, no swap needed on LE hosts) */

    if (header_len > 64 * 1024 * 1024) { fclose(f); INF_ERR("header too large"); return false; }

    char *header = malloc(header_len + 1);
    if (!header) { fclose(f); INF_ERR("OOM for header"); return false; }
    if (fread(header, 1, header_len, f) != header_len) {
        free(header); fclose(f); INF_ERR("short read on header"); return false;
    }
    fclose(f);
    header[header_len] = '\0';

    size_t data_base = 8 + header_len;
    bool ok = _st_parse_header(filepath, header, data_base, reg);
    free(header);
    return ok;
}

/* ============================================================================
 * 3.  InferenceSession STRUCT
 * ========================================================================== */

/* Workspace tensors — all pre-allocated, zero additional alloc per step. */
typedef struct {
    Tensor *x;          /* [1, d_model]  — current hidden state            */
    Tensor *xn;         /* [1, d_model]  — RMSNorm output                  */
    Tensor *Q;          /* [1, n_heads * head_dim]                          */
    Tensor *K;          /* [1, n_kv_heads * head_dim]                       */
    Tensor *V;          /* [1, n_kv_heads * head_dim]                       */
    Tensor *attn_out;   /* [1, n_heads * head_dim]  — concat before wo proj */
    Tensor *y;          /* [1, d_model]  — attn or ffn output before add    */
    Tensor *gate;       /* [1, d_ff]     — FFN gate branch                  */
    Tensor *up;         /* [1, d_ff]     — FFN up branch                    */
    Tensor *logits;     /* [1, vocab_size]                                  */
    /* Head-level view tensors (shape=[1,head_dim], data updated per head) */
    Tensor *q_head;
    Tensor *k_head;
    Tensor *v_head;
    Tensor *ah_head;    /* attention output for one head                    */
} InfWS;

struct InferenceSession {
    ModelConfig  cfg;
    WRegistry    reg;
    KVCache   ***kv;     /* [n_layers][n_kv_heads] */
    InfWS        ws;
    Tokenizer   *tok;    /* borrowed — not freed by inf_free */
    int          pos;    /* current position in context */
};

/* ============================================================================
 * 4.  CONFIG PARSER
 * ========================================================================== */

bool inf_parse_config(const char *path, ModelConfig *cfg) {
    FILE *f = fopen(path, "rb");
    if (!f) { INF_ERRF("cannot open %s", path); return false; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); INF_ERR("OOM"); return false; }
    fread(buf, 1, (size_t)sz, f); fclose(f);
    buf[sz] = '\0';

    memset(cfg, 0, sizeof(*cfg));
    cfg->arch       = INF_ARCH_LLAMA;
    cfg->rms_eps    = 1e-5f;
    cfg->rope_base  = 10000.0f;
    cfg->rope_scale = 1.0f;
    cfg->bos_id     = 1;
    cfg->eos_id     = 2;
    cfg->n_kv_heads = -1; /* detect later */

    const char *p = buf;

    /* Simple key-value scanner for flat JSON fields we care about */
    while (*p) {
        _stj_skip_ws(&p);
        if (*p != '"') { p++; continue; }
        char key[64];
        _stj_str(&p, key, sizeof(key));
        _stj_skip_ws(&p);
        if (*p != ':') continue;
        p++;
        _stj_skip_ws(&p);

        /* String values */
        if (*p == '"') {
            char val[64];
            _stj_str(&p, val, sizeof(val));
            if (strcmp(key, "model_type") == 0) {
                if (strcmp(val, "gpt2") == 0) cfg->arch = INF_ARCH_GPT2;
                else cfg->arch = INF_ARCH_LLAMA;
            }
        }
        /* Numeric values */
        else if (*p == '-' || (*p >= '0' && *p <= '9') || *p == '.') {
            double v = strtod(p, (char**)&p);
            if (strcmp(key, "hidden_size") == 0)              cfg->d_model     = (int)v;
            else if (strcmp(key, "num_hidden_layers") == 0)   cfg->n_layers    = (int)v;
            else if (strcmp(key, "num_attention_heads") == 0) cfg->n_heads     = (int)v;
            else if (strcmp(key, "num_key_value_heads") == 0) cfg->n_kv_heads  = (int)v;
            else if (strcmp(key, "intermediate_size") == 0)   cfg->d_ff        = (int)v;
            else if (strcmp(key, "max_position_embeddings") == 0) cfg->max_seq_len = (int)v;
            else if (strcmp(key, "vocab_size") == 0)          cfg->vocab_size  = (int)v;
            else if (strcmp(key, "rms_norm_eps") == 0)        cfg->rms_eps     = (float)v;
            else if (strcmp(key, "layer_norm_eps") == 0)      cfg->rms_eps     = (float)v;
            else if (strcmp(key, "rope_theta") == 0)          cfg->rope_base   = (float)v;
            else if (strcmp(key, "bos_token_id") == 0)        cfg->bos_id      = (int)v;
            else if (strcmp(key, "eos_token_id") == 0)        cfg->eos_id      = (int)v;
        }
        /* Bool values */
        else if (strncmp(p, "true", 4) == 0) {
            if (strcmp(key, "tie_word_embeddings") == 0) cfg->tie_embeddings = true;
            p += 4;
        } else if (strncmp(p, "false", 5) == 0) {
            p += 5;
        } else {
            /* Skip complex values (arrays, nested objects) */
            int dep = 0;
            while (*p && (dep > 0 || (*p != ',' && *p != '}'))) {
                if (*p == '{' || *p == '[') dep++;
                else if (*p == '}' || *p == ']') { if (dep) dep--; else break; }
                else if (*p == '"') { p++; while (*p && *p != '"') { if (*p=='\\') p++; p++; } }
                if (*p) p++;
            }
        }
    }
    free(buf);

    if (cfg->n_kv_heads < 0) cfg->n_kv_heads = cfg->n_heads;
    if (cfg->max_seq_len <= 0) cfg->max_seq_len = 4096;
    if (cfg->attn_scale == 0.0f && cfg->n_heads > 0 && cfg->d_model > 0)
        cfg->attn_scale = 1.0f / sqrtf((float)(cfg->d_model / cfg->n_heads));

    return true;
}

/* ============================================================================
 * 5.  WORKSPACE ALLOCATION / FREE
 * ========================================================================== */

static Tensor *_ws_alloc(int n, TensorDType dt) {
    int sh[2] = {1, n};
    return tensor_create_uninitialized(2, sh, dt);
}

/* Point a pre-allocated head-view tensor at a slice within a workspace buffer. */
static void _ws_set_view(Tensor *view, float *base, int head, int head_dim) {
    view->data       = base + (size_t)head * (size_t)head_dim;
    view->shape[0]   = 1;
    view->shape[1]   = head_dim;
    view->ndim       = 2;
    view->total_size = (size_t)head_dim;
    view->byte_size  = (size_t)head_dim * sizeof(float);
    view->stride[0]  = (size_t)head_dim;
    view->stride[1]  = 1;
    view->owns_data  = false;
}

static bool _ws_create(InfWS *ws, const ModelConfig *cfg) {
    int d      = cfg->d_model;
    int hd     = (cfg->n_heads > 0) ? d / cfg->n_heads : 64;
    int nkv_d  = cfg->n_kv_heads * hd;
    int nq_d   = cfg->n_heads * hd;

    ws->x        = _ws_alloc(d,            DTYPE_FLOAT32); if (!ws->x)       return false;
    ws->xn       = _ws_alloc(d,            DTYPE_FLOAT32); if (!ws->xn)      return false;
    ws->Q        = _ws_alloc(nq_d,         DTYPE_FLOAT32); if (!ws->Q)       return false;
    ws->K        = _ws_alloc(nkv_d,        DTYPE_FLOAT32); if (!ws->K)       return false;
    ws->V        = _ws_alloc(nkv_d,        DTYPE_FLOAT32); if (!ws->V)       return false;
    ws->attn_out = _ws_alloc(nq_d,         DTYPE_FLOAT32); if (!ws->attn_out)return false;
    ws->y        = _ws_alloc(d,            DTYPE_FLOAT32); if (!ws->y)       return false;
    ws->gate     = _ws_alloc(cfg->d_ff,    DTYPE_FLOAT32); if (!ws->gate)    return false;
    ws->up       = _ws_alloc(cfg->d_ff,    DTYPE_FLOAT32); if (!ws->up)      return false;
    ws->logits   = _ws_alloc(cfg->vocab_size, DTYPE_FLOAT32); if (!ws->logits) return false;

    /* Head-view tensors: allocate Tensor structs (data pointer set per use) */
    ws->q_head = _ws_alloc(hd, DTYPE_FLOAT32); if (!ws->q_head) return false;
    ws->k_head = _ws_alloc(hd, DTYPE_FLOAT32); if (!ws->k_head) return false;
    ws->v_head = _ws_alloc(hd, DTYPE_FLOAT32); if (!ws->v_head) return false;
    ws->ah_head = _ws_alloc(hd, DTYPE_FLOAT32); if (!ws->ah_head) return false;

    /* Convert head-view tensors to non-owning views (data will be updated per use) */
    free(ws->q_head->data);  ws->q_head->data  = NULL; ws->q_head->owns_data  = false;
    free(ws->k_head->data);  ws->k_head->data  = NULL; ws->k_head->owns_data  = false;
    free(ws->v_head->data);  ws->v_head->data  = NULL; ws->v_head->owns_data  = false;
    free(ws->ah_head->data); ws->ah_head->data = NULL; ws->ah_head->owns_data = false;

    return true;
}

static void _ws_free(InfWS *ws) {
    tensor_free(ws->x);   tensor_free(ws->xn);  tensor_free(ws->Q);
    tensor_free(ws->K);   tensor_free(ws->V);   tensor_free(ws->attn_out);
    tensor_free(ws->y);   tensor_free(ws->gate);tensor_free(ws->up);
    tensor_free(ws->logits);
    /* Head views: struct only, data not owned */
    if (ws->q_head)  { ws->q_head->owns_data  = false; tensor_free(ws->q_head); }
    if (ws->k_head)  { ws->k_head->owns_data  = false; tensor_free(ws->k_head); }
    if (ws->v_head)  { ws->v_head->owns_data  = false; tensor_free(ws->v_head); }
    if (ws->ah_head) { ws->ah_head->owns_data = false; tensor_free(ws->ah_head); }
    memset(ws, 0, sizeof(*ws));
}

/* ============================================================================
 * 6.  KV CACHE ALLOCATION / FREE
 * ========================================================================== */

static KVCache ***_kv_alloc(const ModelConfig *cfg) {
    int hd = cfg->d_model / cfg->n_heads;
    KVCache ***kv = calloc((size_t)cfg->n_layers, sizeof(KVCache **));
    if (!kv) return NULL;
    for (int l = 0; l < cfg->n_layers; l++) {
        kv[l] = calloc((size_t)cfg->n_kv_heads, sizeof(KVCache *));
        if (!kv[l]) goto fail;
        for (int kh = 0; kh < cfg->n_kv_heads; kh++) {
            kv[l][kh] = kvcache_create(cfg->max_seq_len, hd);
            if (!kv[l][kh]) goto fail;
        }
    }
    return kv;
fail:
    for (int l = 0; l < cfg->n_layers; l++) {
        if (!kv[l]) break;
        for (int kh = 0; kh < cfg->n_kv_heads; kh++)
            if (kv[l][kh]) kvcache_free(kv[l][kh]);
        free(kv[l]);
    }
    free(kv);
    return NULL;
}

static void _kv_free(KVCache ***kv, int n_layers, int n_kv_heads) {
    if (!kv) return;
    for (int l = 0; l < n_layers; l++) {
        if (!kv[l]) continue;
        for (int kh = 0; kh < n_kv_heads; kh++)
            if (kv[l][kh]) kvcache_free(kv[l][kh]);
        free(kv[l]);
    }
    free(kv);
}

/* ============================================================================
 * 7.  WEIGHT NAME RESOLUTION  (LLaMA-style HF names)
 * ========================================================================== */

#define WG(name) wreg_get(reg, name)

static bool _resolve_weights_llama(WRegistry *reg, ModelConfig *cfg, InferenceSession *sess) {
    /* We build thin structs storing pointers into the weight registry.
     * All Tensor* are borrowed from the registry (do not free them here). */

    /* Embeddings */
    Tensor *tok_emb    = WG("model.embed_tokens.weight");
    Tensor *rms_final  = WG("model.norm.weight");
    Tensor *lm_head_w  = WG("lm_head.weight");
    if (!tok_emb)   { INF_ERR("missing model.embed_tokens.weight"); return false; }
    if (!rms_final) { INF_ERR("missing model.norm.weight");          return false; }
    if (!lm_head_w && cfg->tie_embeddings) lm_head_w = tok_emb;
    if (!lm_head_w) { INF_ERR("missing lm_head.weight");             return false; }

    /* We'll store the resolved pointers on the session by adding dedicated fields.
     * For simplicity here, we just verify and expose via inf_get_weight(). */
    (void)sess; /* weight access is via inf_get_weight() / per-layer naming */

    /* Verify all layer weights exist */
    char name[128];
    for (int l = 0; l < cfg->n_layers; l++) {
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
        if (!WG(name)) { INF_ERRF("missing %s", name); return false; }
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", l);
        if (!WG(name)) { INF_ERRF("missing %s", name); return false; }
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", l);
        if (!WG(name)) { INF_ERRF("missing %s", name); return false; }
    }
    return true;
}

/* ============================================================================
 * 8.  SESSION LOADING
 * ========================================================================== */

static InferenceSession *_inf_alloc(const ModelConfig *cfg, Tokenizer *tok) {
    InferenceSession *sess = calloc(1, sizeof(InferenceSession));
    if (!sess) { INF_ERR("OOM"); return NULL; }
    sess->cfg = *cfg;
    sess->tok = tok;
    sess->pos = 0;

    sess->kv = _kv_alloc(cfg);
    if (!sess->kv) { free(sess); INF_ERR("KV cache alloc failed"); return NULL; }

    if (!_ws_create(&sess->ws, cfg)) {
        _kv_free(sess->kv, cfg->n_layers, cfg->n_kv_heads);
        free(sess);
        INF_ERR("workspace alloc failed");
        return NULL;
    }
    return sess;
}

InferenceSession *inf_load_file(const char *weights_path, const ModelConfig *cfg, Tokenizer *tok) {
    if (!cfg) { INF_ERR("ModelConfig is NULL"); return NULL; }

    InferenceSession *sess = _inf_alloc(cfg, tok);
    if (!sess) return NULL;

    if (!_st_load_file(weights_path, &sess->reg)) {
        inf_free(sess);
        return NULL;
    }

    if (!_resolve_weights_llama(&sess->reg, &sess->cfg, sess)) {
        inf_free(sess);
        return NULL;
    }
    return sess;
}

InferenceSession *inf_load(const char *model_dir, const ModelConfig *cfg_in, Tokenizer *tok) {
    ModelConfig cfg_parsed;
    const ModelConfig *cfg = cfg_in;

    /* Try to parse config.json if no config provided */
    if (!cfg) {
        char cpath[1024];
        snprintf(cpath, sizeof(cpath), "%s/config.json", model_dir);
        if (inf_parse_config(cpath, &cfg_parsed)) {
            cfg = &cfg_parsed;
        } else {
            tensor_clear_error();
            INF_ERR("no ModelConfig and config.json not found/parsed");
            return NULL;
        }
    }

    InferenceSession *sess = _inf_alloc(cfg, tok);
    if (!sess) return NULL;

    /* Scan directory for .safetensors files (sorted order for sharded models) */
    DIR *dir = opendir(model_dir);
    if (!dir) { inf_free(sess); INF_ERRF("cannot open dir %s", model_dir); return NULL; }

    /* Collect .safetensors filenames */
    char *files[256];
    int   n_files = 0;
    struct dirent *de;
    while ((de = readdir(dir)) != NULL && n_files < 256) {
        const char *name = de->d_name;
        size_t nl = strlen(name);
        if (nl > 12 && strcmp(name + nl - 12, ".safetensors") == 0) {
            files[n_files++] = strdup(name);
        }
    }
    closedir(dir);

    /* Sort (insertion sort — small n) */
    for (int i = 1; i < n_files; i++) {
        char *tmp = files[i]; int j = i - 1;
        while (j >= 0 && strcmp(files[j], tmp) > 0) { files[j+1] = files[j]; j--; }
        files[j+1] = tmp;
    }

    char fpath[1024];
    for (int i = 0; i < n_files; i++) {
        snprintf(fpath, sizeof(fpath), "%s/%s", model_dir, files[i]);
        _st_load_file(fpath, &sess->reg); /* non-fatal: log and continue */
        free(files[i]);
    }

    if (n_files == 0) {
        inf_free(sess);
        INF_ERRF("no .safetensors files found in %s", model_dir);
        return NULL;
    }

    if (!_resolve_weights_llama(&sess->reg, &sess->cfg, sess)) {
        inf_free(sess);
        return NULL;
    }
    return sess;
}

void inf_free(InferenceSession *sess) {
    if (!sess) return;
    _kv_free(sess->kv, sess->cfg.n_layers, sess->cfg.n_kv_heads);
    _ws_free(&sess->ws);
    wreg_free(&sess->reg);
    free(sess);
}

/* ============================================================================
 * 9.  FORWARD PASS  (single-token incremental)
 * ========================================================================== */

/* Inline SwiGLU in-place: gate[i] = silu(gate[i]) * up[i]
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x)) */
static void _swiglu_inplace(float *gate, const float *up, int n) {
#ifdef __AVX2__
    /* Scalar fallback with fast-math assumptions; compiler vectorizes */
    int i;
    for (i = 0; i < n; i++) {
        float g = gate[i];
        gate[i] = g * (1.0f / (1.0f + expf(-g))) * up[i];
    }
#else
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        gate[i] = g * (1.0f / (1.0f + expf(-g))) * up[i];
    }
#endif
}

/* RMSNorm in-place + scale by weight.
 * Since tensor_rmsnorm is in-place and tensor_mul_inplace scales by weight,
 * we apply them sequentially on the workspace tensor.
 * The weight tensor must have the same shape as x. */
static void _rmsnorm_and_scale(Tensor *x, Tensor *weight, float eps) {
    tensor_rmsnorm(x, eps);
    tensor_mul_inplace(x, weight);
}

/* Build a weight name for a layer. */
#define LNAME(buf, sz, l, suffix) \
    snprintf(buf, sz, "model.layers.%d." suffix, l)

Tensor *inf_step(InferenceSession *sess, int32_t token_id, int pos) {
    const ModelConfig *cfg = &sess->cfg;
    WRegistry         *reg = &sess->reg;
    InfWS             *ws  = &sess->ws;

    int d    = cfg->d_model;
    int hd   = d / cfg->n_heads;
    int nkv  = cfg->n_kv_heads;
    int nh   = cfg->n_heads;
    int gsize = nh / nkv;  /* query heads per KV group */

    /* ── 1. Embedding lookup (zero-copy raw memcpy) ───────────────────── */
    {
        Tensor *emb = wreg_get(reg, "model.embed_tokens.weight");
        if (!emb) { INF_ERR("missing embed_tokens"); return NULL; }
        float *row = (float *)emb->data + (size_t)token_id * (size_t)d;
        memcpy(ws->x->data, row, (size_t)d * sizeof(float));
    }

    char name[128];

    /* ── 2. Transformer layers ────────────────────────────────────────── */
    for (int l = 0; l < cfg->n_layers; l++) {

        /* ── 2a. Attention RMSNorm ─────────────────────────────────────── */
        memcpy(ws->xn->data, ws->x->data, (size_t)d * sizeof(float));
        LNAME(name, sizeof(name), l, "input_layernorm.weight");
        _rmsnorm_and_scale(ws->xn, wreg_get(reg, name), cfg->rms_eps);

        /* ── 2b. QKV projections ──────────────────────────────────────── */
        LNAME(name, sizeof(name), l, "self_attn.q_proj.weight");
        tensor_matmul_into(ws->Q, ws->xn, wreg_get(reg, name), false, true);

        LNAME(name, sizeof(name), l, "self_attn.k_proj.weight");
        tensor_matmul_into(ws->K, ws->xn, wreg_get(reg, name), false, true);

        LNAME(name, sizeof(name), l, "self_attn.v_proj.weight");
        tensor_matmul_into(ws->V, ws->xn, wreg_get(reg, name), false, true);

        /* Optional Q/K biases */
        LNAME(name, sizeof(name), l, "self_attn.q_proj.bias");
        Tensor *bq = wreg_get(reg, name);
        if (bq) tensor_add_inplace(ws->Q, bq);

        LNAME(name, sizeof(name), l, "self_attn.k_proj.bias");
        Tensor *bk = wreg_get(reg, name);
        if (bk) tensor_add_inplace(ws->K, bk);

        LNAME(name, sizeof(name), l, "self_attn.v_proj.bias");
        Tensor *bv = wreg_get(reg, name);
        if (bv) tensor_add_inplace(ws->V, bv);

        /* ── 2c. RoPE on Q and K (per-head) ──────────────────────────── */
        float *Qdata = (float *)ws->Q->data;
        float *Kdata = (float *)ws->K->data;
        float *Vdata = (float *)ws->V->data;
        float *AO    = (float *)ws->attn_out->data;

        /* Apply RoPE to each K head, then append to KV cache */
        for (int kh = 0; kh < nkv; kh++) {
            _ws_set_view(ws->k_head, Kdata, kh, hd);
            _ws_set_view(ws->v_head, Vdata, kh, hd);
            /* RoPE on a single [1, hd] k head (q done per query head below) */
            /* We need a dummy q and real k — pass same tensor twice and ignore q result */
            /* Workaround: apply RoPE to k_head treating it as the k argument;
             * pass a scratch [1,hd] for q (which gets discarded). */
            /* Actually tensor_apply_rope modifies both q AND k in-place.
             * For K-only, use k_head as both q and k — the rotation is identical. */
            tensor_apply_rope(ws->k_head, ws->k_head, hd, pos,
                              cfg->rope_base, cfg->rope_scale);
            kvcache_append(sess->kv[l][kh], ws->k_head, ws->v_head);
        }

        /* Compute attention per query head */
        for (int h = 0; h < nh; h++) {
            int kh = h / gsize;
            _ws_set_view(ws->q_head, Qdata, h, hd);
            _ws_set_view(ws->ah_head, AO, h, hd);

            /* RoPE on this query head */
            tensor_apply_rope(ws->q_head, ws->q_head, hd, pos,
                              cfg->rope_base, cfg->rope_scale);

            /* KV-cached scaled dot-product attention */
            tensor_attention_kv(ws->ah_head, ws->q_head, sess->kv[l][kh]);
        }

        /* ── 2d. Output projection + residual ────────────────────────── */
        LNAME(name, sizeof(name), l, "self_attn.o_proj.weight");
        tensor_matmul_into(ws->y, ws->attn_out, wreg_get(reg, name), false, true);

        LNAME(name, sizeof(name), l, "self_attn.o_proj.bias");
        Tensor *bo = wreg_get(reg, name);
        if (bo) tensor_add_inplace(ws->y, bo);

        tensor_add_inplace(ws->x, ws->y);

        /* ── 2e. FFN RMSNorm ──────────────────────────────────────────── */
        memcpy(ws->xn->data, ws->x->data, (size_t)d * sizeof(float));
        LNAME(name, sizeof(name), l, "post_attention_layernorm.weight");
        _rmsnorm_and_scale(ws->xn, wreg_get(reg, name), cfg->rms_eps);

        /* ── 2f. SwiGLU FFN ──────────────────────────────────────────── */
        LNAME(name, sizeof(name), l, "mlp.gate_proj.weight");
        tensor_matmul_into(ws->gate, ws->xn, wreg_get(reg, name), false, true);

        LNAME(name, sizeof(name), l, "mlp.up_proj.weight");
        tensor_matmul_into(ws->up, ws->xn, wreg_get(reg, name), false, true);

        /* Fused SwiGLU in-place (zero alloc) */
        _swiglu_inplace((float *)ws->gate->data, (float *)ws->up->data, cfg->d_ff);

        LNAME(name, sizeof(name), l, "mlp.down_proj.weight");
        tensor_matmul_into(ws->y, ws->gate, wreg_get(reg, name), false, true);

        LNAME(name, sizeof(name), l, "mlp.down_proj.bias");
        Tensor *bdown = wreg_get(reg, name);
        if (bdown) tensor_add_inplace(ws->y, bdown);

        tensor_add_inplace(ws->x, ws->y);
    }

    /* ── 3. Final RMSNorm ─────────────────────────────────────────────── */
    memcpy(ws->xn->data, ws->x->data, (size_t)d * sizeof(float));
    _rmsnorm_and_scale(ws->xn, wreg_get(reg, "model.norm.weight"), cfg->rms_eps);

    /* ── 4. LM head ───────────────────────────────────────────────────── */
    {
        const char *lm_name = "lm_head.weight";
        Tensor *lm_w = wreg_get(reg, lm_name);
        if (!lm_w) lm_w = wreg_get(reg, "model.embed_tokens.weight"); /* tied */
        tensor_matmul_into(ws->logits, ws->xn, lm_w, false, true);
    }

    sess->pos++;
    return ws->logits; /* owned by session — do NOT tensor_free() */
}

Tensor *inf_forward(InferenceSession *sess, const int32_t *tokens, int n_tokens) {
    if (!sess || !tokens || n_tokens <= 0) return NULL;
    Tensor *logits = NULL;
    for (int i = 0; i < n_tokens; i++) {
        logits = inf_step(sess, tokens[i], sess->pos);
        if (!logits) return NULL;
    }
    return logits;
}

void inf_reset_kv(InferenceSession *sess) {
    if (!sess) return;
    for (int l = 0; l < sess->cfg.n_layers; l++)
        for (int kh = 0; kh < sess->cfg.n_kv_heads; kh++)
            kvcache_reset(sess->kv[l][kh]);
    sess->pos = 0;
}

/* ============================================================================
 * 10. SAMPLING
 * ========================================================================== */

int32_t inf_sample_greedy(const Tensor *logits) {
    return (int32_t)tensor_argmax((Tensor *)logits);
}

/* Xorshift64 RNG */
static uint64_t _xors64(uint64_t *s) {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
    return *s;
}

/* Comparison for qsort (descending logit order) */
typedef struct { float val; int32_t idx; } FloatIdx;
static int _cmp_fi_desc(const void *a, const void *b) {
    float da = ((const FloatIdx*)a)->val;
    float db = ((const FloatIdx*)b)->val;
    return (da < db) - (da > db);
}

int32_t inf_sample_top_p(const Tensor *logits, float temperature, float top_p, uint64_t *rng) {
    if (!logits) return 0;
    int    n  = logits->shape[logits->ndim - 1];
    float *lp = (float *)logits->data;

    if (temperature <= 0.0f) return inf_sample_greedy(logits);

    /* Apply temperature and compute softmax */
    FloatIdx *fi = malloc((size_t)n * sizeof(FloatIdx));
    if (!fi) return inf_sample_greedy(logits);

    float scale = 1.0f / temperature;
    float max_v = lp[0];
    for (int i = 1; i < n; i++) if (lp[i] > max_v) max_v = lp[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        fi[i].val = expf((lp[i] - max_v) * scale);
        fi[i].idx = (int32_t)i;
        sum += fi[i].val;
    }
    for (int i = 0; i < n; i++) fi[i].val /= sum;

    /* Sort descending by probability */
    if (top_p < 1.0f) {
        qsort(fi, (size_t)n, sizeof(FloatIdx), _cmp_fi_desc);
        /* Truncate to nucleus */
        float cum = 0.0f;
        int   keep = 1;
        for (int i = 0; i < n; i++) {
            cum += fi[i].val;
            keep = i + 1;
            if (cum >= top_p) break;
        }
        /* Re-normalize over kept tokens */
        float ksum = 0.0f;
        for (int i = 0; i < keep; i++) ksum += fi[i].val;
        for (int i = 0; i < keep; i++) fi[i].val /= ksum;
        n = keep;
    }

    /* Sample */
    float r = (float)((_xors64(rng) >> 11) * (1.0 / (UINT64_MAX >> 11)));
    float cum = 0.0f;
    int32_t result = fi[0].idx;
    for (int i = 0; i < n; i++) {
        cum += fi[i].val;
        if (r < cum) { result = fi[i].idx; break; }
    }
    free(fi);
    return result;
}

/* Temperature + top-k sampling directly from a raw logits tensor.
 * No InferenceSession needed — suitable for custom autoregressive loops.
 *
 *   k == 0        → full-vocabulary softmax (no truncation)
 *   temperature ≤ 0 → greedy argmax
 *   seed == 0     → seed from tensor pointer (non-deterministic across runs)
 */
int32_t tensor_sample_topk(const Tensor *logits, int k, float temperature, uint64_t seed) {
    if (!logits || logits->dtype != DTYPE_FLOAT32) return 0;
    if (temperature <= 1e-6f) return inf_sample_greedy(logits);

    int          n  = (int)logits->total_size;
    const float *lp = (const float *)logits->data;

    FloatIdx *fi = malloc((size_t)n * sizeof(FloatIdx));
    if (!fi) return inf_sample_greedy(logits);

    float scale = 1.0f / temperature;
    float max_v = lp[0];
    for (int i = 1; i < n; i++) if (lp[i] > max_v) max_v = lp[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        fi[i].val = expf((lp[i] - max_v) * scale);
        fi[i].idx = (int32_t)i;
        sum += fi[i].val;
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) fi[i].val *= inv;

    int keep = n;
    if (k > 0 && k < n) {
        qsort(fi, (size_t)n, sizeof(FloatIdx), _cmp_fi_desc);
        keep = k;
        float ksum = 0.0f;
        for (int i = 0; i < keep; i++) ksum += fi[i].val;
        if (ksum > 0.0f) {
            float kinv = 1.0f / ksum;
            for (int i = 0; i < keep; i++) fi[i].val *= kinv;
        }
    }

    uint64_t rng = seed ? seed : ((uint64_t)(uintptr_t)logits ^ 0xdeadbeefcafe1234ULL);
    _xors64(&rng);
    float r   = (float)((_xors64(&rng) >> 11) * (1.0 / (double)((uint64_t)1 << 53)));
    float cum = 0.0f;
    int32_t result = fi[0].idx;
    for (int i = 0; i < keep; i++) {
        cum += fi[i].val;
        if (r < cum) { result = fi[i].idx; break; }
    }
    free(fi);
    return result;
}

/* ============================================================================
 * 11. GENERATION LOOP
 * ========================================================================== */

void inf_generate(InferenceSession *sess,
                   const int32_t *prompt_ids, int n_prompt,
                   int max_new_tokens,
                   float temperature, float top_p,
                   uint64_t seed,
                   InfTokenCallback callback, void *userdata) {
    if (!sess || !prompt_ids || n_prompt <= 0 || max_new_tokens <= 0) return;

    uint64_t rng = seed ? seed : (uint64_t)time(NULL);
    _xors64(&rng); /* warm up */

    /* Process prompt — no callback for prompt tokens */
    Tensor *logits = inf_forward(sess, prompt_ids, n_prompt);
    if (!logits) return;

    /* Autoregressive generation */
    for (int step = 0; step < max_new_tokens; step++) {
        int32_t next_token;
        if (temperature <= 0.0f) {
            next_token = inf_sample_greedy(logits);
        } else {
            next_token = inf_sample_top_p(logits, temperature, top_p, &rng);
        }

        /* EOS check */
        if (sess->cfg.eos_id >= 0 && next_token == (int32_t)sess->cfg.eos_id) break;

        /* Token string (if tokenizer available) */
        const char *tok_str = NULL;
        if (sess->tok) tok_str = tok_id_to_str(sess->tok, next_token);

        /* Callback */
        if (callback && !callback(next_token, tok_str, userdata)) break;

        /* KV cache length check */
        if (sess->pos >= sess->cfg.max_seq_len) break;

        logits = inf_step(sess, next_token, sess->pos);
        if (!logits) break;
    }
}

/* ============================================================================
 * 12. PHP-FRIENDLY SAMPLING + GENERATION HELPERS
 * ========================================================================== */

int32_t inf_sample(InferenceSession *sess, const Tensor *logits,
                    float temperature, float top_p) {
    if (!logits) return 0;
    /* Use session pointer as RNG seed base — stable across calls */
    uint64_t rng = (uint64_t)(uintptr_t)sess ^ (uint64_t)sess->pos * 6364136223846793005ULL;
    _xors64(&rng);
    if (temperature <= 0.0f) return inf_sample_greedy(logits);
    return inf_sample_top_p(logits, temperature, top_p, &rng);
}

int inf_generate_ids(InferenceSession *sess,
                      const int32_t *prompt_ids, int n_prompt,
                      int max_new_tokens,
                      float temperature, float top_p,
                      uint64_t seed,
                      int32_t *out_ids) {
    if (!sess || !prompt_ids || n_prompt <= 0 || max_new_tokens <= 0 || !out_ids)
        return 0;

    uint64_t rng = seed ? seed : (uint64_t)time(NULL);
    _xors64(&rng);

    Tensor *logits = inf_forward(sess, prompt_ids, n_prompt);
    if (!logits) return 0;

    int n_gen = 0;
    while (n_gen < max_new_tokens) {
        int32_t next;
        if (temperature <= 0.0f) next = inf_sample_greedy(logits);
        else                     next = inf_sample_top_p(logits, temperature, top_p, &rng);

        if (sess->cfg.eos_id >= 0 && next == (int32_t)sess->cfg.eos_id) break;
        if (sess->pos >= sess->cfg.max_seq_len) break;

        out_ids[n_gen++] = next;
        logits = inf_step(sess, next, sess->pos);
        if (!logits) break;
    }
    return n_gen;
}

/* ============================================================================
 * 13. ACCESSORS
 * ========================================================================== */

Tensor *inf_get_weight(const InferenceSession *sess, const char *name) {
    if (!sess || !name) return NULL;
    return wreg_get(&sess->reg, name);
}
