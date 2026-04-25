/*
 * tokenizer.c  —  Byte-level BPE tokenizer (GPT-2 / LLaMA / Mistral compatible)
 *
 * Encoding pipeline:
 *   1. Pre-tokenize: split UTF-8 text into segments via GPT-2 style rules
 *      (contractions, letter runs, digit runs 1-3, punctuation, whitespace).
 *   2. Per segment: convert each UTF-8 byte to its GPT-2 unicode representation
 *      (byte_to_str[256]), look up each single-char token in the vocab hash
 *      table to get initial ids.
 *   3. BPE merge loop: find the highest-priority (lowest rank) adjacent pair
 *      in the ids array; replace them with the merged token; repeat until
 *      no merge applies.  Work buffer is stack-allocated — no heap alloc in
 *      the hot path.
 *   4. Collect ids across all segments into the output buffer.
 *
 * Data structures:
 *   - String pool: one contiguous malloc for all token strings.
 *   - id_to_str[vocab_size]: pointers into the pool.
 *   - Vocab hash table (encode): open-addressing, FNV-1a, string keys.
 *   - Merge hash table: open-addressing, key = (left_id<<32)|right_id.
 *   - special_flags[vocab_size]: bool, set for BOS/EOS/PAD/UNK + added_tokens.
 */

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>
#include <limits.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * Error helpers (reuse tensor error channel)
 * ========================================================================== */
extern void  tensor_set_error(const char *msg);
extern bool  tensor_check_error(void);

#define TOK_ERR(msg) tensor_set_error("[tokenizer] " msg)
#define TOK_ERRF(fmt, ...) do {                        \
    char _ebuf[256];                                    \
    snprintf(_ebuf, sizeof(_ebuf), "[tokenizer] " fmt, \
             ##__VA_ARGS__);                            \
    tensor_set_error(_ebuf);                            \
} while (0)

/* ============================================================================
 * Constants
 * ========================================================================== */
#define TOK_MAX_SEG_BYTES  4096   /* max bytes in one pre-token segment      */
#define TOK_MAX_OUT_TOKENS 32768  /* max tokens in one encode() call          */
#define VOCAB_LOAD_FACTOR  0.65   /* open-addressing fill limit               */
#define MERGE_LOAD_FACTOR  0.65

/* ============================================================================
 * Vocab hash table  (string → int32_t)
 * ========================================================================== */
typedef struct {
    const char *key;   /* pointer into pool, or NULL = empty slot */
    uint32_t    hash;
    int32_t     id;
} VocabSlot;

typedef struct {
    VocabSlot *slots;
    int        cap;    /* always power of 2 */
    int        size;
} VocabHT;

static uint32_t _fnv1a(const char *s, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

static bool _ht_insert(VocabHT *ht, const char *key, int32_t id) {
    uint32_t h = _fnv1a(key, strlen(key));
    int mask = ht->cap - 1;
    int idx  = (int)(h & mask);
    for (int i = 0; i < ht->cap; i++) {
        VocabSlot *s = &ht->slots[idx];
        if (!s->key) { s->key = key; s->hash = h; s->id = id; ht->size++; return true; }
        if (s->hash == h && strcmp(s->key, key) == 0) { s->id = id; return true; }
        idx = (idx + 1) & mask;
    }
    return false; /* table full — should not happen with load factor check */
}

static int32_t _ht_lookup(const VocabHT *ht, const char *key) {
    if (!ht->slots) return -1;
    uint32_t h = _fnv1a(key, strlen(key));
    int mask = ht->cap - 1;
    int idx  = (int)(h & mask);
    for (int i = 0; i < ht->cap; i++) {
        VocabSlot *s = &ht->slots[idx];
        if (!s->key) return -1;
        if (s->hash == h && strcmp(s->key, key) == 0) return s->id;
        idx = (idx + 1) & mask;
    }
    return -1;
}

static int32_t _ht_lookup_n(const VocabHT *ht, const char *key, size_t len) {
    if (!ht->slots) return -1;
    uint32_t h = _fnv1a(key, len);
    int mask = ht->cap - 1;
    int idx  = (int)(h & mask);
    for (int i = 0; i < ht->cap; i++) {
        VocabSlot *s = &ht->slots[idx];
        if (!s->key) return -1;
        if (s->hash == h && strncmp(s->key, key, len) == 0 && s->key[len] == '\0')
            return s->id;
        idx = (idx + 1) & mask;
    }
    return -1;
}

/* ============================================================================
 * Merge hash table  (u64 key → merged id + rank)
 * ========================================================================== */
typedef struct {
    uint64_t key;      /* (left_id << 32) | (uint32_t)right_id; 0 = empty */
    int32_t  merged;
    int32_t  rank;
} MergeSlot;

typedef struct {
    MergeSlot *slots;
    int        cap;
    int        size;
} MergeHT;

static inline uint32_t _merge_hash(uint64_t k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return (uint32_t)k;
}

static bool _mht_insert(MergeHT *ht, int32_t left, int32_t right,
                         int32_t merged, int32_t rank) {
    uint64_t key  = ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
    uint32_t h    = _merge_hash(key);
    int      mask = ht->cap - 1;
    int      idx  = (int)(h & mask);
    for (int i = 0; i < ht->cap; i++) {
        MergeSlot *s = &ht->slots[idx];
        if (!s->key) {
            s->key = key; s->merged = merged; s->rank = rank;
            ht->size++;
            return true;
        }
        if (s->key == key) { /* duplicate — keep lower rank */
            if (rank < s->rank) { s->merged = merged; s->rank = rank; }
            return true;
        }
        idx = (idx + 1) & mask;
    }
    return false;
}

/* Returns -1 if not found, else populates *merged and *rank */
static int _mht_lookup(const MergeHT *ht, int32_t left, int32_t right,
                        int32_t *merged, int32_t *rank) {
    if (!ht->slots) return -1;
    uint64_t key  = ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
    uint32_t h    = _merge_hash(key);
    int      mask = ht->cap - 1;
    int      idx  = (int)(h & mask);
    for (int i = 0; i < ht->cap; i++) {
        MergeSlot *s = &ht->slots[idx];
        if (!s->key) return -1;
        if (s->key == key) { *merged = s->merged; *rank = s->rank; return 0; }
        idx = (idx + 1) & mask;
    }
    return -1;
}

/* ============================================================================
 * String pool
 * ========================================================================== */
typedef struct {
    char  *data;
    size_t used;
    size_t cap;
} StrPool;

static char *_pool_add(StrPool *p, const char *str, size_t len) {
    size_t need = p->used + len + 1;
    if (need > p->cap) {
        size_t ncap = p->cap ? p->cap : 1024 * 1024;
        while (ncap < need) ncap *= 2;
        char *nd = realloc(p->data, ncap);
        if (!nd) return NULL;
        p->data = nd; p->cap = ncap;
    }
    char *dst = p->data + p->used;
    memcpy(dst, str, len);
    dst[len] = '\0';
    p->used += len + 1;
    return dst;
}

/* ============================================================================
 * Tokenizer struct
 * ========================================================================== */
struct Tokenizer {
    /* id ↔ string */
    char   **id_to_str;  /* [vocab_size] pointers into pool */
    bool    *is_special; /* [vocab_size] */
    int      vocab_size;

    /* string → id */
    VocabHT  encode_ht;

    /* (left,right) → (merged,rank) */
    MergeHT  merge_ht;

    /* string pool */
    StrPool  pool;

    /* byte → GPT-2 unicode string (UTF-8 encoded) and its token id */
    char     byte_str[256][8]; /* UTF-8 for each byte's unicode mapping */
    int      byte_str_len[256];
    int32_t  byte_id[256];     /* vocab id for each byte's single-char token */

    /* special token ids */
    int bos_id;
    int eos_id;
    int pad_id;
    int unk_id;
};

/* ============================================================================
 * GPT-2 byte-to-unicode mapping
 * ========================================================================== */
static void _utf8_encode(uint32_t cp, char *out, int *len) {
    if (cp < 0x80) {
        out[0] = (char)cp; out[1] = '\0'; *len = 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        out[2] = '\0'; *len = 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        out[3] = '\0'; *len = 3;
    } else {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6)  & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        out[4] = '\0'; *len = 4;
    }
}

/* GPT-2 passthrough bytes: 33-126, 161-172, 174-255 */
static bool _is_passthrough(int b) {
    return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
}

static void _build_byte_map(Tokenizer *tok) {
    int shift = 0;
    for (int b = 0; b < 256; b++) {
        uint32_t cp;
        if (_is_passthrough(b)) {
            cp = (uint32_t)b;
        } else {
            cp = 256u + (uint32_t)shift++;
        }
        _utf8_encode(cp, tok->byte_str[b], &tok->byte_str_len[b]);
    }
}

/* Compute byte_id[256] from the vocab hash table (call after loading vocab). */
static void _build_byte_ids(Tokenizer *tok) {
    for (int b = 0; b < 256; b++) {
        tok->byte_id[b] = _ht_lookup(&tok->encode_ht, tok->byte_str[b]);
    }
}

/* ============================================================================
 * UTF-8 helpers
 * ========================================================================== */
/* Returns the number of bytes in the UTF-8 character starting at *p,
 * and writes the codepoint to *cp.  Returns 0 on invalid byte. */
static int _utf8_decode(const unsigned char *p, uint32_t *cp) {
    if (p[0] < 0x80) { *cp = p[0]; return 1; }
    if ((p[0] & 0xE0) == 0xC0) {
        if ((p[1] & 0xC0) != 0x80) return 0;
        *cp = ((uint32_t)(p[0] & 0x1F) << 6) | (p[1] & 0x3F); return 2;
    }
    if ((p[0] & 0xF0) == 0xE0) {
        if ((p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) return 0;
        *cp = ((uint32_t)(p[0]&0x0F)<<12)|((uint32_t)(p[1]&0x3F)<<6)|(p[2]&0x3F); return 3;
    }
    if ((p[0] & 0xF8) == 0xF0) {
        if ((p[1]&0xC0)!=0x80||(p[2]&0xC0)!=0x80||(p[3]&0xC0)!=0x80) return 0;
        *cp = ((uint32_t)(p[0]&7)<<18)|((uint32_t)(p[1]&0x3F)<<12)
             |((uint32_t)(p[2]&0x3F)<<6)|(p[3]&0x3F); return 4;
    }
    return 0;
}

static bool _is_letter(uint32_t cp) {
    /* ASCII letters */
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    /* Common Unicode letter ranges (Latin extended, etc.) */
    if (cp >= 0xC0 && cp <= 0x2AF) return true;
    if (cp >= 0x370 && cp <= 0x3FF) return true;
    if (cp >= 0x400 && cp <= 0x4FF) return true;
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true; /* CJK */
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true; /* Korean */
    return false;
}

static bool _is_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }
static bool _is_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r';
}

/* ============================================================================
 * Pre-tokenizer
 * Approximates the GPT-2 regex: contractions, letter runs, digit runs (1-3),
 * optional-space+punctuation, whitespace-only.
 * Writes byte offsets of segments into segs[2*MAX_SEGS] = {start, len, ...}.
 * Returns segment count.
 * ========================================================================== */
#define MAX_SEGS 8192

static int _pretokenize(const char *text, size_t text_len,
                         size_t *starts, size_t *lens) {
    const unsigned char *p   = (const unsigned char *)text;
    const unsigned char *end = p + text_len;
    int n_segs = 0;

    while (p < end && n_segs < MAX_SEGS) {
        const unsigned char *seg_start = p;

        uint32_t cp;
        int      clen = _utf8_decode(p, &cp);
        if (clen <= 0) { p++; continue; } /* skip invalid byte */

        /* ── Case 1: whitespace-only segment ─────────────────────────────── */
        if (_is_whitespace(cp)) {
            while (p < end) {
                clen = _utf8_decode(p, &cp);
                if (clen <= 0 || !_is_whitespace(cp)) break;
                p += clen;
            }
            /* Check if next char is non-space — if so, the whitespace belongs
             * to the NEXT segment as a prefix (GPT-2 attaches space to word). */
            if (p < end) {
                /* trailing whitespace before word/punct: roll back p to seg_start
                 * and let the next iteration pick up the space as a prefix */
                /* Actually: emit whitespace as its own segment only if followed
                 * by nothing or another whitespace group.  If followed by a
                 * non-space, absorb one leading space into next iteration. */
                size_t ws_len = (size_t)(p - seg_start);
                if (ws_len > 1) {
                    /* emit all-but-last-space as whitespace-only segment */
                    starts[n_segs] = (size_t)(seg_start - (const unsigned char *)text);
                    lens[n_segs]   = ws_len - 1;
                    n_segs++;
                    seg_start = p - 1; /* one space left for next seg */
                }
                /* fall through: seg_start now at a single space before word */
                p = seg_start; /* restart from the remaining space */
            } else {
                starts[n_segs] = (size_t)(seg_start - (const unsigned char *)text);
                lens[n_segs]   = (size_t)(p - seg_start);
                n_segs++;
                continue;
            }
            /* reread leading space */
            clen = _utf8_decode(p, &cp);
        }

        bool leading_space = false;
        if (cp == ' ') {
            leading_space = true;
            p += clen;
            if (p >= end) {
                starts[n_segs] = (size_t)(seg_start - (const unsigned char *)text);
                lens[n_segs]   = (size_t)(p - seg_start);
                n_segs++;
                continue;
            }
            clen = _utf8_decode(p, &cp);
            if (clen <= 0) { p++; continue; }
        }

        /* ── Case 2: contraction 's 't 're 've 'm 'll 'd  ───────────────── */
        if (cp == '\'') {
            const unsigned char *q = p + clen;
            uint32_t nc; int nl;
            if (q < end && (nl = _utf8_decode(q, &nc)) > 0) {
                /* check for s, t, m, d, re, ve, ll */
                bool contraction = false;
                if (nc=='s'||nc=='t'||nc=='m'||nc=='d'||nc=='S'||nc=='T'||nc=='M'||nc=='D')
                    contraction = true;
                else if ((nc=='r'||nc=='R'||nc=='v'||nc=='V'||nc=='l'||nc=='L') && q+nl < end) {
                    uint32_t nc2; int nl2 = _utf8_decode(q+nl, &nc2);
                    if (nl2 > 0 && (nc2=='e'||nc2=='E'||nc2=='e'||nc2=='l'||nc2=='L'))
                        contraction = true;
                }
                if (contraction) {
                    /* emit up to end of contraction (simple: emit 2-3 chars) */
                    p += clen + nl;
                    /* check for second char of 're/'ve/'ll */
                    if ((nc=='r'||nc=='v'||nc=='l') && p < end) {
                        _utf8_decode(p, &nc);
                        if (nc=='e'||nc=='l') { int l2; _utf8_decode(p,&nc); p += clen; (void)l2; }
                    }
                    starts[n_segs] = (size_t)(seg_start - (const unsigned char *)text);
                    lens[n_segs]   = (size_t)(p - seg_start);
                    n_segs++;
                    continue;
                }
            }
        }

        /* ── Case 3: letter run (with optional leading space) ────────────── */
        if (_is_letter(cp)) {
            p += clen;
            while (p < end) {
                int cl = _utf8_decode(p, &cp);
                if (cl <= 0 || !_is_letter(cp)) break;
                p += cl;
            }
            starts[n_segs] = (size_t)(seg_start - (const unsigned char *)text);
            lens[n_segs]   = (size_t)(p - seg_start);
            n_segs++;
            continue;
        }

        /* ── Case 4: digit run, 1-3 digits ───────────────────────────────── */
        if (_is_digit(cp) && !leading_space) {
            p += clen;
            int cnt = 1;
            while (p < end && cnt < 3) {
                int cl = _utf8_decode(p, &cp);
                if (cl <= 0 || !_is_digit(cp)) break;
                p += cl; cnt++;
            }
            starts[n_segs] = (size_t)(seg_start - (const unsigned char *)text);
            lens[n_segs]   = (size_t)(p - seg_start);
            n_segs++;
            continue;
        }

        /* ── Case 5: punctuation / other (optional space + non-space chars) ─ */
        p += clen;
        while (p < end) {
            int cl = _utf8_decode(p, &cp);
            if (cl <= 0) { p++; break; }
            if (_is_whitespace(cp) || _is_letter(cp) || _is_digit(cp)) break;
            p += cl;
        }
        starts[n_segs] = (size_t)(seg_start - (const unsigned char *)text);
        lens[n_segs]   = (size_t)(p - seg_start);
        n_segs++;
    }
    return n_segs;
}

/* ============================================================================
 * BPE encode for a single segment.
 * Input:  raw bytes of the segment (UTF-8).
 * Output: writes token ids into out[]; returns token count.
 * Stack-allocated work buffer — no heap alloc.
 * ========================================================================== */
static int _bpe_segment(const Tokenizer *tok,
                          const unsigned char *bytes, size_t n_bytes,
                          int32_t *out, int out_cap) {
    /* Step 1: convert each byte to its GPT-2 unicode byte-string, look up id */
    int32_t ids[TOK_MAX_SEG_BYTES];
    int     n = 0;

    for (size_t i = 0; i < n_bytes && n < TOK_MAX_SEG_BYTES; i++) {
        unsigned char b = bytes[i];
        int32_t id = tok->byte_id[b];
        if (id < 0) {
            /* unknown byte — substitute unk or skip */
            id = tok->unk_id;
            if (id < 0) continue;
        }
        ids[n++] = id;
    }

    if (n == 0) return 0;

    /* Step 2: BPE merge loop — O(n²) per segment, fine for typical lengths */
    while (n > 1) {
        int best_pos  = -1;
        int best_rank = INT_MAX;
        int32_t best_merged = -1;

        for (int i = 0; i < n - 1; i++) {
            int32_t merged; int32_t rank;
            if (_mht_lookup(&tok->merge_ht, ids[i], ids[i+1], &merged, &rank) == 0) {
                if (rank < best_rank) {
                    best_rank   = rank;
                    best_pos    = i;
                    best_merged = merged;
                }
            }
        }

        if (best_pos < 0) break; /* no more merges */

        ids[best_pos] = best_merged;
        memmove(ids + best_pos + 1, ids + best_pos + 2,
                (size_t)(n - best_pos - 2) * sizeof(int32_t));
        n--;
    }

    /* Step 3: copy to output */
    int copy = n < out_cap ? n : out_cap;
    memcpy(out, ids, (size_t)copy * sizeof(int32_t));
    return copy;
}

/* ============================================================================
 * Minimal JSON parser helpers
 * Only handles the specific structures in tokenizer.json / vocab.json.
 * ========================================================================== */

static void _skip_ws(const char **p) {
    while (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r') (*p)++;
}

/* Parse a JSON string into buf (up to buf_cap-1 bytes). Returns length or -1. */
static int _parse_jstr(const char **p, char *buf, int buf_cap) {
    _skip_ws(p);
    if (**p != '"') return -1;
    (*p)++;
    int len = 0;
    while (**p && **p != '"') {
        if (**p == '\\') {
            (*p)++;
            char esc = **p;
            if (buf && len < buf_cap - 1) {
                switch (esc) {
                case '"': case '\\': case '/': buf[len++] = esc; break;
                case 'n': buf[len++] = '\n'; break;
                case 'r': buf[len++] = '\r'; break;
                case 't': buf[len++] = '\t'; break;
                case 'b': buf[len++] = '\b'; break;
                case 'f': buf[len++] = '\f'; break;
                case 'u': {
                    /* \uXXXX — decode as UTF-8 */
                    unsigned int cp = 0;
                    (*p)++;
                    for (int i = 0; i < 4 && **p; i++, (*p)++) {
                        char c = **p;
                        int v = (c>='0'&&c<='9') ? c-'0' :
                                (c>='a'&&c<='f') ? c-'a'+10 :
                                (c>='A'&&c<='F') ? c-'A'+10 : 0;
                        cp = (cp << 4) | (unsigned)v;
                    }
                    (*p)--;
                    char utf[8]; int ul;
                    _utf8_encode(cp, utf, &ul);
                    for (int i = 0; i < ul && len < buf_cap-1; i++) buf[len++] = utf[i];
                    break;
                }
                default: buf[len++] = esc; break;
                }
            }
        } else {
            if (buf && len < buf_cap - 1) buf[len++] = **p;
            else len++;
        }
        (*p)++;
    }
    if (**p == '"') (*p)++;
    if (buf) buf[len < buf_cap ? len : buf_cap-1] = '\0';
    return len;
}

/* Skip a JSON value (any type). Returns false on error. */
static bool _skip_jvalue(const char **p) {
    _skip_ws(p);
    char c = **p;
    if (c == '"') { char tmp[4]; _parse_jstr(p, NULL, 0); return true; }
    if (c == '{') {
        (*p)++; _skip_ws(p);
        if (**p == '}') { (*p)++; return true; }
        while (**p) {
            _skip_ws(p);
            _parse_jstr(p, NULL, 0); _skip_ws(p);
            if (**p != ':') return false; (*p)++; _skip_ws(p);
            _skip_jvalue(p); _skip_ws(p);
            if (**p == ',') { (*p)++; continue; }
            if (**p == '}') { (*p)++; return true; }
            return false;
        }
        return false;
    }
    if (c == '[') {
        (*p)++; _skip_ws(p);
        if (**p == ']') { (*p)++; return true; }
        while (**p) {
            _skip_jvalue(p); _skip_ws(p);
            if (**p == ',') { (*p)++; continue; }
            if (**p == ']') { (*p)++; return true; }
            return false;
        }
        return false;
    }
    /* number, bool, null */
    while (**p && **p != ',' && **p != '}' && **p != ']' &&
           **p != ' ' && **p != '\n' && **p != '\r' && **p != '\t')
        (*p)++;
    return true;
}

static bool _skip_to_key(const char **p, const char *key) {
    while (**p) {
        _skip_ws(p);
        if (**p != '"') { (*p)++; continue; }
        char kbuf[128];
        _parse_jstr(p, kbuf, sizeof(kbuf));
        _skip_ws(p);
        if (**p != ':') continue;
        (*p)++;
        _skip_ws(p);
        if (strcmp(kbuf, key) == 0) return true;
        _skip_jvalue(p);
    }
    return false;
}

/* ============================================================================
 * Tokenizer construction helpers
 * ========================================================================== */
static int _next_pow2(int n) {
    int p = 1; while (p < n) p <<= 1; return p;
}

static bool _alloc_vocab_ht(VocabHT *ht, int vocab_size) {
    int cap = _next_pow2((int)(vocab_size / VOCAB_LOAD_FACTOR) + 4);
    ht->slots = calloc((size_t)cap, sizeof(VocabSlot));
    if (!ht->slots) return false;
    ht->cap = cap; ht->size = 0;
    return true;
}

static bool _alloc_merge_ht(MergeHT *ht, int n_merges) {
    int cap = _next_pow2((int)(n_merges / MERGE_LOAD_FACTOR) + 4);
    ht->slots = calloc((size_t)cap, sizeof(MergeSlot));
    if (!ht->slots) return false;
    ht->cap = cap; ht->size = 0;
    return true;
}

/* Finish building: resolve special token ids by well-known names. */
static void _resolve_specials(Tokenizer *tok) {
    tok->bos_id = tok->eos_id = tok->pad_id = tok->unk_id = -1;

    static const char *bos_names[] = {"<s>","<|begin_of_text|>","<BOS_TOKEN>","[BOS]",NULL};
    static const char *eos_names[] = {"</s>","<|end_of_text|>","<|eot_id|>","<EOS_TOKEN>","[EOS]",NULL};
    static const char *pad_names[] = {"<pad>","<PAD>","[PAD]",NULL};
    static const char *unk_names[] = {"<unk>","<UNK>","[UNK]",NULL};

    for (int i = 0; bos_names[i] && tok->bos_id < 0; i++)
        tok->bos_id = _ht_lookup(&tok->encode_ht, bos_names[i]);
    for (int i = 0; eos_names[i] && tok->eos_id < 0; i++)
        tok->eos_id = _ht_lookup(&tok->encode_ht, eos_names[i]);
    for (int i = 0; pad_names[i] && tok->pad_id < 0; i++)
        tok->pad_id = _ht_lookup(&tok->encode_ht, pad_names[i]);
    for (int i = 0; unk_names[i] && tok->unk_id < 0; i++)
        tok->unk_id = _ht_lookup(&tok->encode_ht, unk_names[i]);

    /* Fallback: pad = eos */
    if (tok->pad_id < 0) tok->pad_id = tok->eos_id;
}

/* ============================================================================
 * tok_load_json  —  Load HuggingFace tokenizer.json
 * ========================================================================== */
Tokenizer *tok_load_json(const char *path) {
    /* Read file into memory */
    FILE *f = fopen(path, "rb");
    if (!f) { TOK_ERRF("cannot open %s", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)fsz + 1);
    if (!buf) { fclose(f); TOK_ERR("OOM reading tokenizer.json"); return NULL; }
    fread(buf, 1, (size_t)fsz, f);
    fclose(f);
    buf[fsz] = '\0';

    Tokenizer *tok = calloc(1, sizeof(Tokenizer));
    if (!tok) { free(buf); TOK_ERR("OOM"); return NULL; }
    tok->bos_id = tok->eos_id = tok->pad_id = tok->unk_id = -1;
    _build_byte_map(tok);

    /* ── Pass 1: scan for model.vocab to learn vocab_size ─────────────────── */
    const char *p    = buf;
    int         vsz  = 0;
    int         n_merges = 0;

    /* Count vocab entries */
    const char *vocab_start = strstr(buf, "\"vocab\"");
    if (!vocab_start) { free(buf); free(tok); TOK_ERR("no vocab in tokenizer.json"); return NULL; }
    p = vocab_start + 7;
    _skip_ws(&p);
    if (*p != ':') { free(buf); free(tok); TOK_ERR("bad vocab"); return NULL; }
    p++;
    _skip_ws(&p);
    if (*p != '{') { free(buf); free(tok); TOK_ERR("bad vocab"); return NULL; }
    p++;
    while (*p && *p != '}') {
        _skip_ws(&p);
        if (*p == '"') { _parse_jstr(&p, NULL, 0); vsz++; }
        while (*p && *p != '"' && *p != '}') p++;
    }

    /* Count merges */
    const char *merges_start = strstr(buf, "\"merges\"");
    if (merges_start) {
        p = merges_start + 8;
        _skip_ws(&p);
        if (*p == ':') { p++; _skip_ws(&p); }
        if (*p == '[') {
            p++;
            while (*p && *p != ']') {
                _skip_ws(&p);
                if (*p == '"') { _parse_jstr(&p, NULL, 0); n_merges++; }
                while (*p && *p != '"' && *p != ']') p++;
            }
        }
    }

    if (vsz <= 0) { free(buf); free(tok); TOK_ERR("empty vocab"); return NULL; }

    /* ── Allocate structures ─────────────────────────────────────────────── */
    tok->vocab_size = vsz;
    tok->id_to_str  = calloc((size_t)vsz, sizeof(char *));
    tok->is_special = calloc((size_t)vsz, sizeof(bool));
    if (!tok->id_to_str || !tok->is_special) goto oom;

    if (!_alloc_vocab_ht(&tok->encode_ht, vsz))  goto oom;
    if (n_merges > 0 && !_alloc_merge_ht(&tok->merge_ht, n_merges)) goto oom;

    /* ── Pass 2: load vocab ──────────────────────────────────────────────── */
    p = vocab_start + 7;
    _skip_ws(&p);
    p++; /* ':' */
    _skip_ws(&p);
    p++; /* '{' */

    while (*p) {
        _skip_ws(&p);
        if (*p == '}') break;
        if (*p != '"') { p++; continue; }

        char kbuf[1024];
        int klen = _parse_jstr(&p, kbuf, sizeof(kbuf));
        _skip_ws(&p);
        if (*p != ':') break;
        p++;
        _skip_ws(&p);

        /* parse integer id */
        long id = strtol(p, (char **)&p, 10);
        if (id < 0 || id >= vsz) { while (*p && *p != ',' && *p != '}') p++; goto next_vocab; }

        /* add to pool */
        char *stored = _pool_add(&tok->pool, kbuf, (size_t)klen);
        if (!stored) goto oom;
        tok->id_to_str[id] = stored;
        _ht_insert(&tok->encode_ht, stored, (int32_t)id);

next_vocab:
        _skip_ws(&p);
        if (*p == ',') p++;
    }

    /* ── Pass 3: load merges ─────────────────────────────────────────────── */
    if (merges_start && n_merges > 0) {
        p = merges_start + 8;
        _skip_ws(&p);
        if (*p == ':') p++;
        _skip_ws(&p);
        if (*p == '[') p++;

        int rank = 0;
        while (*p && *p != ']') {
            _skip_ws(&p);
            if (*p != '"') { if (*p) p++; continue; }

            char mbuf[512];
            _parse_jstr(&p, mbuf, sizeof(mbuf));

            /* split on first space */
            char *sp = strchr(mbuf, ' ');
            if (!sp) { rank++; continue; }
            *sp = '\0';
            const char *tok_a = mbuf;
            const char *tok_b = sp + 1;
            /* merged = tok_a + tok_b concatenated */
            char merged_str[1024];
            int la = (int)strlen(tok_a), lb = (int)strlen(tok_b);
            if (la + lb < (int)sizeof(merged_str)) {
                memcpy(merged_str, tok_a, (size_t)la);
                memcpy(merged_str + la, tok_b, (size_t)lb + 1);
            }
            int32_t id_a    = _ht_lookup(&tok->encode_ht, tok_a);
            int32_t id_b    = _ht_lookup(&tok->encode_ht, tok_b);
            int32_t id_merged = _ht_lookup(&tok->encode_ht, merged_str);
            if (id_a >= 0 && id_b >= 0 && id_merged >= 0) {
                _mht_insert(&tok->merge_ht, id_a, id_b, id_merged, rank);
            }
            rank++;
            _skip_ws(&p);
            if (*p == ',') p++;
        }
    }

    /* ── Pass 4: added_tokens → mark as special ──────────────────────────── */
    const char *at_start = strstr(buf, "\"added_tokens\"");
    if (at_start) {
        p = at_start + 14;
        _skip_ws(&p);
        if (*p == ':') p++;
        _skip_ws(&p);
        if (*p == '[') {
            p++;
            while (*p && *p != ']') {
                _skip_ws(&p);
                if (*p != '{') { if (*p) p++; continue; }
                p++;
                /* scan for "id", "content", "special" */
                long   at_id      = -1;
                char   at_content[256] = {0};
                bool   at_special = false;
                while (*p && *p != '}') {
                    _skip_ws(&p);
                    if (*p != '"') { if (*p) p++; continue; }
                    char fk[64];
                    _parse_jstr(&p, fk, sizeof(fk));
                    _skip_ws(&p);
                    if (*p != ':') continue;
                    p++;
                    _skip_ws(&p);
                    if (strcmp(fk, "id") == 0) {
                        at_id = strtol(p, (char **)&p, 10);
                    } else if (strcmp(fk, "content") == 0) {
                        _parse_jstr(&p, at_content, sizeof(at_content));
                    } else if (strcmp(fk, "special") == 0) {
                        _skip_ws(&p);
                        if (strncmp(p, "true", 4) == 0)  { at_special = true;  p += 4; }
                        else if (strncmp(p, "false", 5) == 0) p += 5;
                    } else {
                        _skip_jvalue(&p);
                    }
                    _skip_ws(&p);
                    if (*p == ',') p++;
                }
                if (*p == '}') p++;

                if (at_id >= 0 && at_id < vsz) {
                    /* If token not in vocab yet, add it */
                    if (!tok->id_to_str[at_id] && at_content[0]) {
                        char *stored = _pool_add(&tok->pool, at_content,
                                                  strlen(at_content));
                        if (stored) {
                            tok->id_to_str[at_id] = stored;
                            _ht_insert(&tok->encode_ht, stored, (int32_t)at_id);
                        }
                    }
                    if (at_special) tok->is_special[at_id] = true;
                }
                _skip_ws(&p);
                if (*p == ',') p++;
            }
        }
    }

    free(buf);
    _build_byte_ids(tok);
    _resolve_specials(tok);

    /* mark bos/eos/pad/unk as special */
    int specials[] = { tok->bos_id, tok->eos_id, tok->pad_id, tok->unk_id };
    for (int i = 0; i < 4; i++)
        if (specials[i] >= 0) tok->is_special[specials[i]] = true;

    return tok;

oom:
    free(buf);
    tok_free(tok);
    TOK_ERR("OOM");
    return NULL;
}

/* ============================================================================
 * tok_load  —  Load from vocab.json + merges.txt
 * ========================================================================== */
Tokenizer *tok_load(const char *vocab_path, const char *merges_path) {
    /* Read vocab.json */
    FILE *f = fopen(vocab_path, "rb");
    if (!f) { TOK_ERRF("cannot open %s", vocab_path); return NULL; }
    fseek(f, 0, SEEK_END); long vsz = ftell(f); fseek(f, 0, SEEK_SET);
    char *vbuf = malloc((size_t)vsz + 1);
    if (!vbuf) { fclose(f); TOK_ERR("OOM"); return NULL; }
    fread(vbuf, 1, (size_t)vsz, f); fclose(f);
    vbuf[vsz] = '\0';

    /* Count vocab entries */
    int n_vocab = 0;
    const char *p = vbuf;
    _skip_ws(&p);
    if (*p != '{') { free(vbuf); TOK_ERR("vocab.json: expected {"); return NULL; }
    p++;
    {
        const char *q = p;
        while (*q && *q != '}') {
            _skip_ws(&q);
            if (*q == '"') { _parse_jstr(&q, NULL, 0); n_vocab++; }
            while (*q && *q != '"' && *q != '}') q++;
        }
    }

    /* Count merges */
    int n_merges = 0;
    {
        FILE *mf = fopen(merges_path, "r");
        if (mf) {
            char line[512];
            while (fgets(line, sizeof(line), mf)) {
                if (line[0] == '#' || line[0] == '\n') continue;
                n_merges++;
            }
            fclose(mf);
        }
    }

    Tokenizer *tok = calloc(1, sizeof(Tokenizer));
    if (!tok) { free(vbuf); TOK_ERR("OOM"); return NULL; }
    tok->bos_id = tok->eos_id = tok->pad_id = tok->unk_id = -1;
    _build_byte_map(tok);

    tok->vocab_size = n_vocab;
    tok->id_to_str  = calloc((size_t)n_vocab, sizeof(char *));
    tok->is_special = calloc((size_t)n_vocab, sizeof(bool));
    if (!tok->id_to_str || !tok->is_special) goto oom2;
    if (!_alloc_vocab_ht(&tok->encode_ht, n_vocab))  goto oom2;
    if (n_merges > 0 && !_alloc_merge_ht(&tok->merge_ht, n_merges + 1)) goto oom2;

    /* Load vocab */
    p = vbuf + 1; /* skip '{' */
    while (*p) {
        _skip_ws(&p);
        if (*p == '}') break;
        if (*p != '"') { p++; continue; }
        char kbuf[1024];
        int klen = _parse_jstr(&p, kbuf, sizeof(kbuf));
        _skip_ws(&p);
        if (*p != ':') break; p++;
        _skip_ws(&p);
        long id = strtol(p, (char **)&p, 10);
        if (id >= 0 && id < n_vocab) {
            char *stored = _pool_add(&tok->pool, kbuf, (size_t)klen);
            if (!stored) goto oom2;
            tok->id_to_str[id] = stored;
            _ht_insert(&tok->encode_ht, stored, (int32_t)id);
        }
        _skip_ws(&p); if (*p == ',') p++;
    }
    free(vbuf); vbuf = NULL;

    /* Load merges */
    {
        FILE *mf = fopen(merges_path, "r");
        if (mf) {
            char line[512];
            int rank = 0;
            while (fgets(line, sizeof(line), mf)) {
                if (line[0] == '#' || line[0] == '\n') continue;
                /* trim newline */
                int len = (int)strlen(line);
                while (len > 0 && (line[len-1]=='\n'||line[len-1]=='\r')) line[--len]=0;

                char *sp = strchr(line, ' ');
                if (!sp) { sp = strchr(line, '\t'); }
                if (!sp) { rank++; continue; }
                *sp = '\0';
                const char *tok_a = line;
                const char *tok_b = sp + 1;

                char merged_str[1024];
                int la = (int)strlen(tok_a), lb = (int)strlen(tok_b);
                if (la + lb < (int)sizeof(merged_str)) {
                    memcpy(merged_str, tok_a, (size_t)la);
                    memcpy(merged_str + la, tok_b, (size_t)lb + 1);
                }
                int32_t id_a     = _ht_lookup(&tok->encode_ht, tok_a);
                int32_t id_b     = _ht_lookup(&tok->encode_ht, tok_b);
                int32_t id_merge = _ht_lookup(&tok->encode_ht, merged_str);
                if (id_a >= 0 && id_b >= 0 && id_merge >= 0) {
                    _mht_insert(&tok->merge_ht, id_a, id_b, id_merge, rank);
                }
                rank++;
            }
            fclose(mf);
        }
    }

    _build_byte_ids(tok);
    _resolve_specials(tok);
    return tok;

oom2:
    free(vbuf);
    tok_free(tok);
    TOK_ERR("OOM");
    return NULL;
}

/* ============================================================================
 * tok_free
 * ========================================================================== */
void tok_free(Tokenizer *tok) {
    if (!tok) return;
    free(tok->id_to_str);
    free(tok->is_special);
    free(tok->encode_ht.slots);
    free(tok->merge_ht.slots);
    free(tok->pool.data);
    free(tok);
}

/* ============================================================================
 * tok_encode
 * ========================================================================== */
int32_t *tok_encode(const Tokenizer *tok, const char *text,
                    bool add_bos, int *n_out) {
    if (!tok || !text || !n_out) return NULL;

    size_t text_len = strlen(text);

    /* Allocate output buffer (worst case: one token per byte + BOS) */
    int out_cap = (int)text_len + 2;
    if (out_cap < 64) out_cap = 64;
    int32_t *out = malloc((size_t)out_cap * sizeof(int32_t));
    if (!out) { TOK_ERR("OOM in tok_encode"); return NULL; }
    int n = 0;

    if (add_bos && tok->bos_id >= 0) out[n++] = (int32_t)tok->bos_id;

    /* Pre-tokenize */
    size_t starts[MAX_SEGS], lens[MAX_SEGS];
    int n_segs = _pretokenize(text, text_len, starts, lens);

    int32_t seg_out[TOK_MAX_SEG_BYTES];

    for (int s = 0; s < n_segs; s++) {
        const unsigned char *seg = (const unsigned char *)text + starts[s];
        size_t seg_len = lens[s];
        if (seg_len == 0) continue;

        /* Grow output buffer if needed */
        if (n + (int)seg_len + 2 > out_cap) {
            out_cap = (n + (int)seg_len + 2) * 2;
            int32_t *nd = realloc(out, (size_t)out_cap * sizeof(int32_t));
            if (!nd) { free(out); TOK_ERR("OOM"); return NULL; }
            out = nd;
        }

        int seg_n = _bpe_segment(tok, seg, seg_len, seg_out, TOK_MAX_SEG_BYTES);
        for (int i = 0; i < seg_n; i++) out[n++] = seg_out[i];
    }

    *n_out = n;
    return out;
}

/* ============================================================================
 * tok_encode_batch
 * ========================================================================== */
Tensor *tok_encode_batch(const Tokenizer *tok, const char **texts, int n_texts,
                          bool add_bos, int max_len) {
    if (!tok || !texts || n_texts <= 0) return NULL;

    /* Encode each text independently (parallel with OpenMP) */
    int32_t **encoded = calloc((size_t)n_texts, sizeof(int32_t *));
    int      *counts  = calloc((size_t)n_texts, sizeof(int));
    if (!encoded || !counts) {
        free(encoded); free(counts); TOK_ERR("OOM"); return NULL;
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int i = 0; i < n_texts; i++) {
        encoded[i] = tok_encode(tok, texts[i], add_bos, &counts[i]);
    }

    /* Determine padded length */
    int eff_max = max_len;
    if (eff_max <= 0) {
        for (int i = 0; i < n_texts; i++)
            if (counts[i] > eff_max) eff_max = counts[i];
    }
    if (eff_max <= 0) eff_max = 1;

    /* Allocate output tensor [n_texts × eff_max] INT32 */
    int shape[2] = { n_texts, eff_max };
    Tensor *out = tensor_create_dtype(2, shape, DTYPE_INT32);
    if (!out) {
        for (int i = 0; i < n_texts; i++) free(encoded[i]);
        free(encoded); free(counts);
        TOK_ERR("OOM allocating batch tensor");
        return NULL;
    }

    /* Fill with pad_id first */
    int32_t pad = (tok->pad_id >= 0) ? (int32_t)tok->pad_id : 0;
    int32_t *dst = (int32_t *)out->data;
    for (int i = 0; i < n_texts * eff_max; i++) dst[i] = pad;

    /* Copy each sequence, truncating to eff_max */
    for (int i = 0; i < n_texts; i++) {
        if (!encoded[i]) continue;
        int copy = counts[i] < eff_max ? counts[i] : eff_max;
        memcpy(dst + i * eff_max, encoded[i], (size_t)copy * sizeof(int32_t));
        free(encoded[i]);
    }

    free(encoded);
    free(counts);
    return out;
}

/* ============================================================================
 * tok_decode
 * ========================================================================== */

/* Reverse byte-to-unicode mapping: UTF-8 char → original byte.
 * We reconstruct this from the byte_str table. */
static uint8_t _unicode_to_byte(const Tokenizer *tok, const char *utf8_char) {
    /* Linear scan over 256 entries — called rarely, fine */
    for (int b = 0; b < 256; b++) {
        if (strcmp(tok->byte_str[b], utf8_char) == 0) return (uint8_t)b;
    }
    return (uint8_t)'?';
}

char *tok_decode(const Tokenizer *tok, const int32_t *ids, int n,
                 bool skip_special) {
    if (!tok || !ids || n <= 0) {
        char *e = malloc(1); if (e) e[0] = '\0'; return e;
    }

    /* Build the concatenated token string, then reverse-map GPT-2 unicode → bytes */
    size_t total = 0;
    for (int i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id < 0 || id >= tok->vocab_size) continue;
        if (skip_special && tok->is_special[id]) continue;
        const char *s = tok->id_to_str[id];
        if (s) total += strlen(s);
    }

    char *concat = malloc(total + 1);
    if (!concat) { TOK_ERR("OOM in tok_decode"); return NULL; }
    char *cp = concat;
    for (int i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id < 0 || id >= tok->vocab_size) continue;
        if (skip_special && tok->is_special[id]) continue;
        const char *s = tok->id_to_str[id];
        if (s) { size_t l = strlen(s); memcpy(cp, s, l); cp += l; }
    }
    *cp = '\0';

    /* Reverse-map: scan UTF-8, decode each char, look up in byte table */
    char   *out     = malloc(total + 1);
    size_t  out_len = 0;
    const unsigned char *p = (const unsigned char *)concat;
    const unsigned char *end = p + total;

    while (p < end) {
        uint32_t codepoint;
        int clen = _utf8_decode(p, &codepoint);
        if (clen <= 0) { p++; continue; }

        /* Check if this codepoint is in the byte-unicode range */
        bool found = false;
        for (int b = 0; b < 256; b++) {
            /* Match: re-encode the codepoint and compare with byte_str[b] */
            if ((int)strlen(tok->byte_str[b]) == clen &&
                memcmp(tok->byte_str[b], p, (size_t)clen) == 0) {
                out[out_len++] = (char)(uint8_t)b;
                found = true;
                break;
            }
        }
        if (!found) {
            /* Pass through as-is (multi-byte chars in added tokens) */
            memcpy(out + out_len, p, (size_t)clen);
            out_len += (size_t)clen;
        }
        p += clen;
    }
    out[out_len] = '\0';
    free(concat);
    return out;
}

/* ============================================================================
 * Accessors
 * ========================================================================== */
const char *tok_id_to_str(const Tokenizer *tok, int id) {
    if (!tok || id < 0 || id >= tok->vocab_size) return NULL;
    return tok->id_to_str[id];
}

int tok_str_to_id(const Tokenizer *tok, const char *str) {
    if (!tok || !str) return -1;
    return (int)_ht_lookup(&tok->encode_ht, str);
}

bool tok_is_special(const Tokenizer *tok, int id) {
    if (!tok || id < 0 || id >= tok->vocab_size) return false;
    return tok->is_special[id];
}

int tok_vocab_size(const Tokenizer *tok) { return tok ? tok->vocab_size : 0; }
int tok_bos_id(const Tokenizer *tok)     { return tok ? tok->bos_id     : -1; }
int tok_eos_id(const Tokenizer *tok)     { return tok ? tok->eos_id     : -1; }
int tok_pad_id(const Tokenizer *tok)     { return tok ? tok->pad_id     : -1; }
int tok_unk_id(const Tokenizer *tok)     { return tok ? tok->unk_id     : -1; }
