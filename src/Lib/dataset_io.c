/*
 * dataset_io.c — Unified CSV ingestion + Columnar DataFrame ETL
 *
 * This single translation unit owns:
 *
 *   tensor_dataset_from_csv()   — fast numeric-only path, returns [Tensor*, Tensor*]
 *                                  (legacy compatibility; still used internally)
 *
 *   df_read_csv()               — mixed-type columnar path via the DataFrame struct
 *   df_free / df_drop_nans /
 *   df_one_hot_encode /
 *   df_select_columns /
 *   df_to_tensor                — ETL pipeline functions
 *   df_num_rows / df_num_cols /
 *   df_col_*                    — safe PHP-facing introspection accessors
 *
 * Build: auto-included by TensorEngine's glob:
 *   gcc -O3 -mavx2 -mfma -ffast-math -fopenmp -shared -fPIC \
 *       -o libtensor.so tensor.c dataset_io.c <other>.c \
 *       -lopenblas -llapacke -lm
 */

#include "tensor.h"
#include "dataframe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

/* POSIX mmap — Linux / macOS */
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ============================================================================
 * Shared internal macros
 * ========================================================================== */

#define F32(t)         ((float   *)(t)->data)
#define DF_COL_F32(c)  ((float   *)(c)->data)
#define DF_COL_I32(c)  ((int32_t *)(c)->data)
#define DF_CAT_INIT    32

/* ============================================================================
 * 0.  Error bridge
 *     tensor_set_error() is defined in tensor.c (added alongside tensor_clear_error).
 *     All DF_ERR usages forward into the same global error slot that PHP checks
 *     via tensor_check_error() / tensor_get_last_error().
 * ========================================================================== */

/* Sets global error AND returns NULL in one statement */
#define DF_ERR(msg) do {                          \
    tensor_set_error("[DataFrame] " msg);         \
    return NULL;                                  \
} while (0)

#define DF_ERR_FMT(fmt, ...) do {                 \
    char _ebuf[256];                              \
    snprintf(_ebuf, sizeof(_ebuf),                \
             "[DataFrame] " fmt, ##__VA_ARGS__);  \
    tensor_set_error(_ebuf);                      \
    return NULL;                                  \
} while (0)

/* ============================================================================
 * 1.  mmap helpers  (shared by both CSV paths)
 * ========================================================================== */

static char *_mmap_open(const char *filepath, size_t *out_size) {
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat sb;
    if (fstat(fd, &sb) < 0 || sb.st_size == 0) { close(fd); return NULL; }

    *out_size = (size_t)sb.st_size;
    char *data = mmap(NULL, *out_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) return NULL;

    /* Sequential scan hint — kernel pre-fetches next pages automatically */
    madvise(data, *out_size, MADV_SEQUENTIAL | MADV_WILLNEED);
    return data;
}

static inline void _mmap_close(char *data, size_t size) {
    if (data && data != MAP_FAILED) munmap(data, size);
}

/* ============================================================================
 * 2.  RFC-4180 field reader
 *
 * Advances *pp past the field and its delimiter.
 * Returns a pointer to the field's content; *out_len is its byte count.
 * *out_eor is set true when this field was the last one on the row.
 *
 * scratch  — caller-supplied DF_FIELD_SCRATCH-byte buffer for unquoting.
 *            Returned pointer is into scratch only when the field contained
 *            escaped quotes (""); otherwise it points into the mmap region.
 * ========================================================================== */

static const char *_next_field(const char **pp, const char *end,
                                size_t *out_len, bool *out_eor,
                                char *scratch) {
    const char *p = *pp;

    if (p >= end) { *out_len = 0; *out_eor = true; *pp = p; return p; }

    /* ── Quoted field ─────────────────────────────────────────────────────── */
    if (*p == '"') {
        p++;                                /* skip opening quote             */
        const char *fs = p;
        bool has_escape = false;

        while (p < end) {
            if (*p == '"') {
                if (p + 1 < end && *(p + 1) == '"') { has_escape = true; p += 2; }
                else break;                 /* closing quote                  */
            } else { p++; }
        }
        size_t raw = (size_t)(p - fs);
        if (p < end) p++;                   /* skip closing quote             */

        const char *result;
        if (has_escape) {
            size_t j = 0;
            for (size_t i = 0; i < raw && j < DF_FIELD_SCRATCH - 1; i++) {
                if (fs[i] == '"' && i + 1 < raw && fs[i + 1] == '"')
                    { scratch[j++] = '"'; i++; }
                else scratch[j++] = fs[i];
            }
            scratch[j] = '\0';
            *out_len = j; result = scratch;
        } else {
            *out_len = raw; result = fs;
        }

        if (p < end && *p == '\r') p++;
        if (p < end && *p == '\n') { *out_eor = true; p++; }
        else if (p < end && *p == ',') { *out_eor = false; p++; }
        else *out_eor = true;

        *pp = p; return result;
    }

    /* ── Unquoted field ───────────────────────────────────────────────────── */
    const char *fs = p;
    while (p < end && *p != ',' && *p != '\n' && *p != '\r') p++;
    *out_len = (size_t)(p - fs);

    if (p < end && *p == '\r') p++;
    if (p < end && *p == '\n') { *out_eor = true;  p++; }
    else if (p < end && *p == ',') { *out_eor = false; p++; }
    else *out_eor = true;

    *pp = p; return fs;
}

/* ============================================================================
 * 3.  Type-detection helper
 *
 * Empty field → true (treated as NaN, still numeric).
 * A field is numeric iff strtof advances all non-whitespace bytes.
 * ========================================================================== */

static bool _is_numeric(const char *s, size_t len) {
    if (len == 0) return true;
    char *ep;
    strtof(s, &ep);
    while ((size_t)(ep - s) < len &&
           (*ep == ' ' || *ep == '\t' || *ep == '\r')) ep++;
    return (size_t)(ep - s) == len;
}

/* ============================================================================
 * 4.  LEGACY PATH: tensor_dataset_from_csv
 *
 * Numeric-only, two-pass, fgets-based parser.
 * Retained for internal compatibility; still called by the PHP fast path
 * when no ETL (dropNans / oneHotEncode) is requested.
 * ========================================================================== */

static void _csv_shape(const char *filepath, int has_header,
                        int *out_rows, int *out_cols) {
    FILE *fp = fopen(filepath, "r");
    if (!fp) { *out_rows = 0; *out_cols = 0; return; }

    char buf[65536];
    int rows = 0, cols = 0;

    if (fgets(buf, sizeof(buf), fp)) {
        cols = 1;
        for (int i = 0; buf[i]; i++) if (buf[i] == ',') cols++;
        if (!has_header) rows++;
    }
    while (fgets(buf, sizeof(buf), fp)) rows++;

    fclose(fp);
    *out_rows = rows; *out_cols = cols;
}

Tensor **tensor_dataset_from_csv(const char *filepath,
                                   int label_col, int has_header) {
    int rows = 0, cols = 0;
    _csv_shape(filepath, has_header, &rows, &cols);
    if (rows == 0 || cols == 0) return NULL;

    int feat_cols = (label_col >= 0) ? cols - 1 : cols;

    Tensor *samples = tensor_create_dtype(2, (int[]){rows, feat_cols}, DTYPE_FLOAT32);
    Tensor *labels  = (label_col >= 0)
                    ? tensor_create_dtype(1, (int[]){rows}, DTYPE_FLOAT32) : NULL;

    FILE *fp = fopen(filepath, "r");
    char buf[65536];
    if (has_header) fgets(buf, sizeof(buf), fp);

    for (int r = 0; r < rows && fgets(buf, sizeof(buf), fp); r++) {
        char *ptr = buf, *next;
        int cf = 0;
        for (int c = 0; c < cols; c++) {
            float val = strtof(ptr, &next);
            if (c == label_col) F32(labels)[r] = val;
            else { F32(samples)[r * feat_cols + cf] = val; cf++; }
            ptr = (*next == ',') ? next + 1 : next;
        }
    }
    fclose(fp);

    Tensor **out = (Tensor **)malloc(2 * sizeof(Tensor *));
    out[0] = samples; out[1] = labels;
    return out;
}

/* ============================================================================
 * 5.  DataFrame column helpers
 * ========================================================================== */

static void _col_free(DFColumn *col) {
    if (!col) return;
    free(col->data); col->data = NULL;
    if (col->categories) {
        for (int32_t i = 0; i < col->n_categories; i++) free(col->categories[i]);
        free(col->categories);
        col->categories = NULL; col->n_categories = 0; col->_cat_cap = 0;
    }
}

static bool _col_alloc(DFColumn *col, size_t n_rows) {
    size_t elem = (col->dtype == DF_DTYPE_FLOAT32) ? sizeof(float) : sizeof(int32_t);
    col->data = malloc(n_rows * elem);
    if (!col->data) return false;
    if (col->dtype == DF_DTYPE_STRING) {
        col->_cat_cap   = DF_CAT_INIT;
        col->categories = malloc(DF_CAT_INIT * sizeof(char *));
        if (!col->categories) { free(col->data); col->data = NULL; return false; }
    }
    return true;
}

/*
 * Deep-copy src column into dst (zero-initialised dst assumed).
 * dst->data is newly allocated; categories are strdup'd.
 */
static bool _col_copy(DFColumn *dst, const DFColumn *src, size_t n_rows) {
    memcpy(dst->name, src->name, DF_MAX_COL_NAME);
    dst->dtype = src->dtype;

    size_t elem  = (src->dtype == DF_DTYPE_FLOAT32) ? sizeof(float) : sizeof(int32_t);
    size_t bytes = n_rows * elem;
    dst->data    = malloc(bytes);
    if (!dst->data) return false;
    if (bytes) memcpy(dst->data, src->data, bytes);

    if (src->dtype == DF_DTYPE_STRING && src->categories) {
        dst->_cat_cap     = src->n_categories;
        dst->n_categories = src->n_categories;
        dst->categories   = malloc((size_t)src->n_categories * sizeof(char *));
        if (!dst->categories) { free(dst->data); return false; }
        for (int32_t i = 0; i < src->n_categories; i++) {
            dst->categories[i] = src->categories[i] ? strdup(src->categories[i]) : NULL;
        }
    }
    return true;
}

/*
 * Intern a string value into col->categories[].
 * Linear scan — correct for ML cardinality (< 1 000 distinct values).
 */
static int32_t _cat_intern(DFColumn *col, const char *s, size_t len) {
    for (int32_t i = 0; i < col->n_categories; i++) {
        if (col->categories[i] &&
            strlen(col->categories[i]) == len &&
            memcmp(col->categories[i], s, len) == 0) return i;
    }
    if (col->n_categories >= col->_cat_cap) {
        int32_t nc  = col->_cat_cap ? col->_cat_cap * 2 : DF_CAT_INIT;
        char  **tmp = realloc(col->categories, (size_t)nc * sizeof(char *));
        if (!tmp) return -1;
        col->categories = tmp; col->_cat_cap = nc;
    }
    char *stored = malloc(len + 1);
    if (!stored) return -1;
    memcpy(stored, s, len); stored[len] = '\0';
    col->categories[col->n_categories] = stored;
    return col->n_categories++;
}

/* ============================================================================
 * 6.  DataFrame CSV — Pass 1: metadata scan
 *
 * Single mmap scan: counts content rows, determines column count,
 * extracts header names, and decides DFDType per column.
 * ========================================================================== */

typedef struct {
    size_t   n_rows;
    size_t   n_cols;
    DFDType *col_types;   /* caller must free */
    char   **col_names;   /* caller must free each entry + the array */
} _CSVMeta;

static void _csv_meta_free(_CSVMeta *m) {
    free(m->col_types);
    if (m->col_names) {
        for (size_t c = 0; c < m->n_cols; c++) free(m->col_names[c]);
        free(m->col_names);
    }
}

static _CSVMeta _csv_scan(const char *data, size_t size, bool has_header) {
    _CSVMeta m = {0};
    const char *p = data, *end = data + size;
    char scratch[DF_FIELD_SCRATCH];

    /* ── Count columns from the very first line ───────────────────────────── */
    {
        const char *probe = p;
        bool eor = false;
        while (!eor && probe < end) {
            size_t len; _next_field(&probe, end, &len, &eor, scratch); m.n_cols++;
        }
        if (m.n_cols == 0) return m;
    }

    m.col_types = calloc(m.n_cols, sizeof(DFDType));   /* default = 0 = FLOAT32 */
    m.col_names = calloc(m.n_cols, sizeof(char *));
    bool *is_str = calloc(m.n_cols, sizeof(bool));

    if (!m.col_types || !m.col_names || !is_str) {
        free(is_str); _csv_meta_free(&m); memset(&m, 0, sizeof(m)); return m;
    }

    /* ── Extract (or generate) column names from first row ─────────────────── */
    {
        const char *hp = p;
        for (size_t c = 0; c < m.n_cols; c++) {
            size_t len; bool eor;
            const char *field = _next_field(&hp, end, &len, &eor, scratch);
            if (has_header && len > 0) {
                m.col_names[c] = malloc(len + 1);
                if (m.col_names[c]) { memcpy(m.col_names[c], field, len); m.col_names[c][len] = '\0'; }
            } else {
                m.col_names[c] = malloc(24);
                if (m.col_names[c]) snprintf(m.col_names[c], 24, "col_%zu", c);
            }
        }
        if (has_header) p = hp;     /* advance cursor past header row */
    }

    /* ── Scan data rows ────────────────────────────────────────────────────── */
    while (p < end) {
        bool eor = false, has_content = false;
        size_t c = 0;
        while (!eor) {
            size_t len;
            const char *field = _next_field(&p, end, &len, &eor, scratch);
            if (len > 0) {
                has_content = true;
                if (c < m.n_cols && !is_str[c] && !_is_numeric(field, len))
                    is_str[c] = true;
            }
            c++;
        }
        if (has_content) m.n_rows++;
    }

    for (size_t c = 0; c < m.n_cols; c++)
        m.col_types[c] = is_str[c] ? DF_DTYPE_STRING : DF_DTYPE_FLOAT32;

    free(is_str);
    return m;
}

/* ============================================================================
 * 7.  DataFrame CSV — Pass 2: fill column buffers
 * ========================================================================== */

static void _csv_fill(const char *data, size_t size, bool has_header, DataFrame *df) {
    const char *p = data, *end = data + size;
    char scratch[DF_FIELD_SCRATCH];

    if (has_header) {   /* skip header row */
        bool eor = false;
        while (!eor && p < end) { size_t len; _next_field(&p, end, &len, &eor, scratch); }
    }

    size_t row = 0;
    while (p < end && row < df->n_rows) {
        bool eor = false, has_content = false;
        size_t c = 0;
        while (!eor) {
            size_t len;
            const char *field = _next_field(&p, end, &len, &eor, scratch);
            if (len > 0) has_content = true;
            if (c < df->n_cols) {
                DFColumn *col = &df->columns[c];
                switch (col->dtype) {
                case DF_DTYPE_FLOAT32:
                    DF_COL_F32(col)[row] = (len == 0) ? NAN : strtof(field, NULL);
                    break;
                case DF_DTYPE_INT32:
                    DF_COL_I32(col)[row] = (len == 0) ? INT32_MIN
                                         : (int32_t)strtol(field, NULL, 10);
                    break;
                case DF_DTYPE_STRING:
                    DF_COL_I32(col)[row] = (len == 0) ? -1 : _cat_intern(col, field, len);
                    break;
                }
            }
            c++;
        }
        if (has_content) row++;
    }
}

/* ============================================================================
 * 8.  PUBLIC: DataFrame lifecycle
 * ========================================================================== */

DataFrame *df_create(size_t n_rows, size_t n_cols) {
    if (n_cols == 0) return NULL;
    DataFrame *df = malloc(sizeof(DataFrame));
    if (!df) return NULL;
    df->n_rows   = n_rows;
    df->n_cols   = n_cols;
    df->columns  = calloc(n_cols, sizeof(DFColumn));
    if (!df->columns) { free(df); return NULL; }
    return df;
}

void df_free(DataFrame *df) {
    if (!df) return;
    for (size_t c = 0; c < df->n_cols; c++) _col_free(&df->columns[c]);
    free(df->columns);
    free(df);
}

/* ============================================================================
 * 9.  PUBLIC: df_read_csv
 *
 * Mixed-type path.  PHP holds only a DataFrame* — an opaque pointer.
 * All struct internals are accessed via df_* accessor functions.
 * ========================================================================== */

DataFrame *df_read_csv(const char *filepath, bool has_header) {
    if (!filepath) DF_ERR("df_read_csv: NULL filepath");

    size_t fsz = 0;
    char *fdata = _mmap_open(filepath, &fsz);
    if (!fdata) DF_ERR("df_read_csv: cannot open / mmap file");

    /* Pass 1 ─────────────────────────────────────────────────────────────── */
    _CSVMeta meta = _csv_scan(fdata, fsz, has_header);
    if (meta.n_rows == 0 || meta.n_cols == 0) {
        _mmap_close(fdata, fsz); _csv_meta_free(&meta);
        DF_ERR("df_read_csv: empty or unreadable CSV");
    }

    /* Allocate frame + column buffers ─────────────────────────────────────── */
    DataFrame *df = df_create(meta.n_rows, meta.n_cols);
    if (!df) { _mmap_close(fdata, fsz); _csv_meta_free(&meta); return NULL; }

    for (size_t c = 0; c < meta.n_cols; c++) {
        DFColumn *col = &df->columns[c];
        if (meta.col_names[c]) {
            strncpy(col->name, meta.col_names[c], DF_MAX_COL_NAME - 1);
        } else {
            snprintf(col->name, DF_MAX_COL_NAME, "col_%zu", c);
        }
        col->dtype = meta.col_types[c];
        if (!_col_alloc(col, meta.n_rows)) {
            _mmap_close(fdata, fsz); _csv_meta_free(&meta); df_free(df); return NULL;
        }
    }

    /* Pass 2 ─────────────────────────────────────────────────────────────── */
    _csv_fill(fdata, fsz, has_header, df);

    _mmap_close(fdata, fsz);
    _csv_meta_free(&meta);
    return df;
}

/* ============================================================================
 * 10. PUBLIC: ETL — df_select_columns
 * ========================================================================== */

DataFrame *df_select_columns(const DataFrame *df, const int *col_indices, int n) {
    if (!df || !col_indices || n <= 0) return NULL;
    DataFrame *out = df_create(df->n_rows, (size_t)n);
    if (!out) return NULL;
    for (int i = 0; i < n; i++) {
        int idx = col_indices[i];
        if (idx < 0 || (size_t)idx >= df->n_cols) {
            df_free(out); DF_ERR_FMT("df_select_columns: index %d out of range", idx);
        }
        if (!_col_copy(&out->columns[i], &df->columns[idx], df->n_rows)) {
            df_free(out); DF_ERR("df_select_columns: allocation failure");
        }
    }
    return out;
}

/* ============================================================================
 * 11. PUBLIC: ETL — df_drop_nans
 *
 * Builds a keep-mask in one pass, then copies valid rows in a second pass.
 * O(n_rows × n_cols) — both passes are sequential column-major reads.
 * ========================================================================== */

DataFrame *df_drop_nans(const DataFrame *df) {
    if (!df) return NULL;

    bool *keep = malloc(df->n_rows * sizeof(bool));
    if (!keep) return NULL;

    size_t valid = 0;
    for (size_t r = 0; r < df->n_rows; r++) {
        keep[r] = true;
        for (size_t c = 0; c < df->n_cols && keep[r]; c++) {
            const DFColumn *col = &df->columns[c];
            if      (col->dtype == DF_DTYPE_FLOAT32 && isnan(DF_COL_F32(col)[r])) keep[r] = false;
            else if (col->dtype == DF_DTYPE_INT32   && DF_COL_I32(col)[r] == INT32_MIN) keep[r] = false;
            else if (col->dtype == DF_DTYPE_STRING  && DF_COL_I32(col)[r] < 0)      keep[r] = false;
        }
        if (keep[r]) valid++;
    }

    DataFrame *out = df_create(valid, df->n_cols);
    if (!out) { free(keep); return NULL; }

    /* Allocate column buffers + deep-copy metadata */
    for (size_t c = 0; c < df->n_cols; c++) {
        const DFColumn *src = &df->columns[c];
        DFColumn       *dst = &out->columns[c];
        memcpy(dst->name, src->name, DF_MAX_COL_NAME);
        dst->dtype = src->dtype;
        size_t elem = (src->dtype == DF_DTYPE_FLOAT32) ? sizeof(float) : sizeof(int32_t);
        dst->data = malloc(valid * elem);
        if (!dst->data) { free(keep); df_free(out); return NULL; }
        if (src->dtype == DF_DTYPE_STRING && src->categories) {
            dst->_cat_cap = dst->n_categories = src->n_categories;
            dst->categories = malloc((size_t)src->n_categories * sizeof(char *));
            if (dst->categories) {
                for (int32_t i = 0; i < src->n_categories; i++)
                    dst->categories[i] = src->categories[i] ? strdup(src->categories[i]) : NULL;
            }
        }
    }

    size_t out_row = 0;
    for (size_t r = 0; r < df->n_rows; r++) {
        if (!keep[r]) continue;
        for (size_t c = 0; c < df->n_cols; c++) {
            const DFColumn *src = &df->columns[c];
            DFColumn       *dst = &out->columns[c];
            if (src->dtype == DF_DTYPE_FLOAT32) DF_COL_F32(dst)[out_row] = DF_COL_F32(src)[r];
            else                                DF_COL_I32(dst)[out_row] = DF_COL_I32(src)[r];
        }
        out_row++;
    }

    free(keep);
    return out;
}

/* ============================================================================
 * 12. PUBLIC: ETL — df_one_hot_encode
 *
 * Replaces the STRING column at col_idx with K FLOAT32 binary columns
 * named "{col_name}_{category}".  All other columns are deep-copied.
 * The new columns are placed at the same position as the original.
 * ========================================================================== */

DataFrame *df_one_hot_encode(const DataFrame *df, int col_idx) {
    if (!df) return NULL;
    if (col_idx < 0 || (size_t)col_idx >= df->n_cols)
        DF_ERR_FMT("df_one_hot_encode: col_idx %d out of range", col_idx);
    const DFColumn *cat = &df->columns[col_idx];
    if (cat->dtype != DF_DTYPE_STRING)
        DF_ERR("df_one_hot_encode: target column is not STRING type");

    int32_t K        = cat->n_categories;
    size_t  n_rows   = df->n_rows;
    size_t  new_cols = df->n_cols - 1 + (size_t)K;

    DataFrame *out = df_create(n_rows, new_cols);
    if (!out) return NULL;

    size_t oc = 0;
    for (size_t c = 0; c < df->n_cols; c++) {
        if (c == (size_t)col_idx) {
            /* Expand STRING column → K binary FLOAT32 columns */
            const int32_t *idx_data = DF_COL_I32(cat);
            for (int32_t k = 0; k < K; k++) {
                DFColumn *dst = &out->columns[oc];
                snprintf(dst->name, DF_MAX_COL_NAME, "%s_%s",
                         cat->name, cat->categories[k] ? cat->categories[k] : "");
                dst->dtype = DF_DTYPE_FLOAT32;
                dst->data  = malloc(n_rows * sizeof(float));
                if (!dst->data) { df_free(out); return NULL; }
                float *ohe = DF_COL_F32(dst);
                /* Branchless fill — compiles to CMOV / vectorised compare */
                for (size_t r = 0; r < n_rows; r++)
                    ohe[r] = (idx_data[r] == k) ? 1.0f : 0.0f;
                oc++;
            }
        } else {
            if (!_col_copy(&out->columns[oc], &df->columns[c], n_rows)) {
                df_free(out); return NULL;
            }
            oc++;
        }
    }
    return out;
}

/* ============================================================================
 * 13. PUBLIC: df_to_tensor
 *
 * Packs numeric columns into a row-major [n_rows × n] FLOAT32 Tensor.
 * Column-by-column write keeps the destination cache-line warm across rows.
 * ========================================================================== */

Tensor *df_to_tensor(const DataFrame *df, const int *col_indices, int n) {
    if (!df || !col_indices || n <= 0) return NULL;

    for (int i = 0; i < n; i++) {
        int idx = col_indices[i];
        if (idx < 0 || (size_t)idx >= df->n_cols)
            DF_ERR_FMT("df_to_tensor: column index %d out of range", idx);
        if (df->columns[idx].dtype == DF_DTYPE_STRING)
            DF_ERR("df_to_tensor: STRING column cannot be packed directly — call df_one_hot_encode first");
    }

    size_t n_rows   = df->n_rows;
    int    shape[2] = { (int)n_rows, n };
    Tensor *out     = tensor_create_dtype(2, shape, DTYPE_FLOAT32);
    if (!out) return NULL;

    float *dst = F32(out);
    for (int ci = 0; ci < n; ci++) {
        const DFColumn *col = &df->columns[col_indices[ci]];
        if (col->dtype == DF_DTYPE_FLOAT32) {
            const float *src = DF_COL_F32(col);
            for (size_t r = 0; r < n_rows; r++) dst[r * (size_t)n + (size_t)ci] = src[r];
        } else {
            const int32_t *src = DF_COL_I32(col);
            for (size_t r = 0; r < n_rows; r++) dst[r * (size_t)n + (size_t)ci] = (float)src[r];
        }
    }
    return out;
}

/* ============================================================================
 * 14. PUBLIC: Introspection  (opaque accessors for the PHP FFI layer)
 * ========================================================================== */

size_t      df_num_rows(const DataFrame *df)           { return df ? df->n_rows : 0; }
size_t      df_num_cols(const DataFrame *df)           { return df ? df->n_cols : 0; }

const char *df_col_name(const DataFrame *df, int idx) {
    if (!df || idx < 0 || (size_t)idx >= df->n_cols) return NULL;
    return df->columns[idx].name;
}
int df_col_dtype(const DataFrame *df, int idx) {
    if (!df || idx < 0 || (size_t)idx >= df->n_cols) return -1;
    return (int)df->columns[idx].dtype;
}
int df_col_n_categories(const DataFrame *df, int idx) {
    if (!df || idx < 0 || (size_t)idx >= df->n_cols) return 0;
    return (int)df->columns[idx].n_categories;
}
const char *df_col_category_name(const DataFrame *df, int col_idx, int cat_idx) {
    if (!df || col_idx < 0 || (size_t)col_idx >= df->n_cols) return NULL;
    const DFColumn *col = &df->columns[col_idx];
    if (cat_idx < 0 || cat_idx >= col->n_categories) return NULL;
    return col->categories[cat_idx];
}
