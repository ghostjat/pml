/*
 * dataset_io.c — High‑Performance DataFrame & NLP ETL
 *
 * Features:
 *   - mmap‑backed CSV parsing (RFC‑4180)
 *   - Columnar storage with string interning
 *   - Zero‑copy ETL: dropNans, oneHotEncode, selectColumns
 *   - NLP: vocabulary building, bag‑of‑words transformation
 *   - Zero PHP↔C data copies; all heavy data stays in C
 *
 * Build:
 *   gcc -O3 -mavx2 -mfma -ffast-math -fopenmp -shared -fPIC \
 *       -o libtensor.so tensor.c dataset_io.c ... -lopenblas -lm
 */

#include "tensor.h"
#include "dataframe.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ============================================================================
 * Macros & Shared Helpers
 * ========================================================================== */

#define F32(t)         ((float   *)(t)->data)
#define DF_COL_F32(c)  ((float   *)(c)->data)
#define DF_COL_I32(c)  ((int32_t *)(c)->data)
#define DF_CAT_INIT    32

/* Error reporting – bridges to tensor.c */
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
 * 1. mmap & RFC‑4180 CSV Parser
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

    madvise(data, *out_size, MADV_SEQUENTIAL | MADV_WILLNEED);
    return data;
}

static inline void _mmap_close(char *data, size_t size) {
    if (data && data != MAP_FAILED) munmap(data, size);
}
/* Bit-level NaN check — immune to -ffast-math which breaks isnan()/isfinite() */
static inline int _f32_is_nan(float x) {
    uint32_t bits; memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x7FFFFFFFu) > 0x7F800000u;
}

/*
 * RFC‑4180 field reader.
 * Advances *pp past the field and its delimiter.
 * Returns pointer to field content; *out_len is byte count.
 * *out_eor = true if this field ended the row.
 * scratch is a caller‑provided buffer for unescaping quotes.
 */
static const char *_next_field(const char **pp, const char *end,
                               size_t *out_len, bool *out_eor,
                               char *scratch) {
    const char *p = *pp;
    if (p >= end) { *out_len = 0; *out_eor = true; *pp = p; return p; }

    /* Quoted field */
    if (*p == '"') {
        p++;
        const char *fs = p;
        bool has_escape = false;

        while (p < end) {
            if (*p == '"') {
                if (p + 1 < end && *(p + 1) == '"') { has_escape = true; p += 2; }
                else break;
            } else { p++; }
        }
        size_t raw = (size_t)(p - fs);
        if (p < end) p++;  // skip closing quote

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

    /* Unquoted field */
    const char *fs = p;
    while (p < end && *p != ',' && *p != '\n' && *p != '\r') p++;
    *out_len = (size_t)(p - fs);

    if (p < end && *p == '\r') p++;
    if (p < end && *p == '\n') { *out_eor = true;  p++; }
    else if (p < end && *p == ',') { *out_eor = false; p++; }
    else *out_eor = true;

    *pp = p; return fs;
}

static bool _is_numeric(const char *s, size_t len) {
    if (len == 0) return true;
    char *ep;
    strtof(s, &ep);
    while ((size_t)(ep - s) < len &&
           (*ep == ' ' || *ep == '\t' || *ep == '\r')) ep++;
    return (size_t)(ep - s) == len;
}

/* ============================================================================
 * LEGACY PATH: tensor_dataset_from_csv (numeric-only fast path)
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
 * 2. DataFrame Column Helpers
 * ========================================================================== */

/* Forward declarations — ht_* are defined in section 7 (NLP) */
static HashTable  *ht_create(void);
static HashEntry  *ht_put(HashTable *ht, const char *key, size_t len, int value);
static HashEntry  *ht_get(HashTable *ht, const char *key, size_t len);
static void        ht_free(HashTable *ht);

static void _col_free(DFColumn *col) {
    if (!col) return;
    free(col->data);
    col->data = NULL;
    if (col->cat_map) { ht_free(col->cat_map); col->cat_map = NULL; }
    if (col->categories) {
        for (int32_t i = 0; i < col->n_categories; i++) free(col->categories[i]);
        free(col->categories);
        col->categories = NULL;
        col->n_categories = 0;
        col->_cat_cap = 0;
    }
}

static bool _col_alloc(DFColumn *col, size_t n_rows) {
    size_t elem = (col->dtype == DF_DTYPE_FLOAT32) ? sizeof(float) : sizeof(int32_t);
    col->data = malloc(n_rows * elem);
    if (!col->data) return false;
    if (col->dtype == DF_DTYPE_STRING) {
        col->_cat_cap = DF_CAT_INIT;
        col->categories = malloc(DF_CAT_INIT * sizeof(char *));
        if (!col->categories) { free(col->data); col->data = NULL; return false; }
        col->n_categories = 0;
        col->cat_map = ht_create();
        if (!col->cat_map) { free(col->categories); free(col->data); col->data = NULL; return false; }
    }
    return true;
}

static bool _col_copy(DFColumn *dst, const DFColumn *src, size_t n_rows) {
    memcpy(dst->name, src->name, DF_MAX_COL_NAME);
    dst->dtype = src->dtype;

    size_t elem  = (src->dtype == DF_DTYPE_FLOAT32) ? sizeof(float) : sizeof(int32_t);
    size_t bytes = n_rows * elem;
    dst->data = malloc(bytes);
    if (!dst->data) return false;
    if (bytes) memcpy(dst->data, src->data, bytes);

    if (src->dtype == DF_DTYPE_STRING && src->categories) {
        dst->_cat_cap = dst->n_categories = src->n_categories;
        dst->categories = malloc((size_t)src->n_categories * sizeof(char *));
        if (!dst->categories) { free(dst->data); return false; }
        dst->cat_map = ht_create();
        for (int32_t i = 0; i < src->n_categories; i++) {
            dst->categories[i] = src->categories[i] ? strdup(src->categories[i]) : NULL;
            if (dst->categories[i])
                ht_put(dst->cat_map, dst->categories[i], strlen(dst->categories[i]), i);
        }
    }
    return true;
}

static int32_t _cat_intern(DFColumn *col, const char *s, size_t len) {
    /* O(1) lookup via hash map — eliminates the O(n²) linear scan */
    HashEntry *e = ht_get(col->cat_map, s, len);
    if (e) return (int32_t)e->value;

    /* New category: grow the categories array if needed */
    if (col->n_categories >= col->_cat_cap) {
        int32_t nc  = col->_cat_cap ? col->_cat_cap * 2 : DF_CAT_INIT;
        char  **tmp = realloc(col->categories, (size_t)nc * sizeof(char *));
        if (!tmp) return -1;
        col->categories = tmp; col->_cat_cap = nc;
    }
    char *stored = malloc(len + 1);
    if (!stored) return -1;
    memcpy(stored, s, len); stored[len] = '\0';
    int32_t idx = col->n_categories;
    col->categories[idx] = stored;
    col->n_categories++;
    ht_put(col->cat_map, stored, len, (int)idx);
    return idx;
}

/* ============================================================================
 * 3. CSV Scanning & Metadata (Pass 1)
 * ========================================================================== */

typedef struct {
    size_t   n_rows;
    size_t   n_cols;
    DFDType *col_types;   /* caller must free */
    char   **col_names;   /* caller must free each entry + array */
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

    /* Count columns from first line */
    {
        const char *probe = p;
        bool eor = false;
        while (!eor && probe < end) {
            size_t len; _next_field(&probe, end, &len, &eor, scratch); m.n_cols++;
        }
        if (m.n_cols == 0) return m;
    }

    m.col_types = calloc(m.n_cols, sizeof(DFDType));
    m.col_names = calloc(m.n_cols, sizeof(char *));
    bool *is_str = calloc(m.n_cols, sizeof(bool));

    if (!m.col_types || !m.col_names || !is_str) {
        free(is_str); _csv_meta_free(&m); memset(&m, 0, sizeof(m)); return m;
    }

    /* Extract/generate column names */
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
        if (has_header) p = hp;
    }

    /* Scan data rows, detect non‑numeric columns */
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
 * 4. CSV Data Filling (Pass 2)
 * ========================================================================== */

static void _csv_fill(const char *data, size_t size, bool has_header, DataFrame *df) {
    const char *p = data, *end = data + size;
    char scratch[DF_FIELD_SCRATCH];

    if (has_header) {
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
 * 5. DataFrame Lifecycle
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

DataFrame *df_read_csv(const char *filepath, bool has_header) {
    if (!filepath) DF_ERR("df_read_csv: NULL filepath");

    size_t fsz = 0;
    char *fdata = _mmap_open(filepath, &fsz);
    if (!fdata) DF_ERR("df_read_csv: cannot open / mmap file");

    _CSVMeta meta = _csv_scan(fdata, fsz, has_header);
    if (meta.n_rows == 0 || meta.n_cols == 0) {
        _mmap_close(fdata, fsz); _csv_meta_free(&meta);
        DF_ERR("df_read_csv: empty or unreadable CSV");
    }

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

    _csv_fill(fdata, fsz, has_header, df);
    _mmap_close(fdata, fsz);
    _csv_meta_free(&meta);
    return df;
}

/* ============================================================================
 * 6. ETL Operations
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

DataFrame *df_drop_nans(const DataFrame *df) {
    if (!df) return NULL;

    bool *keep = malloc(df->n_rows * sizeof(bool));
    if (!keep) return NULL;

    size_t valid = 0;
    for (size_t r = 0; r < df->n_rows; r++) {
        keep[r] = true;
        for (size_t c = 0; c < df->n_cols && keep[r]; c++) {
            const DFColumn *col = &df->columns[c];
            if      (col->dtype == DF_DTYPE_FLOAT32 && _f32_is_nan(DF_COL_F32(col)[r])) keep[r] = false;
            else if (col->dtype == DF_DTYPE_INT32   && DF_COL_I32(col)[r] == INT32_MIN) keep[r] = false;
            else if (col->dtype == DF_DTYPE_STRING  && DF_COL_I32(col)[r] < 0)      keep[r] = false;
        }
        if (keep[r]) valid++;
    }

    DataFrame *out = df_create(valid, df->n_cols);
    if (!out) { free(keep); return NULL; }

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

DataFrame *df_slice_rows(const DataFrame *df, size_t offset, size_t n) {
    if (!df) return NULL;
    if (offset >= df->n_rows) DF_ERR("df_slice_rows: offset >= n_rows");
    if (n == 0) DF_ERR("df_slice_rows: n must be > 0");
    if (offset + n > df->n_rows) n = df->n_rows - offset;

    DataFrame *out = df_create(n, df->n_cols);
    if (!out) return NULL;

    for (size_t c = 0; c < df->n_cols; c++) {
        const DFColumn *src = &df->columns[c];
        DFColumn       *dst = &out->columns[c];

        memcpy(dst->name, src->name, DF_MAX_COL_NAME);
        dst->dtype = src->dtype;

        size_t elem = (src->dtype == DF_DTYPE_FLOAT32) ? sizeof(float) : sizeof(int32_t);
        dst->data = malloc(n * elem);
        if (!dst->data) { df_free(out); return NULL; }
        /* Copy the slice starting at offset */
        memcpy(dst->data, (const char *)src->data + offset * elem, n * elem);

        if (src->dtype == DF_DTYPE_STRING) {
            /* Compact category table: keep only entries used by this slice */
            bool *used = calloc((size_t)src->n_categories, sizeof(bool));
            if (!used) { df_free(out); return NULL; }
            const int32_t *idx_data = (const int32_t *)dst->data;
            for (size_t r = 0; r < n; r++)
                if (idx_data[r] >= 0 && idx_data[r] < src->n_categories)
                    used[idx_data[r]] = true;

            int32_t *remap = malloc((size_t)src->n_categories * sizeof(int32_t));
            if (!remap) { free(used); df_free(out); return NULL; }
            dst->cat_map   = ht_create();
            dst->_cat_cap  = src->n_categories;
            dst->categories = malloc((size_t)src->n_categories * sizeof(char *));
            if (!dst->categories || !dst->cat_map) {
                free(used); free(remap); df_free(out); return NULL;
            }
            dst->n_categories = 0;
            for (int32_t i = 0; i < src->n_categories; i++) {
                if (used[i] && src->categories[i]) {
                    size_t len = strlen(src->categories[i]);
                    dst->categories[dst->n_categories] = strdup(src->categories[i]);
                    ht_put(dst->cat_map, dst->categories[dst->n_categories], len,
                           (int)dst->n_categories);
                    remap[i] = dst->n_categories++;
                } else {
                    remap[i] = -1;
                }
            }
            int32_t *rows = (int32_t *)dst->data;
            for (size_t r = 0; r < n; r++)
                rows[r] = (rows[r] >= 0) ? remap[rows[r]] : -1;

            free(used);
            free(remap);
        }
    }
    return out;
}

DataFrame *df_head_rows(const DataFrame *df, size_t n) {
    return df_slice_rows(df, 0, n);
}

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
            const int32_t *idx_data = DF_COL_I32(cat);
            for (int32_t k = 0; k < K; k++) {
                DFColumn *dst = &out->columns[oc];
                snprintf(dst->name, DF_MAX_COL_NAME, "%s_%s",
                         cat->name, cat->categories[k] ? cat->categories[k] : "");
                dst->dtype = DF_DTYPE_FLOAT32;
                dst->data  = malloc(n_rows * sizeof(float));
                if (!dst->data) { df_free(out); return NULL; }
                float *ohe = DF_COL_F32(dst);
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
 * 7. NLP: Tokenization & Vocabulary (only one copy)
 * ========================================================================== */

/* Forward declaration — defined in section 9 (C Transform Pipeline).
 * Placed here so df_vocab_build and df_transform_bow can use it below. */
static int _token_next_nb(const char **pp, char *buf, int cap);

/*
 * Fast tokenizer – splits on non‑alphanumeric characters.
 * Lowercases and returns a malloc'd copy of each token.
 * Caller must free each token.
 */
static char* _token_next(const char **pp) {
    const char *p = *pp;
    while (*p && !isalnum((unsigned char)*p)) p++;
    if (!*p) return NULL;
    const char *start = p;
    while (*p && isalnum((unsigned char)*p)) p++;
    size_t len = p - start;
    char *tok = malloc(len + 1);
    for (size_t i = 0; i < len; i++) tok[i] = tolower((unsigned char)start[i]);
    tok[len] = '\0';
    *pp = p;
    return tok;
}

/* WordCount used during vocabulary building */
typedef struct WordCount {
    char*  word;
    int    count;
} WordCount;

static uint32_t _hash(const char* s, size_t len) {
    uint32_t h = 5381;
    for (size_t i = 0; i < len; i++) h = ((h << 5) + h) ^ (unsigned char)s[i];
    return h;
}

static HashTable* ht_create(void) {
    HashTable* ht = malloc(sizeof(HashTable));
    ht->capacity = 4096;
    ht->size = 0;
    ht->buckets = calloc(ht->capacity, sizeof(HashEntry*));
    return ht;
}

static HashEntry* ht_put(HashTable* ht, const char* key, size_t len, int value) {
    uint32_t idx = _hash(key, len) % ht->capacity;
    HashEntry* e = ht->buckets[idx];
    while (e) {
        if (e->len == len && memcmp(e->key, key, len) == 0) {
            e->value = value;
            return e;
        }
        e = e->next;
    }
    e = malloc(sizeof(HashEntry));
    char* kcopy = malloc(len + 1);
    memcpy(kcopy, key, len); kcopy[len] = '\0';
    e->key = kcopy;
    e->len = len;
    e->value = value;
    e->next = ht->buckets[idx];
    ht->buckets[idx] = e;
    ht->size++;
    return e;
}

static HashEntry* ht_get(HashTable* ht, const char* key, size_t len) {
    uint32_t idx = _hash(key, len) % ht->capacity;
    HashEntry* e = ht->buckets[idx];
    while (e) {
        if (e->len == len && memcmp(e->key, key, len) == 0) return e;
        e = e->next;
    }
    return NULL;
}

static void ht_free(HashTable* ht) {
    for (size_t i = 0; i < ht->capacity; i++) {
        HashEntry* e = ht->buckets[i];
        while (e) {
            HashEntry* next = e->next;
            free((void*)e->key);
            free(e);
            e = next;
        }
    }
    free(ht->buckets);
    free(ht);
}

/* Build vocabulary from a STRING column */
Vocab* df_vocab_build(const DataFrame* df, int col_idx, int max_features) {
    if (!df) DF_ERR("df_vocab_build: NULL DataFrame");
    if (col_idx < 0 || (size_t)col_idx >= df->n_cols)
        DF_ERR_FMT("df_vocab_build: column index %d out of range", col_idx);
    const DFColumn* col = &df->columns[col_idx];
    if (col->dtype != DF_DTYPE_STRING)
        DF_ERR("df_vocab_build: column is not STRING");

    HashTable* freq = ht_create();
    const int32_t* cat_idx = DF_COL_I32(col);
    char tok[512];
    for (size_t r = 0; r < df->n_rows; r++) {
        int32_t idx = cat_idx[r];
        if (idx < 0) continue;
        const char* text = col->categories[idx];
        if (!text) continue;

        const char* p = text;
        int len;
        while ((len = _token_next_nb(&p, tok, (int)sizeof(tok))) > 0) {
            HashEntry* e = ht_get(freq, tok, (size_t)len);
            if (e) e->value++;
            else ht_put(freq, tok, (size_t)len, 1);
        }
    }

    int total = (int)freq->size;
    WordCount* wc = malloc(total * sizeof(WordCount));
    int n = 0;
    for (size_t i = 0; i < freq->capacity; i++) {
        for (HashEntry* e = freq->buckets[i]; e; e = e->next) {
            wc[n].word = (char*)e->key;
            wc[n].count = e->value;
            n++;
        }
    }
    for (int i = 1; i < n; i++) {
        WordCount tmp = wc[i];
        int j = i - 1;
        while (j >= 0 && wc[j].count < tmp.count) {
            wc[j+1] = wc[j];
            j--;
        }
        wc[j+1] = tmp;
    }

    int vocab_size = (max_features > 0 && max_features < n) ? max_features : n;

    Vocab* v = malloc(sizeof(Vocab));
    v->map = ht_create();
    v->size = vocab_size;
    v->words = malloc(vocab_size * sizeof(char*));

    for (int i = 0; i < vocab_size; i++) {
        v->words[i] = strdup(wc[i].word);
        ht_put(v->map, wc[i].word, strlen(wc[i].word), i);
    }

    free(wc);
    ht_free(freq);
    return v;
}

void vocab_free(Vocab* v) {
    if (!v) return;
    ht_free(v->map);
    for (int i = 0; i < v->size; i++) free(v->words[i]);
    free(v->words);
    free(v);
}

int vocab_size(const Vocab* v) {
    return v ? v->size : 0;
}

/* ============================================================================
 * Vocabulary Persistence (for PHP serialization)
 * ========================================================================== */

void vocab_save(Vocab* v, const char* filepath) {
    if (!v || !filepath) return;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return;

    // Write vocab size
    fwrite(&v->size, sizeof(int), 1, fp);

    // Write each word
    for (int i = 0; i < v->size; i++) {
        size_t len = strlen(v->words[i]);
        fwrite(&len, sizeof(size_t), 1, fp);
        fwrite(v->words[i], 1, len, fp);
    }
    fclose(fp);
}

Vocab* vocab_load(const char* filepath) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return NULL;

    Vocab* v = malloc(sizeof(Vocab));
    fread(&v->size, sizeof(int), 1, fp);

    v->words = malloc(v->size * sizeof(char*));
    v->map = ht_create();

    for (int i = 0; i < v->size; i++) {
        size_t len;
        fread(&len, sizeof(size_t), 1, fp);
        v->words[i] = malloc(len + 1);
        fread(v->words[i], 1, len, fp);
        v->words[i][len] = '\0';
        ht_put(v->map, v->words[i], len, i);
    }
    fclose(fp);
    return v;
}

Tensor* df_transform_bow(const DataFrame* df, int col_idx, const Vocab* v) {
    if (!df || !v) DF_ERR("df_transform_bow: NULL arguments");
    if (col_idx < 0 || (size_t)col_idx >= df->n_cols)
        DF_ERR_FMT("df_transform_bow: column index %d out of range", col_idx);
    const DFColumn* col = &df->columns[col_idx];
    if (col->dtype != DF_DTYPE_STRING)
        DF_ERR("df_transform_bow: column is not STRING");

    size_t n_rows = df->n_rows;
    int shape[2] = { (int)n_rows, v->size };
    Tensor* out = tensor_create_dtype(2, shape, DTYPE_FLOAT32);
    if (!out) return NULL;
    float* data = (float*)out->data;
    /* tensor_create_dtype zeroes — no memset needed */

    const int32_t* cat_idx = DF_COL_I32(col);

#pragma omp parallel
    {
        char tok[512];
#pragma omp for schedule(dynamic, 64)
        for (size_t r = 0; r < n_rows; r++) {
            int32_t idx = cat_idx[r];
            if (idx < 0) continue;
            const char* text = col->categories[idx];
            if (!text) continue;

            const char* p   = text;
            float*      row = data + r * v->size;
            int len;
            while ((len = _token_next_nb(&p, tok, (int)sizeof(tok))) > 0) {
                HashEntry* e = ht_get(v->map, tok, (size_t)len);
                if (e) row[e->value] += 1.0f;
            }
        }
    }
    return out;
}

/* ============================================================================
 * 9. C Transform Pipeline  (AVX2 + OpenMP, zero-alloc per batch)
 *
 * API:
 *   df_fit_transformers()     — 2-pass parallel IDF + ZScale fit
 *   pipeline_create/free()    — opaque handle (borrows vocab/idf/stds)
 *   pipeline_transform_batch()— BoW→TfIdf→ZScale→OneHot in one C call
 * ========================================================================== */

#ifdef _OPENMP
#include <omp.h>
#endif

/* Non-allocating tokenizer identical in behaviour to _token_next().
 * Writes lowercased alnum token into buf[0..cap-1], returns token length.
 * Returns 0 when the string is exhausted.  Advances *pp past the token. */
static int _token_next_nb(const char **pp, char *buf, int cap)
{
    const char *p = *pp;
    while (*p && !isalnum((unsigned char)*p)) p++;
    if (!*p) { *pp = p; return 0; }
    int len = 0;
    while (*p && isalnum((unsigned char)*p)) {
        if (len < cap - 1) buf[len++] = (char)tolower((unsigned char)*p);
        p++;
    }
    buf[len] = '\0';
    *pp = p;
    return len;
}

/* Opaque pipeline handle — PHP never sees struct internals. */
struct TransformPipeline {
    const Vocab  *vocab;
    const float  *idf;       /* borrowed pointer into a Tensor */
    const float  *stds;      /* borrowed pointer into a Tensor */
    int           text_col;
    int           label_col; /* -1 = no labels */
    int           n_classes;
};

/* ── df_fit_transformers ─────────────────────────────────────────────────────
 * 2-pass OpenMP parallel fit.
 *   Pass 1: document-frequency counts → IDF  (smooth: log((N+1)/(df+1))+1)
 *   Pass 2: TF-IDF variance accumulation → ZScale stds (center=false)
 *
 * Returns heap-allocated Tensor*[2] = { idf_tensor, stds_tensor }.
 * Caller must tensor_free() each element then free() the array itself.
 * Returns NULL on allocation failure or bad arguments.
 * -------------------------------------------------------------------------- */
Tensor** df_fit_transformers(const DataFrame *df, size_t train_rows,
                               int text_col, const Vocab *vocab)
{
    if (!df || !vocab) return NULL;
    if (text_col < 0 || (size_t)text_col >= df->n_cols) return NULL;
    if (train_rows > df->n_rows) train_rows = df->n_rows;
    if (train_rows == 0) return NULL;

    const DFColumn *col = &df->columns[text_col];
    if (col->dtype != DF_DTYPE_STRING) return NULL;
    const int32_t *cat_idx = DF_COL_I32(col);

    int    V = vocab->size;
    size_t N = train_rows;

    /* Global accumulators */
    int32_t *global_df  = calloc((size_t)V, sizeof(int32_t));
    float   *global_sx  = calloc((size_t)V, sizeof(float));
    float   *global_sx2 = calloc((size_t)V, sizeof(float));
    if (!global_df || !global_sx || !global_sx2) goto oom_pass0;

    /* ── Pass 1: document-frequency counts ─────────────────────────────── */
#pragma omp parallel
    {
        int32_t *my_df   = calloc((size_t)V, sizeof(int32_t));
        int32_t *scratch = calloc((size_t)V, sizeof(int32_t));
        int     *touched = malloc((size_t)V * sizeof(int));
        char     tok[512];

        if (my_df && scratch && touched) {
#pragma omp for schedule(dynamic, 128)
            for (size_t r = 0; r < N; r++) {
                int32_t ci = cat_idx[r];
                if (ci < 0) continue;
                const char *text = col->categories[ci];
                if (!text) continue;

                int n_t = 0;
                const char *p = text;
                int len;
                while ((len = _token_next_nb(&p, tok, (int)sizeof(tok))) > 0) {
                    HashEntry *e = ht_get(vocab->map, tok, (size_t)len);
                    if (!e) continue;
                    int vi = e->value;
                    if (!scratch[vi]) { scratch[vi] = 1; touched[n_t++] = vi; }
                }
                for (int i = 0; i < n_t; i++) {
                    my_df[touched[i]]++;
                    scratch[touched[i]] = 0;
                }
            }
#pragma omp critical
            { for (int i = 0; i < V; i++) global_df[i] += my_df[i]; }
        }
        free(my_df); free(scratch); free(touched);
    }

    /* Compute IDF: log((N+1)/(df_count+1)) + 1 */
    int shape1[1] = { V };
    Tensor *idf_t = tensor_create_dtype(1, shape1, DTYPE_FLOAT32);
    if (!idf_t) goto oom_idf;
    {
        float *idf = (float*)idf_t->data;
        float  Nf  = (float)(N + 1);
        for (int i = 0; i < V; i++)
            idf[i] = logf(Nf / (float)(global_df[i] + 1)) + 1.0f;
    }
    free(global_df); global_df = NULL;

    /* ── Pass 2: accumulate sum_x, sum_x2 on TF-IDF-transformed rows ───── */
    {
        const float *idf = (const float*)idf_t->data;

#pragma omp parallel
        {
            float *my_sx  = calloc((size_t)V, sizeof(float));
            float *my_sx2 = calloc((size_t)V, sizeof(float));
            float *feat   = calloc((size_t)V, sizeof(float));
            int   *touched = malloc((size_t)V * sizeof(int));
            char   tok[512];

            if (my_sx && my_sx2 && feat && touched) {
#pragma omp for schedule(dynamic, 128)
                for (size_t r = 0; r < N; r++) {
                    int32_t ci = cat_idx[r];
                    if (ci < 0) continue;
                    const char *text = col->categories[ci];
                    if (!text) continue;

                    int n_t = 0;
                    const char *p = text;
                    int len;
                    while ((len = _token_next_nb(&p, tok, (int)sizeof(tok))) > 0) {
                        HashEntry *e = ht_get(vocab->map, tok, (size_t)len);
                        if (!e) continue;
                        int vi = e->value;
                        if (!feat[vi]) touched[n_t++] = vi;
                        feat[vi] += 1.0f;
                    }
                    for (int i = 0; i < n_t; i++) {
                        int vi = touched[i];
                        float v = feat[vi] * idf[vi];
                        my_sx[vi]  += v;
                        my_sx2[vi] += v * v;
                        feat[vi] = 0.0f;
                    }
                }
#pragma omp critical
                {
                    for (int i = 0; i < V; i++) {
                        global_sx[i]  += my_sx[i];
                        global_sx2[i] += my_sx2[i];
                    }
                }
            }
            free(my_sx); free(my_sx2); free(feat); free(touched);
        }
    }

    /* Compute ZScale stds: sqrt(max(0, E[X²] - E[X]²)), clipped to [1e-8, ∞) */
    Tensor *stds_t = tensor_create_dtype(1, shape1, DTYPE_FLOAT32);
    if (!stds_t) { tensor_free(idf_t); goto oom_stds; }
    {
        float *stds = (float*)stds_t->data;
        float inv_N = 1.0f / (float)N;
        for (int i = 0; i < V; i++) {
            float mean_i = global_sx[i] * inv_N;
            float var_i  = fmaxf(0.0f, global_sx2[i] * inv_N - mean_i * mean_i);
            stds[i] = fmaxf(1e-8f, sqrtf(var_i));
        }
    }
    free(global_sx); free(global_sx2);

    /* Return heap-allocated [idf, stds] array */
    Tensor **out = malloc(2 * sizeof(Tensor*));
    if (!out) { tensor_free(idf_t); tensor_free(stds_t); return NULL; }
    out[0] = idf_t;
    out[1] = stds_t;
    return out;

oom_pass0:
    free(global_df); free(global_sx); free(global_sx2); return NULL;
oom_idf:
    free(global_df); free(global_sx); free(global_sx2); return NULL;
oom_stds:
    free(global_sx); free(global_sx2); return NULL;
}

/* ── pipeline_create ─────────────────────────────────────────────────────── */

TransformPipeline *pipeline_create(const Vocab *vocab,
                                    const Tensor *idf,  const Tensor *stds,
                                    int text_col, int label_col, int n_classes)
{
    if (!vocab || !idf || !stds) return NULL;
    TransformPipeline *pl = malloc(sizeof(TransformPipeline));
    if (!pl) return NULL;
    pl->vocab     = vocab;
    pl->idf       = (const float*)idf->data;
    pl->stds      = (const float*)stds->data;
    pl->text_col  = text_col;
    pl->label_col = label_col;
    pl->n_classes = n_classes;
    return pl;
}

/* ── pipeline_free ───────────────────────────────────────────────────────── */

void pipeline_free(TransformPipeline *pl) { free(pl); }

/* ── pipeline_transform_batch ───────────────────────────────────────────────
 * Process rows [offset, offset+n) of df through the full chain:
 *   tokenize → BoW → TfIdf → ZScale → one-hot labels
 *
 * Returns heap-allocated Tensor*[2]:
 *   [0] features  [n × vocab_size]  FLOAT32
 *   [1] labels    [n × n_classes]   FLOAT32  (all-zero if label_col < 0)
 * Caller must tensor_free() each element then free() the array.
 * -------------------------------------------------------------------------- */
Tensor **pipeline_transform_batch(const DataFrame *df, size_t offset, size_t n,
                                    const TransformPipeline *pl)
{
    if (!df || !pl) return NULL;
    if (offset >= df->n_rows) return NULL;
    if (offset + n > df->n_rows) n = df->n_rows - offset;
    if (n == 0) return NULL;

    int V  = pl->vocab->size;
    int NC = pl->n_classes;

    if (pl->text_col < 0 || (size_t)pl->text_col >= df->n_cols) return NULL;
    const DFColumn *txt_col = &df->columns[pl->text_col];
    if (txt_col->dtype != DF_DTYPE_STRING) return NULL;
    const int32_t  *txt_cat = DF_COL_I32(txt_col);

    /* Allocate output tensors */
    int feat_shape[2] = { (int)n, V  };
    int lbl_shape[2]  = { (int)n, NC };
    Tensor *feat_t = tensor_create_dtype(2, feat_shape, DTYPE_FLOAT32);
    Tensor *lbl_t  = tensor_create_dtype(2, lbl_shape,  DTYPE_FLOAT32);
    if (!feat_t || !lbl_t) { tensor_free(feat_t); tensor_free(lbl_t); return NULL; }

    float *feat_data = (float*)feat_t->data;
    float *lbl_data  = (float*)lbl_t->data;
    /* tensor_create_dtype zero-initialises, but be explicit for labels */
    memset(lbl_data, 0, (size_t)n * NC * sizeof(float));

    /* ── Fill one-hot labels ────────────────────────────────────────────── */
    if (pl->label_col >= 0 && (size_t)pl->label_col < df->n_cols) {
        const DFColumn *lbl_col = &df->columns[pl->label_col];
        if (lbl_col->dtype == DF_DTYPE_FLOAT32) {
            const float *lv = DF_COL_F32(lbl_col);
            for (size_t r = 0; r < n; r++) {
                int cls = (int)lv[offset + r];
                if (cls >= 0 && cls < NC) lbl_data[r * NC + cls] = 1.0f;
            }
        } else if (lbl_col->dtype == DF_DTYPE_INT32) {
            const int32_t *lv = DF_COL_I32(lbl_col);
            for (size_t r = 0; r < n; r++) {
                int cls = (int)lv[offset + r];
                if (cls >= 0 && cls < NC) lbl_data[r * NC + cls] = 1.0f;
            }
        } else if (lbl_col->dtype == DF_DTYPE_STRING) {
            const int32_t *ci = DF_COL_I32(lbl_col);
            for (size_t r = 0; r < n; r++) {
                int32_t cat = ci[offset + r];
                if (cat >= 0 && cat < lbl_col->n_categories) {
                    int cls = atoi(lbl_col->categories[cat]);
                    if (cls >= 0 && cls < NC) lbl_data[r * NC + cls] = 1.0f;
                }
            }
        }
    }

    /* ── Feature transform: BoW → fused TfIdf+ZScale (OpenMP parallel) ── */
    const float *idf  = pl->idf;
    const float *stds = pl->stds;

#pragma omp parallel
    {
        float *row_scratch = calloc((size_t)V, sizeof(float));
        int   *touched     = malloc((size_t)V * sizeof(int));
        char   tok[512];

        if (row_scratch && touched) {
#pragma omp for schedule(static)
            for (size_t r = 0; r < n; r++) {
                int32_t ci = txt_cat[offset + r];
                if (ci < 0) continue;
                const char *text = txt_col->categories[ci];
                if (!text) continue;

                int n_t = 0;
                const char *p = text;
                int len;
                while ((len = _token_next_nb(&p, tok, (int)sizeof(tok))) > 0) {
                    HashEntry *e = ht_get(pl->vocab->map, tok, (size_t)len);
                    if (!e) continue;
                    int vi = e->value;
                    if (!row_scratch[vi]) touched[n_t++] = vi;
                    row_scratch[vi] += 1.0f;
                }

                /* Fused TfIdf + ZScale write into output row; clear scratch */
                float *dst = feat_data + r * V;
                for (int i = 0; i < n_t; i++) {
                    int vi = touched[i];
                    dst[vi] = row_scratch[vi] * idf[vi] / stds[vi];
                    row_scratch[vi] = 0.0f;
                }
            }
        }
        free(row_scratch); free(touched);
    }

    Tensor **out = malloc(2 * sizeof(Tensor*));
    if (!out) { tensor_free(feat_t); tensor_free(lbl_t); return NULL; }
    out[0] = feat_t;
    out[1] = lbl_t;
    return out;
}

/* ============================================================================
 * 8. Introspection Accessors
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

/* ============================================================================
 * Sections 10–15: Extended DataFrame Operations
 *
 * filter/where, sort, groupby, join, schema mutations, describe, sampling.
 * All heavy work runs in C; PHP is orchestration only.
 * ========================================================================== */

#ifdef _OPENMP
#  include <omp.h>
#endif
#include <time.h>

/* ── Internal helpers ───────────────────────────────────────────────────── */

/* Copy only category metadata (names + hash map) from src to dst col. */
static bool _col_copy_cats(DFColumn *dst, const DFColumn *src) {
    if (src->dtype != DF_DTYPE_STRING) return true;
    dst->_cat_cap = dst->n_categories = src->n_categories;
    if (src->n_categories == 0) {
        dst->categories = NULL;
        dst->cat_map    = ht_create();
        return dst->cat_map != NULL;
    }
    dst->categories = malloc((size_t)src->n_categories * sizeof(char *));
    if (!dst->categories) return false;
    dst->cat_map = ht_create();
    if (!dst->cat_map) { free(dst->categories); dst->categories = NULL; return false; }
    for (int32_t i = 0; i < src->n_categories; i++) {
        dst->categories[i] = src->categories[i] ? strdup(src->categories[i]) : NULL;
        if (dst->categories[i])
            ht_put(dst->cat_map, dst->categories[i], strlen(dst->categories[i]), (int)i);
    }
    return true;
}

/*
 * Allocate dst->data and scatter rows at indices[] from src.
 * dst must already have dtype, name, and (STRING) categories set.
 */
static bool _col_scatter(DFColumn *dst, const DFColumn *src,
                          const size_t *indices, size_t n) {
    size_t esz = (src->dtype == DF_DTYPE_FLOAT32) ? sizeof(float) : sizeof(int32_t);
    dst->data = malloc(n ? n * esz : 1); /* malloc(0) is implementation-defined */
    if (!dst->data) return false;
    const char *sp = (const char *)src->data;
    char       *dp = (char *)dst->data;
    for (size_t i = 0; i < n; i++)
        memcpy(dp + i * esz, sp + indices[i] * esz, esz);
    return true;
}

/*
 * Build an output DataFrame shell: schema copied from src, data buffers empty.
 */
static DataFrame *_df_shell(size_t out_n, size_t n_cols,
                              const DFColumn *src_cols) {
    DataFrame *out = df_create(out_n, n_cols);
    if (!out) return NULL;
    for (size_t c = 0; c < n_cols; c++) {
        DFColumn *dc = &out->columns[c];
        const DFColumn *sc = &src_cols[c];
        memcpy(dc->name, sc->name, DF_MAX_COL_NAME);
        dc->dtype = sc->dtype;
        if (!_col_copy_cats(dc, sc)) { df_free(out); return NULL; }
    }
    return out;
}

/* ── Section 10: Vectorized Filtering ─────────────────────────────────── */

DataFrame *df_apply_mask(const DataFrame *df, const int32_t *mask) {
    if (!df || !mask) DF_ERR("df_apply_mask: NULL argument");
    size_t n = df->n_rows;

    /* Count matching rows */
    size_t out_n = 0;
    for (size_t i = 0; i < n; i++) out_n += (mask[i] != 0);

    /* Build dense index array */
    size_t *idx = (size_t *)malloc(out_n ? out_n * sizeof(size_t) : 1);
    if (!idx) DF_ERR("df_apply_mask: OOM index array");
    size_t j = 0;
    for (size_t i = 0; i < n; i++) if (mask[i]) idx[j++] = i;

    /* Scatter into new DataFrame */
    DataFrame *out = _df_shell(out_n, df->n_cols, df->columns);
    if (!out) { free(idx); DF_ERR("df_apply_mask: OOM DataFrame shell"); }
    for (size_t c = 0; c < df->n_cols; c++) {
        if (!_col_scatter(&out->columns[c], &df->columns[c], idx, out_n)) {
            free(idx); df_free(out); DF_ERR("df_apply_mask: OOM column data");
        }
    }
    free(idx);
    return out;
}

DataFrame *df_where_f32(const DataFrame *df, int col_idx, int cmp_op, float val) {
    if (!df || col_idx < 0 || (size_t)col_idx >= df->n_cols)
        DF_ERR("df_where_f32: invalid argument");
    const DFColumn *col = &df->columns[col_idx];
    if (col->dtype == DF_DTYPE_STRING)
        DF_ERR("df_where_f32: column is STRING; use df_where_str");

    size_t n = df->n_rows;
    int32_t *mask = (int32_t *)malloc(n ? n * sizeof(int32_t) : 1);
    if (!mask) DF_ERR("df_where_f32: OOM mask");

    if (col->dtype == DF_DTYPE_FLOAT32) {
        const float *d = (const float *)col->data;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            switch ((DFCmpOp)cmp_op) {
                case DF_CMP_EQ:  mask[i]=(d[i]==val); break;
                case DF_CMP_NEQ: mask[i]=(d[i]!=val); break;
                case DF_CMP_GT:  mask[i]=(d[i]> val); break;
                case DF_CMP_GTE: mask[i]=(d[i]>=val); break;
                case DF_CMP_LT:  mask[i]=(d[i]< val); break;
                default:         mask[i]=(d[i]<=val); break;
            }
        }
    } else {
        const int32_t *d = (const int32_t *)col->data;
        int32_t iv = (int32_t)val;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            switch ((DFCmpOp)cmp_op) {
                case DF_CMP_EQ:  mask[i]=(d[i]==iv); break;
                case DF_CMP_NEQ: mask[i]=(d[i]!=iv); break;
                case DF_CMP_GT:  mask[i]=(d[i]> iv); break;
                case DF_CMP_GTE: mask[i]=(d[i]>=iv); break;
                case DF_CMP_LT:  mask[i]=(d[i]< iv); break;
                default:         mask[i]=(d[i]<=iv); break;
            }
        }
    }
    DataFrame *out = df_apply_mask(df, mask);
    free(mask);
    return out;
}

DataFrame *df_where_str(const DataFrame *df, int col_idx, const char *val) {
    if (!df || col_idx < 0 || (size_t)col_idx >= df->n_cols || !val)
        DF_ERR("df_where_str: invalid argument");
    const DFColumn *col = &df->columns[col_idx];
    if (col->dtype != DF_DTYPE_STRING)
        DF_ERR("df_where_str: column is not STRING");

    /* Intern val to category index — O(1) */
    HashEntry *e = col->cat_map ? ht_get(col->cat_map, val, strlen(val)) : NULL;
    if (!e) {
        /* value absent → empty result DataFrame */
        DataFrame *out = _df_shell(0, df->n_cols, df->columns);
        if (!out) DF_ERR("df_where_str: OOM empty df");
        for (size_t c = 0; c < df->n_cols; c++) {
            out->columns[c].data = malloc(1);
            if (!out->columns[c].data) { df_free(out); DF_ERR("df_where_str: OOM col data"); }
        }
        return out;
    }
    int32_t cat_idx = (int32_t)e->value;

    size_t n = df->n_rows;
    int32_t *mask = (int32_t *)malloc(n ? n * sizeof(int32_t) : 1);
    if (!mask) DF_ERR("df_where_str: OOM mask");

    const int32_t *d = (const int32_t *)col->data;
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) mask[i] = (d[i] == cat_idx);

    DataFrame *out = df_apply_mask(df, mask);
    free(mask);
    return out;
}

/* ── Section 11: Sorting ─────────────────────────────────────────────── */

typedef struct { size_t idx; float key_f; int32_t key_i; } _SortEnt;

static int _sort_f32_asc (const void *a, const void *b) {
    float ka = ((_SortEnt*)a)->key_f, kb = ((_SortEnt*)b)->key_f;
    return (ka > kb) - (ka < kb);
}
static int _sort_f32_desc(const void *a, const void *b) {
    return _sort_f32_asc(b, a);
}
static int _sort_i32_asc (const void *a, const void *b) {
    int32_t ka = ((_SortEnt*)a)->key_i, kb = ((_SortEnt*)b)->key_i;
    return (ka > kb) - (ka < kb);
}
static int _sort_i32_desc(const void *a, const void *b) {
    return _sort_i32_asc(b, a);
}

DataFrame *df_sort_by_col(const DataFrame *df, int col_idx, bool ascending) {
    if (!df || col_idx < 0 || (size_t)col_idx >= df->n_cols)
        DF_ERR("df_sort_by_col: invalid argument");

    const DFColumn *col = &df->columns[col_idx];
    size_t n = df->n_rows;

    _SortEnt *ents = (_SortEnt *)malloc(n ? n * sizeof(_SortEnt) : 1);
    if (!ents) DF_ERR("df_sort_by_col: OOM sort entries");

    if (col->dtype == DF_DTYPE_FLOAT32) {
        const float *d = (const float *)col->data;
        for (size_t i = 0; i < n; i++) { ents[i].idx = i; ents[i].key_f = d[i]; }
        qsort(ents, n, sizeof(_SortEnt), ascending ? _sort_f32_asc : _sort_f32_desc);
    } else { /* INT32 or STRING (sort by category index) */
        const int32_t *d = (const int32_t *)col->data;
        for (size_t i = 0; i < n; i++) { ents[i].idx = i; ents[i].key_i = d[i]; }
        qsort(ents, n, sizeof(_SortEnt), ascending ? _sort_i32_asc : _sort_i32_desc);
    }

    /* Build permutation index array */
    size_t *perm = (size_t *)malloc(n ? n * sizeof(size_t) : 1);
    if (!perm) { free(ents); DF_ERR("df_sort_by_col: OOM perm array"); }
    for (size_t i = 0; i < n; i++) perm[i] = ents[i].idx;
    free(ents);

    DataFrame *out = _df_shell(n, df->n_cols, df->columns);
    if (!out) { free(perm); DF_ERR("df_sort_by_col: OOM DataFrame shell"); }
    for (size_t c = 0; c < df->n_cols; c++) {
        if (!_col_scatter(&out->columns[c], &df->columns[c], perm, n)) {
            free(perm); df_free(out); DF_ERR("df_sort_by_col: OOM column data");
        }
    }
    free(perm);
    return out;
}

/* ── Section 12: GroupBy Aggregation ─────────────────────────────────── */

/*
 * Per-group accumulator tracks all stats needed for any agg type in one pass.
 * Two passes required for STD (mean needed before variance accumulation).
 */
typedef struct {
    double  sum;
    double  sum2;  /* sum of squares — for std */
    float   min;
    float   max;
    int32_t count;
} _GBAcc;

DataFrame *df_groupby_multi_agg(const DataFrame *df,
                                 int group_col_idx,
                                 const int *agg_col_idxs,
                                 const int *agg_types,
                                 int n_agg) {
    if (!df || group_col_idx < 0 || (size_t)group_col_idx >= df->n_cols
            || !agg_col_idxs || !agg_types || n_agg <= 0)
        DF_ERR("df_groupby_multi_agg: invalid argument");

    const DFColumn *gcol = &df->columns[group_col_idx];
    if (gcol->dtype != DF_DTYPE_STRING)
        DF_ERR("df_groupby_multi_agg: group column must be STRING (categorical)");

    int n_groups = gcol->n_categories;
    size_t n_rows = df->n_rows;

    /* Validate agg columns */
    for (int ac = 0; ac < n_agg; ac++) {
        int ci = agg_col_idxs[ac];
        if (ci < 0 || (size_t)ci >= df->n_cols
                || df->columns[ci].dtype == DF_DTYPE_STRING)
            DF_ERR("df_groupby_multi_agg: agg column must be FLOAT32 or INT32");
    }

    /* ── Pass 1: parallel accumulation per group ───────────────────────── */
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    /* thread_acc[thread][group * n_agg + ac_col] */
    size_t acc_stride = (size_t)n_groups * (size_t)n_agg;
    _GBAcc *thread_acc = (_GBAcc *)malloc((size_t)n_threads * acc_stride * sizeof(_GBAcc));
    if (!thread_acc) DF_ERR("df_groupby_multi_agg: OOM accumulators");

    /* Initialise */
    for (size_t k = 0; k < (size_t)n_threads * acc_stride; k++) {
        thread_acc[k].sum   = 0.0;
        thread_acc[k].sum2  = 0.0;
        thread_acc[k].min   =  1e38f;
        thread_acc[k].max   = -1e38f;
        thread_acc[k].count = 0;
    }

    /* Gather pointers to agg column data */
    const float **agg_data = (const float **)malloc((size_t)n_agg * sizeof(float *));
    if (!agg_data) { free(thread_acc); DF_ERR("df_groupby_multi_agg: OOM agg_data ptrs"); }
    for (int ac = 0; ac < n_agg; ac++)
        agg_data[ac] = (const float *)df->columns[agg_col_idxs[ac]].data;

    const int32_t *gdata = (const int32_t *)gcol->data;

#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        _GBAcc *my_acc = thread_acc + (size_t)tid * acc_stride;
#pragma omp for schedule(static)
        for (size_t r = 0; r < n_rows; r++) {
            int32_t g = gdata[r];
            if (g < 0 || g >= n_groups) continue;
            for (int ac = 0; ac < n_agg; ac++) {
                float v = agg_data[ac][r];
                if (v != v) continue; /* NaN sentinel skip */
                _GBAcc *a = &my_acc[g * n_agg + ac];
                a->sum  += v;
                a->sum2 += (double)v * v;
                if (v < a->min) a->min = v;
                if (v > a->max) a->max = v;
                a->count++;
            }
        }
    }
    free(agg_data);

    /* ── Reduce across threads ─────────────────────────────────────────── */
    _GBAcc *global_acc = (_GBAcc *)calloc(acc_stride, sizeof(_GBAcc));
    if (!global_acc) { free(thread_acc); DF_ERR("df_groupby_multi_agg: OOM global acc"); }
    for (size_t k = 0; k < acc_stride; k++) {
        global_acc[k].min = 1e38f; global_acc[k].max = -1e38f;
    }
    for (int t = 0; t < n_threads; t++) {
        _GBAcc *ta = thread_acc + (size_t)t * acc_stride;
        for (size_t k = 0; k < acc_stride; k++) {
            global_acc[k].sum   += ta[k].sum;
            global_acc[k].sum2  += ta[k].sum2;
            global_acc[k].count += ta[k].count;
            if (ta[k].min < global_acc[k].min) global_acc[k].min = ta[k].min;
            if (ta[k].max > global_acc[k].max) global_acc[k].max = ta[k].max;
        }
    }
    free(thread_acc);

    /* ── Build output DataFrame: [group_col | agg_col_0 | ... ] ─────────── */
    size_t out_cols = 1 + (size_t)n_agg;
    DataFrame *out = df_create((size_t)n_groups, out_cols);
    if (!out) { free(global_acc); DF_ERR("df_groupby_multi_agg: OOM output df"); }

    /* Group column (STRING) */
    DFColumn *ogc = &out->columns[0];
    memcpy(ogc->name, gcol->name, DF_MAX_COL_NAME);
    ogc->dtype = DF_DTYPE_STRING;
    if (!_col_copy_cats(ogc, gcol)) {
        free(global_acc); df_free(out); DF_ERR("df_groupby_multi_agg: OOM group col cats");
    }
    ogc->data = malloc((size_t)n_groups * sizeof(int32_t));
    if (!ogc->data) { free(global_acc); df_free(out); DF_ERR("df_groupby_multi_agg: OOM group col data"); }
    for (int g = 0; g < n_groups; g++) ((int32_t *)ogc->data)[g] = (int32_t)g;

    /* Aggregated columns */
    for (int ac = 0; ac < n_agg; ac++) {
        DFColumn *oc = &out->columns[1 + ac];
        const DFColumn *sc = &df->columns[agg_col_idxs[ac]];
        memcpy(oc->name, sc->name, DF_MAX_COL_NAME);
        oc->dtype = DF_DTYPE_FLOAT32;
        oc->data = malloc((size_t)n_groups * sizeof(float));
        if (!oc->data) {
            free(global_acc); df_free(out); DF_ERR("df_groupby_multi_agg: OOM agg col data");
        }
        float *od = (float *)oc->data;
        int agg_t = agg_types[ac];
        for (int g = 0; g < n_groups; g++) {
            _GBAcc *a = &global_acc[g * n_agg + ac];
            switch ((DFAggType)agg_t) {
                case DF_AGG_SUM:   od[g] = (float)a->sum;   break;
                case DF_AGG_MEAN:  od[g] = a->count ? (float)(a->sum / a->count) : 0.0f; break;
                case DF_AGG_MIN:   od[g] = a->count ? a->min : 0.0f; break;
                case DF_AGG_MAX:   od[g] = a->count ? a->max : 0.0f; break;
                case DF_AGG_COUNT: od[g] = (float)a->count; break;
                case DF_AGG_STD: {
                    if (a->count < 2) { od[g] = 0.0f; break; }
                    double mean = a->sum / a->count;
                    double var  = a->sum2 / a->count - mean * mean;
                    od[g] = (float)sqrt(var < 0.0 ? 0.0 : var);
                    break;
                }
                default: od[g] = 0.0f; break;
            }
        }
    }
    free(global_acc);
    return out;
}

DataFrame *df_groupby_agg(const DataFrame *df,
                           int group_col_idx,
                           const int *agg_col_idxs, int n_agg,
                           int agg_type) {
    /* Broadcast single agg_type to all columns */
    int *types = (int *)malloc((size_t)n_agg * sizeof(int));
    if (!types) DF_ERR("df_groupby_agg: OOM");
    for (int i = 0; i < n_agg; i++) types[i] = agg_type;
    DataFrame *out = df_groupby_multi_agg(df, group_col_idx,
                                           agg_col_idxs, types, n_agg);
    free(types);
    return out;
}

/* ── Section 13: Join / Merge ─────────────────────────────────────────── */

/*
 * Sort-merge equijoin on a single column.
 * Supports INT32 and FLOAT32 key columns (treated as int32 for equality).
 * For STRING keys: uses category index (so same string in both dfs must map
 * to same category value — valid when using df_where_str-style pipelines).
 */

typedef struct { size_t orig; float key; } _JoinEnt;

static int _je_asc(const void *a, const void *b) {
    float ka = ((_JoinEnt*)a)->key, kb = ((_JoinEnt*)b)->key;
    return (ka > kb) - (ka < kb);
}

/*
 * Build sorted key+index array for a column.
 */
static _JoinEnt *_build_join_ents(const DFColumn *col, size_t n) {
    _JoinEnt *ents = (_JoinEnt *)malloc(n ? n * sizeof(_JoinEnt) : 1);
    if (!ents) return NULL;
    if (col->dtype == DF_DTYPE_FLOAT32) {
        const float *d = (const float *)col->data;
        for (size_t i = 0; i < n; i++) { ents[i].orig = i; ents[i].key = d[i]; }
    } else {
        const int32_t *d = (const int32_t *)col->data;
        for (size_t i = 0; i < n; i++) { ents[i].orig = i; ents[i].key = (float)d[i]; }
    }
    qsort(ents, n, sizeof(_JoinEnt), _je_asc);
    return ents;
}

/*
 * Scatter one element from column to output at position out_row.
 * Output column data must already be allocated.
 */
static void _col_put_elem(DFColumn *dst, const DFColumn *src,
                          size_t src_row, size_t dst_row) {
    size_t esz = (src->dtype == DF_DTYPE_FLOAT32) ? sizeof(float) : sizeof(int32_t);
    memcpy((char *)dst->data + dst_row * esz,
           (const char *)src->data + src_row * esz, esz);
}

static void _col_put_null(DFColumn *dst, size_t dst_row) {
    if (dst->dtype == DF_DTYPE_FLOAT32) {
        float nan = 0.0f / 0.0f; /* NaN sentinel */
        memcpy((char *)dst->data + dst_row * sizeof(float), &nan, sizeof(float));
    } else {
        int32_t miss = INT32_MIN;
        memcpy((char *)dst->data + dst_row * sizeof(int32_t), &miss, sizeof(int32_t));
    }
}

DataFrame *df_join(const DataFrame *left,
                   const DataFrame *right,
                   int left_col_idx,
                   int right_col_idx,
                   int join_type) {
    if (!left || !right
            || left_col_idx  < 0 || (size_t)left_col_idx  >= left->n_cols
            || right_col_idx < 0 || (size_t)right_col_idx >= right->n_cols)
        DF_ERR("df_join: invalid argument");

    size_t ln = left->n_rows, rn = right->n_rows;

    _JoinEnt *le = _build_join_ents(&left->columns[left_col_idx],   ln);
    _JoinEnt *re = _build_join_ents(&right->columns[right_col_idx], rn);
    if (!le || !re) { free(le); free(re); DF_ERR("df_join: OOM sort entries"); }

    /* Phase 1 — count output rows */
    size_t n_right_cols = right->n_cols - 1; /* exclude right join key */
    size_t out_cols     = left->n_cols + n_right_cols;
    size_t out_n        = 0;

    size_t li = 0, ri = 0;
    while (li < ln && ri < rn) {
        float lk = le[li].key, rk = re[ri].key;
        if (lk == rk) {
            /* Count right group size */
            size_t rg = ri;
            while (rg < rn && re[rg].key == lk) rg++;
            size_t rg_sz = rg - ri;
            /* Count left group size */
            size_t lg = li;
            while (lg < ln && le[lg].key == lk) lg++;
            out_n += (lg - li) * rg_sz;
            li = lg; ri = rg;
        } else if (lk < rk) {
            if ((DFJoinType)join_type == DF_JOIN_LEFT) out_n++;
            li++;
        } else {
            ri++;
        }
    }
    if ((DFJoinType)join_type == DF_JOIN_LEFT) out_n += (ln - li); /* remaining unmatched left */

    /* Phase 2 — build output schema */
    /* Right columns: all except right_col_idx */
    int *right_col_map = (int *)malloc(n_right_cols * sizeof(int));
    if (!right_col_map) { free(le); free(re); DF_ERR("df_join: OOM col map"); }
    size_t rm = 0;
    for (size_t c = 0; c < right->n_cols; c++)
        if ((int)c != right_col_idx) right_col_map[rm++] = (int)c;

    /* Build combined column list for schema */
    DFColumn *all_cols = (DFColumn *)calloc(out_cols, sizeof(DFColumn));
    if (!all_cols) { free(le); free(re); free(right_col_map); DF_ERR("df_join: OOM cols"); }
    for (size_t c = 0; c < left->n_cols; c++) {
        memcpy(all_cols[c].name, left->columns[c].name, DF_MAX_COL_NAME);
        all_cols[c].dtype = left->columns[c].dtype;
    }
    for (size_t c = 0; c < n_right_cols; c++) {
        int rc = right_col_map[c];
        memcpy(all_cols[left->n_cols + c].name, right->columns[rc].name, DF_MAX_COL_NAME);
        all_cols[left->n_cols + c].dtype = right->columns[rc].dtype;
    }

    DataFrame *out = _df_shell(out_n, out_cols, all_cols);
    free(all_cols);
    if (!out) { free(le); free(re); free(right_col_map); DF_ERR("df_join: OOM output df"); }

    /* Allocate output column data buffers */
    for (size_t c = 0; c < out_cols; c++) {
        DFColumn *oc = &out->columns[c];
        size_t esz = (oc->dtype == DF_DTYPE_FLOAT32) ? sizeof(float) : sizeof(int32_t);
        oc->data = malloc(out_n ? out_n * esz : 1);
        if (!oc->data) {
            free(le); free(re); free(right_col_map); df_free(out);
            DF_ERR("df_join: OOM column data");
        }
    }

    /* Phase 3 — scatter matching rows */
    size_t out_r = 0;
    li = 0; ri = 0;
    while (li < ln && ri < rn) {
        float lk = le[li].key, rk = re[ri].key;
        if (lk == rk) {
            size_t rg = ri;
            while (rg < rn && re[rg].key == lk) rg++;
            size_t lg = li;
            while (lg < ln && le[lg].key == lk) lg++;

            for (size_t l = li; l < lg; l++) {
                for (size_t r = ri; r < rg; r++) {
                    for (size_t c = 0; c < left->n_cols; c++)
                        _col_put_elem(&out->columns[c], &left->columns[c], le[l].orig, out_r);
                    for (size_t c = 0; c < n_right_cols; c++)
                        _col_put_elem(&out->columns[left->n_cols + c],
                                      &right->columns[right_col_map[c]], re[r].orig, out_r);
                    out_r++;
                }
            }
            li = lg; ri = rg;
        } else if (lk < rk) {
            if ((DFJoinType)join_type == DF_JOIN_LEFT) {
                for (size_t c = 0; c < left->n_cols; c++)
                    _col_put_elem(&out->columns[c], &left->columns[c], le[li].orig, out_r);
                for (size_t c = 0; c < n_right_cols; c++)
                    _col_put_null(&out->columns[left->n_cols + c], out_r);
                out_r++;
            }
            li++;
        } else {
            ri++;
        }
    }
    /* Remaining unmatched left rows (left join only) */
    if ((DFJoinType)join_type == DF_JOIN_LEFT) {
        while (li < ln) {
            for (size_t c = 0; c < left->n_cols; c++)
                _col_put_elem(&out->columns[c], &left->columns[c], le[li].orig, out_r);
            for (size_t c = 0; c < n_right_cols; c++)
                _col_put_null(&out->columns[left->n_cols + c], out_r);
            out_r++; li++;
        }
    }

    free(le); free(re); free(right_col_map);
    return out;
}

/* ── Section 14: Schema Mutations ─────────────────────────────────────── */

DataFrame *df_add_f32_column(const DataFrame *df, const char *name,
                              const float *data, size_t n_rows) {
    if (!df || !name || !data || n_rows != df->n_rows)
        DF_ERR("df_add_f32_column: invalid argument");

    size_t nc = df->n_cols + 1;
    DataFrame *out = df_create(df->n_rows, nc);
    if (!out) DF_ERR("df_add_f32_column: OOM");

    for (size_t c = 0; c < df->n_cols; c++) {
        if (!_col_copy(&out->columns[c], &df->columns[c], df->n_rows)) {
            df_free(out); DF_ERR("df_add_f32_column: OOM col copy");
        }
    }

    DFColumn *nc_col = &out->columns[df->n_cols];
    strncpy(nc_col->name, name, DF_MAX_COL_NAME - 1);
    nc_col->dtype = DF_DTYPE_FLOAT32;
    nc_col->data  = malloc(n_rows * sizeof(float));
    if (!nc_col->data) { df_free(out); DF_ERR("df_add_f32_column: OOM new col data"); }
    memcpy(nc_col->data, data, n_rows * sizeof(float));
    return out;
}

DataFrame *df_drop_column_new(const DataFrame *df, int col_idx) {
    if (!df || col_idx < 0 || (size_t)col_idx >= df->n_cols)
        DF_ERR("df_drop_column_new: invalid argument");

    size_t nc = df->n_cols - 1;
    DataFrame *out = df_create(df->n_rows, nc);
    if (!out) DF_ERR("df_drop_column_new: OOM");

    size_t oc = 0;
    for (size_t c = 0; c < df->n_cols; c++) {
        if ((int)c == col_idx) continue;
        if (!_col_copy(&out->columns[oc++], &df->columns[c], df->n_rows)) {
            df_free(out); DF_ERR("df_drop_column_new: OOM col copy");
        }
    }
    return out;
}

void df_rename_column(DataFrame *df, int col_idx, const char *new_name) {
    if (!df || col_idx < 0 || (size_t)col_idx >= df->n_cols || !new_name) return;
    strncpy(df->columns[col_idx].name, new_name, DF_MAX_COL_NAME - 1);
    df->columns[col_idx].name[DF_MAX_COL_NAME - 1] = '\0';
}

DataFrame *df_cast_to_f32(const DataFrame *df, int col_idx) {
    if (!df || col_idx < 0 || (size_t)col_idx >= df->n_cols)
        DF_ERR("df_cast_to_f32: invalid argument");
    if (df->columns[col_idx].dtype == DF_DTYPE_FLOAT32)
        DF_ERR("df_cast_to_f32: column is already FLOAT32");

    DataFrame *out = df_create(df->n_rows, df->n_cols);
    if (!out) DF_ERR("df_cast_to_f32: OOM");

    for (size_t c = 0; c < df->n_cols; c++) {
        if ((int)c != col_idx) {
            if (!_col_copy(&out->columns[c], &df->columns[c], df->n_rows)) {
                df_free(out); DF_ERR("df_cast_to_f32: OOM col copy");
            }
            continue;
        }
        DFColumn *oc = &out->columns[c];
        const DFColumn *sc = &df->columns[c];
        memcpy(oc->name, sc->name, DF_MAX_COL_NAME);
        oc->dtype = DF_DTYPE_FLOAT32;
        oc->data  = malloc(df->n_rows * sizeof(float));
        if (!oc->data) { df_free(out); DF_ERR("df_cast_to_f32: OOM cast col"); }
        const int32_t *src = (const int32_t *)sc->data;
        float         *dst = (float *)oc->data;
#pragma omp parallel for simd schedule(static)
        for (size_t r = 0; r < df->n_rows; r++)
            dst[r] = (src[r] == INT32_MIN) ? (0.0f/0.0f) : (float)src[r];
    }
    return out;
}

DataFrame *df_fill_null_f32(const DataFrame *df, int col_idx, float fill_val) {
    if (!df || col_idx < 0 || (size_t)col_idx >= df->n_cols)
        DF_ERR("df_fill_null_f32: invalid argument");
    if (df->columns[col_idx].dtype != DF_DTYPE_FLOAT32)
        DF_ERR("df_fill_null_f32: column is not FLOAT32");

    DataFrame *out = df_create(df->n_rows, df->n_cols);
    if (!out) DF_ERR("df_fill_null_f32: OOM");
    for (size_t c = 0; c < df->n_cols; c++) {
        if (!_col_copy(&out->columns[c], &df->columns[c], df->n_rows)) {
            df_free(out); DF_ERR("df_fill_null_f32: OOM col copy");
        }
    }
    float *d = (float *)out->columns[col_idx].data;
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < df->n_rows; r++)
        if (_f32_is_nan(d[r])) d[r] = fill_val;
    return out;
}

DataFrame *df_concat_rows(const DataFrame *a, const DataFrame *b) {
    if (!a || !b || a->n_cols != b->n_cols)
        DF_ERR("df_concat_rows: NULL or mismatched column count");

    size_t total = a->n_rows + b->n_rows;
    DataFrame *out = df_create(total, a->n_cols);
    if (!out) DF_ERR("df_concat_rows: OOM");

    for (size_t c = 0; c < a->n_cols; c++) {
        const DFColumn *ac = &a->columns[c];
        const DFColumn *bc = &b->columns[c];
        if (ac->dtype != bc->dtype) {
            df_free(out); DF_ERR("df_concat_rows: dtype mismatch in column");
        }
        DFColumn *oc = &out->columns[c];
        memcpy(oc->name, ac->name, DF_MAX_COL_NAME);
        oc->dtype = ac->dtype;

        size_t esz = (ac->dtype == DF_DTYPE_FLOAT32) ? sizeof(float) : sizeof(int32_t);
        oc->data   = malloc(total ? total * esz : 1);
        if (!oc->data) { df_free(out); DF_ERR("df_concat_rows: OOM col data"); }
        memcpy((char *)oc->data,                   ac->data, a->n_rows * esz);
        memcpy((char *)oc->data + a->n_rows * esz, bc->data, b->n_rows * esz);

        /* For STRING columns: merge categories (reindex b's category indices) */
        if (ac->dtype == DF_DTYPE_STRING) {
            /* Copy a's categories as-is */
            _col_copy_cats(oc, ac);
            /* Append b's categories, remapping indices in b's data region */
            int32_t *odata = (int32_t *)oc->data;
            for (size_t r = a->n_rows; r < total; r++) {
                int32_t old_idx = odata[r];
                if (old_idx < 0 || old_idx >= bc->n_categories) {
                    odata[r] = -1; continue;
                }
                const char *cat_str = bc->categories[old_idx];
                if (!cat_str) { odata[r] = -1; continue; }
                /* intern into oc's category map */
                HashEntry *e = ht_get(oc->cat_map, cat_str, strlen(cat_str));
                if (e) {
                    odata[r] = (int32_t)e->value;
                } else {
                    odata[r] = _cat_intern(oc, cat_str, strlen(cat_str));
                }
            }
        }
    }
    return out;
}

/* ── Section 15: Describe / Sample / ValueCounts ─────────────────────── */

Tensor *df_describe(const DataFrame *df) {
    if (!df) DF_ERR("df_describe: NULL DataFrame");

    /* Collect FLOAT32 column indices */
    int *f32_cols = (int *)malloc(df->n_cols * sizeof(int));
    if (!f32_cols) DF_ERR("df_describe: OOM");
    int nf = 0;
    for (size_t c = 0; c < df->n_cols; c++)
        if (df->columns[c].dtype == DF_DTYPE_FLOAT32) f32_cols[nf++] = (int)c;

    if (nf == 0) { free(f32_cols); DF_ERR("df_describe: no FLOAT32 columns"); }

    /* Output: [nf × 5] — [count, mean, std, min, max] */
    int shape[2] = {nf, 5};
    Tensor *out = tensor_create(2, shape);
    if (!out) { free(f32_cols); DF_ERR("df_describe: OOM tensor"); }

    float *od = F32(out);
    size_t n  = df->n_rows;

#pragma omp parallel for schedule(static)
    for (int fi = 0; fi < nf; fi++) {
        const float *d = (const float *)df->columns[f32_cols[fi]].data;
        double sum = 0.0, sum2 = 0.0;
        float  mn  =  1e38f, mx = -1e38f;
        int    cnt = 0;
        for (size_t r = 0; r < n; r++) {
            float v = d[r];
            if (v != v) continue; /* NaN */
            sum  += v;
            sum2 += (double)v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            cnt++;
        }
        double mean = cnt ? sum / cnt : 0.0;
        double var  = cnt > 1 ? (sum2 / cnt - mean * mean) : 0.0;
        /* Row fi: [count, mean, std, min, max] */
        od[fi * 5 + 0] = (float)cnt;
        od[fi * 5 + 1] = (float)mean;
        od[fi * 5 + 2] = (float)sqrt(var < 0.0 ? 0.0 : var);
        od[fi * 5 + 3] = cnt ? mn : 0.0f;
        od[fi * 5 + 4] = cnt ? mx : 0.0f;
    }

    free(f32_cols);
    return out;
}

DataFrame *df_value_counts(const DataFrame *df, int col_idx) {
    if (!df || col_idx < 0 || (size_t)col_idx >= df->n_cols)
        DF_ERR("df_value_counts: invalid argument");
    const DFColumn *col = &df->columns[col_idx];
    if (col->dtype != DF_DTYPE_STRING)
        DF_ERR("df_value_counts: column must be STRING");

    int n_cats = col->n_categories;
    int32_t *counts = (int32_t *)calloc((size_t)n_cats, sizeof(int32_t));
    if (!counts) DF_ERR("df_value_counts: OOM counts");

    const int32_t *d = (const int32_t *)col->data;
    for (size_t r = 0; r < df->n_rows; r++)
        if (d[r] >= 0 && d[r] < n_cats) counts[d[r]]++;

    /* Sort categories by count descending (argsort) */
    _SortEnt *ents = (_SortEnt *)malloc((size_t)n_cats * sizeof(_SortEnt));
    if (!ents) { free(counts); DF_ERR("df_value_counts: OOM sort ents"); }
    for (int i = 0; i < n_cats; i++) {
        ents[i].idx   = (size_t)i;
        ents[i].key_f = -(float)counts[i]; /* negate for desc sort */
    }
    qsort(ents, (size_t)n_cats, sizeof(_SortEnt), _sort_f32_asc);

    /* Build output DataFrame: [category(STRING) | count(FLOAT32)] */
    DataFrame *out = df_create((size_t)n_cats, 2);
    if (!out) { free(counts); free(ents); DF_ERR("df_value_counts: OOM df"); }

    /* Category column */
    DFColumn *cc = &out->columns[0];
    memcpy(cc->name, col->name, DF_MAX_COL_NAME);
    cc->dtype = DF_DTYPE_STRING;
    _col_copy_cats(cc, col);
    cc->data = malloc((size_t)n_cats * sizeof(int32_t));
    if (!cc->data) { free(counts); free(ents); df_free(out); DF_ERR("df_value_counts: OOM cat col"); }
    for (int i = 0; i < n_cats; i++)
        ((int32_t *)cc->data)[i] = (int32_t)ents[i].idx;

    /* Count column */
    DFColumn *cnt_col = &out->columns[1];
    strncpy(cnt_col->name, "count", DF_MAX_COL_NAME - 1);
    cnt_col->dtype = DF_DTYPE_FLOAT32;
    cnt_col->data  = malloc((size_t)n_cats * sizeof(float));
    if (!cnt_col->data) { free(counts); free(ents); df_free(out); DF_ERR("df_value_counts: OOM cnt col"); }
    for (int i = 0; i < n_cats; i++)
        ((float *)cnt_col->data)[i] = (float)counts[ents[i].idx];

    free(counts); free(ents);
    return out;
}

DataFrame *df_sample_rows(const DataFrame *df, size_t n, bool replace, uint64_t seed) {
    if (!df) DF_ERR("df_sample_rows: NULL DataFrame");
    if (n == 0) return df_create(0, df->n_cols);
    if (!replace && n > df->n_rows)
        DF_ERR("df_sample_rows: n > n_rows with replace=false");

    /* Seed the RNG */
    uint64_t state = seed ? seed : (uint64_t)time(NULL);
    /* Xorshift64 */
#define XS64(s) do { (s)^=(s)<<13; (s)^=(s)>>7; (s)^=(s)<<17; } while(0)

    size_t *idx = (size_t *)malloc(n * sizeof(size_t));
    if (!idx) DF_ERR("df_sample_rows: OOM index array");

    if (replace) {
        for (size_t i = 0; i < n; i++) {
            XS64(state);
            idx[i] = (size_t)(state % df->n_rows);
        }
    } else {
        /* Fisher-Yates on an index array of size n_rows */
        size_t *pool = (size_t *)malloc(df->n_rows * sizeof(size_t));
        if (!pool) { free(idx); DF_ERR("df_sample_rows: OOM pool"); }
        for (size_t i = 0; i < df->n_rows; i++) pool[i] = i;
        for (size_t i = 0; i < n; i++) {
            XS64(state);
            size_t j = i + (size_t)(state % (df->n_rows - i));
            size_t tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
            idx[i] = pool[i];
        }
        free(pool);
    }
#undef XS64

    DataFrame *out = _df_shell(n, df->n_cols, df->columns);
    if (!out) { free(idx); DF_ERR("df_sample_rows: OOM df shell"); }
    for (size_t c = 0; c < df->n_cols; c++) {
        if (!_col_scatter(&out->columns[c], &df->columns[c], idx, n)) {
            free(idx); df_free(out); DF_ERR("df_sample_rows: OOM col data");
        }
    }
    free(idx);
    return out;
}

/* ============================================================================
 * Categorical Encoding — Target & Frequency
 * ========================================================================== */

/* ── Target Encoding ─────────────────────────────────────────────────────── *
 * Fit: compute per-category smoothed mean using additive (James-Stein) formula:
 *   TE(cat) = (n_i * mean_i + smoothing * global_mean) / (n_i + smoothing)
 * Missing rows (cat_idx < 0) are excluded from statistics but receive global_mean.
 * Returns [n_cats] FLOAT32 tensor of smoothed means.
 * ========================================================================== */
Tensor* df_target_encode_fit(const DataFrame* df, int col_idx,
                              const Tensor* y, float smoothing)
{
    if (!df || col_idx < 0 || (size_t)col_idx >= df->n_cols)
        DF_ERR("df_target_encode_fit: invalid col_idx");
    const DFColumn* col = &df->columns[col_idx];
    if (col->dtype != DF_DTYPE_STRING)
        DF_ERR("df_target_encode_fit: column must be STRING");
    if (!y || y->ndim < 1 || (int)y->total_size != (int)df->n_rows)
        DF_ERR("df_target_encode_fit: y size mismatch");

    int N       = (int)df->n_rows;
    int n_cats  = col->n_categories;
    const int32_t* cat_idx = DF_COL_I32(col);
    const float*   yp      = (const float*)y->data;

    /* Global mean over all non-missing rows */
    double gsum = 0.0; int gcnt = 0;
    for (int i = 0; i < N; i++)
        if (cat_idx[i] >= 0) { gsum += yp[i]; gcnt++; }
    float global_mean = gcnt > 0 ? (float)(gsum / gcnt) : 0.0f;

    /* Per-category accumulation */
    double* cat_sum   = (double*)calloc((size_t)n_cats, sizeof(double));
    int*    cat_count = (int*)   calloc((size_t)n_cats, sizeof(int));
    if (!cat_sum || !cat_count) {
        free(cat_sum); free(cat_count);
        DF_ERR("df_target_encode_fit: OOM accumulators");
    }

    for (int i = 0; i < N; i++) {
        int c = cat_idx[i];
        if (c >= 0 && c < n_cats) {
            cat_sum[c]   += yp[i];
            cat_count[c] += 1;
        }
    }

    /* Smoothed means */
    int shape = n_cats;
    Tensor* out = tensor_zeros(1, &shape);
    if (!out) { free(cat_sum); free(cat_count); DF_ERR("OOM"); }
    float* op = (float*)out->data;

    for (int c = 0; c < n_cats; c++) {
        float cat_mean = cat_count[c] > 0 ? (float)(cat_sum[c] / cat_count[c]) : global_mean;
        op[c] = ((float)cat_count[c] * cat_mean + smoothing * global_mean)
                / ((float)cat_count[c] + smoothing);
    }

    free(cat_sum); free(cat_count);
    return out;
}

/* ── Target Encoding — Transform ─────────────────────────────────────────── *
 * Map each row's category index → its smoothed mean from the fit step.
 * Missing rows (cat_idx < 0) receive global_mean.
 * Returns [N] FLOAT32 tensor.
 * ========================================================================== */
Tensor* df_target_encode_transform(const DataFrame* df, int col_idx,
                                    const Tensor* cat_means, float global_mean)
{
    if (!df || col_idx < 0 || (size_t)col_idx >= df->n_cols)
        DF_ERR("df_target_encode_transform: invalid col_idx");
    const DFColumn* col = &df->columns[col_idx];
    if (col->dtype != DF_DTYPE_STRING)
        DF_ERR("df_target_encode_transform: column must be STRING");
    if (!cat_means || cat_means->ndim != 1)
        DF_ERR("df_target_encode_transform: cat_means must be [n_cats]");

    int N       = (int)df->n_rows;
    int n_cats  = (int)cat_means->total_size;
    const int32_t* cat_idx = DF_COL_I32(col);
    const float*   mp      = (const float*)cat_means->data;

    int shape = N;
    Tensor* out = tensor_create_uninitialized(1, &shape, DTYPE_FLOAT32);
    if (!out) DF_ERR("df_target_encode_transform: OOM");
    float* op = (float*)out->data;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        int c = cat_idx[i];
        op[i] = (c >= 0 && c < n_cats) ? mp[c] : global_mean;
    }
    return out;
}

/* ── Frequency Encoding — Fit ────────────────────────────────────────────── *
 * Compute fraction-of-total for each category.
 * Missing rows (cat_idx < 0) are excluded from total.
 * Returns [n_cats] FLOAT32 tensor of frequencies.
 * ========================================================================== */
Tensor* df_freq_encode_fit(const DataFrame* df, int col_idx)
{
    if (!df || col_idx < 0 || (size_t)col_idx >= df->n_cols)
        DF_ERR("df_freq_encode_fit: invalid col_idx");
    const DFColumn* col = &df->columns[col_idx];
    if (col->dtype != DF_DTYPE_STRING)
        DF_ERR("df_freq_encode_fit: column must be STRING");

    int N      = (int)df->n_rows;
    int n_cats = col->n_categories;
    const int32_t* cat_idx = DF_COL_I32(col);

    int* counts = (int*)calloc((size_t)n_cats, sizeof(int));
    if (!counts) DF_ERR("df_freq_encode_fit: OOM");

    int total = 0;
    for (int i = 0; i < N; i++) {
        int c = cat_idx[i];
        if (c >= 0 && c < n_cats) { counts[c]++; total++; }
    }

    int shape = n_cats;
    Tensor* out = tensor_zeros(1, &shape);
    if (!out) { free(counts); DF_ERR("OOM"); }
    float* op = (float*)out->data;
    float inv = total > 0 ? 1.0f / (float)total : 0.0f;
    for (int c = 0; c < n_cats; c++) op[c] = counts[c] * inv;

    free(counts);
    return out;
}

/* ── Frequency Encoding — Transform ─────────────────────────────────────── *
 * Map each row's category → its frequency.  Missing → 0.
 * Returns [N] FLOAT32 tensor.
 * ========================================================================== */
Tensor* df_freq_encode_transform(const DataFrame* df, int col_idx,
                                  const Tensor* cat_freqs)
{
    if (!df || col_idx < 0 || (size_t)col_idx >= df->n_cols)
        DF_ERR("df_freq_encode_transform: invalid col_idx");
    const DFColumn* col = &df->columns[col_idx];
    if (col->dtype != DF_DTYPE_STRING)
        DF_ERR("df_freq_encode_transform: column must be STRING");
    if (!cat_freqs || cat_freqs->ndim != 1)
        DF_ERR("df_freq_encode_transform: cat_freqs must be [n_cats]");

    int N      = (int)df->n_rows;
    int n_cats = (int)cat_freqs->total_size;
    const int32_t* cat_idx = DF_COL_I32(col);
    const float*   fp      = (const float*)cat_freqs->data;

    int shape = N;
    Tensor* out = tensor_create_uninitialized(1, &shape, DTYPE_FLOAT32);
    if (!out) DF_ERR("df_freq_encode_transform: OOM");
    float* op = (float*)out->data;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        int c = cat_idx[i];
        op[i] = (c >= 0 && c < n_cats) ? fp[c] : 0.0f;
    }
    return out;
}

/* ── Direct Tensor → DataFrame column add (zero-copy from tensor side) ───── *
 * Avoids PHP-side float copies when adding an encoded tensor as a column.
 * ========================================================================== */
DataFrame* df_add_tensor_f32_column(const DataFrame* df, const char* name,
                                     const Tensor* t)
{
    if (!t || t->dtype != DTYPE_FLOAT32)
        DF_ERR("df_add_tensor_f32_column: tensor must be FLOAT32");
    if (!tensor_is_contiguous(t))
        DF_ERR("df_add_tensor_f32_column: tensor must be contiguous");
    if ((size_t)t->total_size != df->n_rows)
        DF_ERR("df_add_tensor_f32_column: tensor size != n_rows");
    return df_add_f32_column(df, name, (const float*)t->data, df->n_rows);
}
