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
            if      (col->dtype == DF_DTYPE_FLOAT32 && isnan(DF_COL_F32(col)[r])) keep[r] = false;
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
