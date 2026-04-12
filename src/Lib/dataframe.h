#ifndef DATAFRAME_H
#define DATAFRAME_H

/*
 * dataframe.h — Columnar DataFrame for high-performance ETL
 *
 * Design goals:
 *   - Column-major storage: each column is a single contiguous allocation.
 *     Iterating one column = linear memory access = hardware prefetcher happy.
 *   - Mixed types: FLOAT32, INT32, STRING (stored as interned category indices).
 *   - Opaque C pointer exposed to PHP via FFI — PHP never sees struct internals.
 *   - Zero PHP ↔ C array copies during ingestion: CSV → C columns directly.
 *   - All allocations are checked; df_free() is the single deallocation path.
 */

#include "tensor.h"     /* TensorC, TensorDType, safe_malloc, safe_free */
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1.  COLUMN DATA TYPES
 * ========================================================================== */

typedef enum {
    DF_DTYPE_FLOAT32 = 0,   /* float*   — NaN sentinel for missing values      */
    DF_DTYPE_INT32   = 1,   /* int32_t* — INT32_MIN sentinel for missing        */
    DF_DTYPE_STRING  = 2    /* int32_t* (category index); -1 = missing          */
} DFDType;

/* Maximum column name length (including NUL terminator). */
#define DF_MAX_COL_NAME  64

/* Scratch buffer size for quoted-field unescaping during CSV parse. */
#define DF_FIELD_SCRATCH 4096

/* ============================================================================
 * 2.  COLUMN STRUCT
 *
 * `data`       : heap-allocated column buffer.
 *   FLOAT32    → float[n_rows]
 *   INT32      → int32_t[n_rows]
 *   STRING     → int32_t[n_rows]  (index into categories[])
 *
 * `categories` : STRING columns only — interned distinct string values.
 * `n_categories`: number of distinct category strings.
 * `_cat_cap`   : internal capacity of `categories`; not visible to PHP.
 * ========================================================================== */

typedef struct {
    char     name[DF_MAX_COL_NAME];
    DFDType  dtype;
    void    *data;          /* float* | int32_t*                                */
    char   **categories;    /* STRING only: categories[0..n_categories)         */
    int32_t  n_categories;  /* STRING only: distinct value count                */
    int32_t  _cat_cap;      /* internal: allocated capacity of categories[]     */
} DFColumn;

/* ============================================================================
 * 3.  DATAFRAME STRUCT
 * ========================================================================== */

typedef struct {
    size_t    n_rows;
    size_t    n_cols;
    DFColumn *columns;      /* columns[0..n_cols)                               */
} DataFrame;

/* ============================================================================
 * 4.  LIFECYCLE
 * ========================================================================== */

/**
 * Allocate a DataFrame shell (n_rows × n_cols). Column data buffers are NOT
 * allocated here; callers fill them in via column initialisation helpers.
 * Returns NULL on allocation failure.
 */
DataFrame *df_create(size_t n_rows, size_t n_cols);

/**
 * Release all C memory owned by df, including column buffers, category
 * strings, and the DataFrame itself. Safe to call with NULL.
 */
void df_free(DataFrame *df);

/* ============================================================================
 * 5.  INGESTION
 * ========================================================================== */

/**
 * Parse a CSV file directly into a columnar DataFrame.
 *
 * - Uses mmap(2) + MADV_SEQUENTIAL for maximum read bandwidth.
 * - Two-pass: first pass counts rows and detects column types; second pass
 *   fills pre-allocated column buffers — no realloc during fill.
 * - Type detection: if every non-empty value in a column parses as float32,
 *   the column is FLOAT32; otherwise STRING (categorical).
 * - Missing / empty fields → NaN (FLOAT32), INT32_MIN (INT32), or -1 (STRING).
 * - Handles RFC 4180 quoted fields ("a,b","c") and CRLF line endings.
 *
 * @param filepath   Path to the CSV file (NUL-terminated).
 * @param has_header True if the first row contains column names.
 * @return Heap-allocated DataFrame*, or NULL on error.
 */
DataFrame *df_read_csv(const char *filepath, bool has_header);

/* ============================================================================
 * 6.  ETL OPERATIONS  (all return a new DataFrame; caller owns it)
 * ========================================================================== */

/**
 * Return a new DataFrame containing only the requested columns.
 * Column data is deep-copied (categories included).
 *
 * @param col_indices  Array of column indices to keep (0-based).
 * @param n            Length of col_indices.
 */
DataFrame *df_select_columns(const DataFrame *df,
                              const int *col_indices, int n);

/**
 * Return a new DataFrame with every row that contains at least one missing
 * value removed.
 * Missing sentinel: NaN (FLOAT32), INT32_MIN (INT32), index < 0 (STRING).
 */
DataFrame *df_drop_nans(const DataFrame *df);

/**
 * One-hot encode a STRING column.
 *
 * Replaces column `col_idx` with `n_categories` new FLOAT32 columns named
 * `{original_name}_{category_value}`. The original STRING column is removed.
 * All other columns are deep-copied into the returned DataFrame.
 *
 * @param col_idx  Index of the STRING column to encode (0-based).
 * @return New DataFrame, or NULL if col_idx is out of range or not STRING.
 */
DataFrame *df_one_hot_encode(const DataFrame *df, int col_idx);

/* ============================================================================
 * 7.  TENSOR INTEGRATION
 * ========================================================================== */

/**
 * Pack numeric columns (FLOAT32 or INT32) into a row-major [n_rows × n]
 * Tensor compatible with the existing tensor math / ML pipeline.
 *
 * INT32 column values are cast to float32. STRING columns are rejected
 * (function sets the global tensor error and returns NULL).
 *
 * @param col_indices  Column indices to include (must be FLOAT32 or INT32).
 * @param n            Number of columns to pack.
 * @return Newly allocated Tensor* [n_rows × n] FLOAT32, or NULL on error.
 */
Tensor *df_to_tensor(const DataFrame *df,
                     const int *col_indices, int n);

/* ============================================================================
 * 8.  INTROSPECTION  (safe opaque accessors for the PHP FFI layer)
 * ========================================================================== */

size_t      df_num_rows(const DataFrame *df);
size_t      df_num_cols(const DataFrame *df);

/**
 * Column name — pointer is valid for the lifetime of df. Do not free.
 * Returns NULL if col_idx is out of range.
 */
const char *df_col_name(const DataFrame *df, int col_idx);

/**
 * Column DFDType as int (0 = FLOAT32, 1 = INT32, 2 = STRING).
 * Returns -1 if col_idx is out of range.
 */
int         df_col_dtype(const DataFrame *df, int col_idx);

/**
 * Number of distinct categories in a STRING column.
 * Returns 0 for FLOAT32/INT32 columns or out-of-range indices.
 */
int         df_col_n_categories(const DataFrame *df, int col_idx);

/**
 * Category string for a STRING column's category index.
 * Returns NULL if arguments are out of range.
 * Pointer is valid for the lifetime of df. Do not free.
 */
const char *df_col_category_name(const DataFrame *df,
                                  int col_idx, int cat_idx);

#ifdef __cplusplus
}
#endif

#endif /* DATAFRAME_H */
