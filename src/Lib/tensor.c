#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <immintrin.h>
#include <cblas.h>     
#include <lapacke.h>
#include <omp.h>

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

// ============================================================================
// DTYPE POINTER MACROS (Zero-overhead void* casting)
// ============================================================================
#define F32(tensor) ((float*)(tensor)->data)
#define I32(tensor) ((int32_t*)(tensor)->data)
#define I64(tensor) ((int64_t*)(tensor)->data)

static inline size_t dtype_size(TensorDType dtype) {
    switch(dtype) {
        case DTYPE_FLOAT32: return sizeof(float);
        case DTYPE_INT32:   return sizeof(int32_t);
        case DTYPE_INT64:   return sizeof(int64_t);
        default: return sizeof(float);
    }
}

// ============================================================================
// 0. GLOBAL ERROR HANDLING (Replaces exit(1))
// ============================================================================
static char tensor_last_error[512] = {0};
static bool tensor_had_error = false;

bool tensor_check_error(void) { return tensor_had_error; }
const char* tensor_get_last_error(void) { return tensor_last_error; }
void tensor_clear_error(void) { tensor_had_error = false; tensor_last_error[0] = 0; }

/* Called by dataframe.c and other translation units that share the .so */
void tensor_set_error(const char *msg) {
    snprintf(tensor_last_error, sizeof(tensor_last_error), "%s", msg);
    tensor_had_error = true;
}

#define TENSOR_ERROR(fmt, ...) do { \
    snprintf(tensor_last_error, sizeof(tensor_last_error), fmt, ##__VA_ARGS__); \
    tensor_had_error = true; \
    return NULL; \
} while(0)

#define TENSOR_ERROR_VOID(fmt, ...) do { \
    snprintf(tensor_last_error, sizeof(tensor_last_error), fmt, ##__VA_ARGS__); \
    tensor_had_error = true; \
    return; \
} while(0)

#define TENSOR_ERROR_VAL(ret, fmt, ...) do { \
    snprintf(tensor_last_error, sizeof(tensor_last_error), fmt, ##__VA_ARGS__); \
    tensor_had_error = true; \
    return ret; \
} while(0)

// ============================================================================
// 1. SAFE MEMORY & LIFECYCLE
// ============================================================================
void* safe_malloc(size_t size) {
    if (size == 0) return NULL;
    void* ptr = malloc(size);
    if (!ptr) TENSOR_ERROR_VAL(NULL, "FATAL: Out of memory (malloc).");
    memset(ptr, 0, size);
    return ptr;
}

void* safe_memalign(size_t alignment, size_t size) {
    if (size == 0) return NULL;
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0 || !ptr) {
        TENSOR_ERROR_VAL(NULL, "FATAL: Memalign failed.");
    }
    memset(ptr, 0, size);
    return ptr;
}

void safe_free(void** ptr) {
    if (ptr && *ptr) { free(*ptr); *ptr = NULL; }
}

// --- ARENA ALLOCATOR ---
TensorArena* arena_create(size_t capacity) {
    TensorArena* arena = (TensorArena*)malloc(sizeof(TensorArena));
    if (!arena) TENSOR_ERROR("FATAL: arena_create struct alloc failed.");
    if (posix_memalign((void**)&arena->memory, 32, capacity) != 0) {
        free(arena); return NULL;
    }
    arena->capacity = capacity;
    arena->offset = 0;
    return arena;
}

void* arena_alloc(TensorArena* arena, size_t size) {
    size_t aligned_size = (size + 31) & ~31; 
    if (arena->offset + aligned_size > arena->capacity) return NULL;
    void* ptr = arena->memory + arena->offset;
    arena->offset += aligned_size;
    return ptr;
}

void arena_reset(TensorArena* arena) { arena->offset = 0; }
void arena_destroy(TensorArena* arena) { free(arena->memory); free(arena); }

void _compute_strides(Tensor* t) {
    t->stride[t->ndim - 1] = 1;
    for (int i = t->ndim - 2; i >= 0; i--) {
        t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
    }
}

bool tensor_is_contiguous(const Tensor* t) {
    size_t expected_stride = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->stride[i] != expected_stride) return false;
        expected_stride *= t->shape[i];
    }
    return true;
}

// Uninitialized alloc bypasses expensive memsets on the data payload,
// but zeroes the struct itself so shape[]/stride[] padding fields are clean.
Tensor* tensor_create_uninitialized(int ndim, int* shape, TensorDType dtype) {
    if (ndim < 1 || ndim > 8) TENSOR_ERROR("FATAL: Invalid ndim.");
    Tensor* t = (Tensor*)calloc(1, sizeof(Tensor));
    if (!t) TENSOR_ERROR("FATAL: Out of memory (Tensor struct).");
    t->ndim = ndim;
    t->total_size = 1;
    t->dtype = dtype;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->total_size *= shape[i];
    }
    _compute_strides(t);
    t->byte_size = t->total_size * dtype_size(dtype);
    void* data = NULL;
    // posix_memalign requires size > 0; fall back to a 32-byte sentinel for empty tensors
    size_t alloc_size = t->byte_size > 0 ? t->byte_size : 32;
    if (posix_memalign(&data, 32, alloc_size) != 0) {
        free(t);
        TENSOR_ERROR("FATAL: Memalign failed.");
    }
    t->data = data;
    t->owns_data = true;
    t->is_arena = false;
    return t;
}

Tensor* tensor_create_dtype(int ndim, int* shape, TensorDType dtype) {
    Tensor* t = tensor_create_uninitialized(ndim, shape, dtype);
    if (!t) return NULL;
    memset(t->data, 0, t->byte_size);
    return t;
}

Tensor* tensor_create_arena(int ndim, int* shape, TensorDType dtype, TensorArena* arena) {
    if (ndim < 1 || ndim > 8 || !arena) TENSOR_ERROR("FATAL: Invalid arena params.");
    Tensor* t = (Tensor*)arena_alloc(arena, sizeof(Tensor));
    if (!t) TENSOR_ERROR("FATAL: Arena OOM (Struct).");
    
    t->ndim = ndim;
    t->total_size = 1;
    t->dtype = dtype;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->total_size *= shape[i];
    }
    _compute_strides(t);
    t->byte_size = t->total_size * dtype_size(dtype);
    
    t->data = arena_alloc(arena, t->byte_size);
    if (!t->data) TENSOR_ERROR("FATAL: Arena OOM (Payload).");
    
    memset(t->data, 0, t->byte_size);
    t->owns_data = false;
    t->is_arena = true;
    return t;
}

Tensor* tensor_create(int ndim, int* shape) {
    return tensor_create_dtype(ndim, shape, DTYPE_FLOAT32);
}

// Bypasses deep alloc + safe_free entirely
Tensor* tensor_from_external(void* data, int ndim, int* shape, TensorDType dtype) {
    if (ndim < 1 || ndim > 8) TENSOR_ERROR("FATAL: Invalid ndim.");
    Tensor* t = (Tensor*)safe_malloc(sizeof(Tensor));
    if (!t) TENSOR_ERROR("FATAL: Malloc failed.");
    t->ndim = ndim;
    t->dtype = dtype;
    t->total_size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->total_size *= shape[i];
    }
    _compute_strides(t);
    t->byte_size = t->total_size * dtype_size(dtype);
    t->data = data;      
    t->owns_data = false; 
    t->is_arena = false;
    return t;
}

void tensor_free(Tensor* t) {
    if (t) {
        if (t->is_arena) return; 
        if (t->owns_data) safe_free(&t->data);
        safe_free((void**)&t);
    }
}

Tensor* tensor_copy(Tensor* A) {
    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, A->dtype);
    if (!out) return NULL;
    if (tensor_is_contiguous(A)) {
        memcpy(out->data, A->data, A->byte_size);
    } else {
        int idx[8] = {0};
        size_t el_size = dtype_size(A->dtype);
        for (size_t i = 0; i < out->total_size; i++) {
            size_t offset_a = 0;
            for (int d = 0; d < A->ndim; d++) offset_a += idx[d] * A->stride[d];
            memcpy((char*)out->data + i * el_size, (char*)A->data + offset_a * el_size, el_size);
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return out;
}

bool _tensor_shape_assert(Tensor* A, Tensor* B, const char* op) {
    if (A->total_size != B->total_size) {
        TENSOR_ERROR_VAL(false, "FATAL [%s]: Shape mismatch.", op);
    }
    return true;
}

// ============================================================================
// 2. INITIALIZERS & CREATION
// ============================================================================

void tensor_fill(Tensor* t, float val) {
    if (t->dtype != DTYPE_FLOAT32) return;
    if (tensor_is_contiguous(t)) {
        size_t i = 0;
#ifdef __AVX2__
        __m256 v_val = _mm256_set1_ps(val);
        for (; i + 7 < t->total_size; i += 8) _mm256_storeu_ps(&F32(t)[i], v_val);
#endif
        for (; i < t->total_size; i++) F32(t)[i] = val;
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < t->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < t->ndim; d++) offset += idx[d] * t->stride[d];
            F32(t)[offset] = val;
            for (int d = t->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < t->shape[d]) break; idx[d] = 0;
            }
        }
    }
}

// tensor_create_dtype already zeroes via memset — no second fill needed.
Tensor* tensor_zeros(int ndim, int* shape) { return tensor_create_dtype(ndim, shape, DTYPE_FLOAT32); }
// tensor_ones: uninitialized alloc + single AVX2 fill (saves one full memory pass).
Tensor* tensor_ones(int ndim, int* shape) {
    Tensor* t = tensor_create_uninitialized(ndim, shape, DTYPE_FLOAT32);
    if (t) tensor_fill(t, 1.0f);
    return t;
}

Tensor* tensor_range(float start, float end, float step) {
    int size = (int)ceilf((end - start) / step);
    if (size <= 0) size = 1;
    Tensor* t = tensor_create(1, &size);
    for (int i = 0; i < size; i++) F32(t)[i] = start + i * step;
    return t;
}

Tensor* tensor_linspace(float start, float end, int steps) {
    if (steps <= 0) steps = 1;
    Tensor* t = tensor_create(1, &steps);
    float step_val = (steps > 1) ? (end - start) / (steps - 1) : 0;
    for (int i = 0; i < steps; i++) F32(t)[i] = start + i * step_val;
    return t;
}

// High-entropy seed: mixes thread ID, microsecond wall time, and a Knuth multiplier.
// Avoids seed collision when multiple calls land within the same second.
static inline unsigned int _make_thread_seed(int tid) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (unsigned int)((uint64_t)tid * 2654435761u
                        ^ (uint64_t)tv.tv_usec * 1000003u
                        ^ (uint64_t)tv.tv_sec  * 999983u);
}

void tensor_random_uniform(Tensor* t, float min_val, float max_val) {
    if (!tensor_is_contiguous(t) || t->dtype != DTYPE_FLOAT32) return;
    #pragma omp parallel
    {
        unsigned int seed = _make_thread_seed(omp_get_thread_num());
        #pragma omp for
        for (size_t i = 0; i < t->total_size; i++) {
            F32(t)[i] = min_val + ((float)rand_r(&seed) / RAND_MAX) * (max_val - min_val);
        }
    }
}

void tensor_random_normal(Tensor* t, float mean, float stddev) {
    if(!tensor_is_contiguous(t) || t->dtype != DTYPE_FLOAT32) return;
    #pragma omp parallel
    {
        unsigned int seed = _make_thread_seed(omp_get_thread_num());
        #pragma omp for
        for (size_t i = 0; i < t->total_size; i += 2) {
            float u1 = fmaxf(1e-7f, (float)rand_r(&seed) / RAND_MAX);
            float u2 = (float)rand_r(&seed) / RAND_MAX;
            float mag = sqrtf(-2.0f * logf(u1));
            
            F32(t)[i] = mag * cosf(2.0f * (float)M_PI * u2) * stddev + mean;
            if (i + 1 < t->total_size) {
                F32(t)[i + 1] = mag * sinf(2.0f * (float)M_PI * u2) * stddev + mean;
            }
        }
    }
}

// ============================================================================
// 3. ZERO-COPY VIEWS, BROADCASTING & MUTATIONS
// ============================================================================

Tensor* tensor_view(Tensor* t) {
    Tensor* v = (Tensor*)safe_malloc(sizeof(Tensor));
    if (!v) TENSOR_ERROR("FATAL: Malloc failed for view.");
    memcpy(v, t, sizeof(Tensor));
    v->owns_data = false; 
    v->is_arena = false; // View struct lives on heap and must be freed
    return v;
}

Tensor* tensor_row_view(Tensor* t, int row) {
    if (t->ndim != 2 || row < 0 || row >= t->shape[0]) return NULL;
    Tensor* v = tensor_view(t);
    v->ndim = 1;
    v->shape[0] = t->shape[1];
    v->stride[0] = t->stride[1];
    v->total_size = t->shape[1];
    v->byte_size = v->total_size * dtype_size(t->dtype);
    v->data = (char*)v->data + (row * t->stride[0] * dtype_size(t->dtype));
    return v;
}

Tensor* tensor_column_view(Tensor* t, int col) {
    if (t->ndim != 2 || col < 0 || col >= t->shape[1]) return NULL;
    Tensor* v = tensor_view(t);
    v->ndim = 1;
    v->shape[0] = t->shape[0];
    v->stride[0] = t->stride[0];
    v->total_size = t->shape[0];
    v->byte_size = v->total_size * dtype_size(t->dtype); 
    v->data = (char*)v->data + (col * t->stride[1] * dtype_size(t->dtype));
    return v;
}

Tensor* tensor_flatten(Tensor* t) {
    int shape[1] = { (int)t->total_size };
    return tensor_reshape(t, 1, shape);
}

Tensor* tensor_expand_dims(Tensor* t, int axis) {
    if (axis < 0 || axis > t->ndim || t->ndim >= 8) {
        TENSOR_ERROR("FATAL [ExpandDims]: Invalid axis.");
    }
    int new_shape[8];
    for (int i = 0, j = 0; i <= t->ndim; i++) {
        if (i == axis) new_shape[i] = 1;
        else new_shape[i] = t->shape[j++];
    }
    return tensor_reshape(t, t->ndim + 1, new_shape);
}

Tensor* tensor_squeeze(Tensor* t) {
    int new_shape[8];
    int new_ndim = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (t->shape[i] != 1) new_shape[new_ndim++] = t->shape[i];
    }
    if (new_ndim == 0) { new_ndim = 1; new_shape[0] = 1; }
    return tensor_reshape(t, new_ndim, new_shape);
}

Tensor* tensor_slice(Tensor* t, int axis, int start, int length) {
    if (axis < 0 || axis >= t->ndim || start + length > t->shape[axis]) return NULL;
    Tensor* v = tensor_view(t);
    v->shape[axis] = length;
    v->total_size = 1;
    for (int i = 0; i < v->ndim; i++) v->total_size *= v->shape[i];
    v->byte_size = v->total_size * dtype_size(t->dtype);
    v->data = (char*)v->data + (start * v->stride[axis] * dtype_size(t->dtype));
    return v;
}

bool tensor_broadcast_shapes(const Tensor* A, const Tensor* B, int* out_ndim, int* out_shape, size_t* out_stride_A, size_t* out_stride_B) {
    int ndim = (A->ndim > B->ndim) ? A->ndim : B->ndim;
    *out_ndim = ndim;
    int offset_A = ndim - A->ndim;
    int offset_B = ndim - B->ndim;

    for (int i = 0; i < ndim; i++) {
        int dim_A = (i >= offset_A) ? A->shape[i - offset_A] : 1;
        int dim_B = (i >= offset_B) ? B->shape[i - offset_B] : 1;
        if (dim_A != dim_B && dim_A != 1 && dim_B != 1) return false;
        out_shape[i] = (dim_A > dim_B) ? dim_A : dim_B;
        out_stride_A[i] = (dim_A == 1) ? 0 : A->stride[i - offset_A];
        out_stride_B[i] = (dim_B == 1) ? 0 : B->stride[i - offset_B];
    }
    return true;
}

bool _check_inplace_broadcast(Tensor* A, Tensor* B) {
    if (A->total_size != B->total_size || A->ndim != B->ndim) {
        int out_ndim, out_shape[8];
        size_t stride_A[8] = {0}, stride_B[8] = {0};
        if (!tensor_broadcast_shapes(A, B, &out_ndim, out_shape, stride_A, stride_B)) return false;
        for (int d = 0; d < out_ndim; d++) {
            if (stride_A[d] == 0 && out_shape[d] > 1) {
                return false;
            }
        }
    }
    return true;
}

Tensor* tensor_reshape(Tensor* t, int ndim, int* new_shape) {
    size_t new_total = 1;
    for (int i=0; i < ndim; i++) new_total *= new_shape[i];
    if (new_total != t->total_size) TENSOR_ERROR("FATAL [Reshape]: Size mismatch.");
    
    if (tensor_is_contiguous(t)) {
        Tensor* out = tensor_view(t);
        out->ndim = ndim;
        for (int i=0; i < ndim; i++) out->shape[i] = new_shape[i];
        _compute_strides(out);
        return out;
    }
    Tensor* compacted = tensor_copy(t);
    compacted->ndim = ndim;
    for (int i=0; i < ndim; i++) compacted->shape[i] = new_shape[i];
    _compute_strides(compacted);
    return compacted;
}

Tensor* tensor_transpose_2d(Tensor* t) {
    if (t->ndim != 2) TENSOR_ERROR("FATAL [Transpose]: Only 2D supported.");
    Tensor* out = tensor_view(t);
    out->shape[0] = t->shape[1]; out->shape[1] = t->shape[0];
    out->stride[0] = t->stride[1]; out->stride[1] = t->stride[0];
    return out;
}

Tensor* tensor_transpose_nd(Tensor* t, int* axes) {
    Tensor* out = tensor_view(t);
    for (int i = 0; i < t->ndim; i++) {
        out->shape[i] = t->shape[axes[i]];
        out->stride[i] = t->stride[axes[i]];
    }
    return out;
}

// ============================================================================
// 4. SIMD MATH & BINARY OPS (Broadcasting Supported, Float Only)
// ============================================================================
#define TENSOR_BINARY_IMPL(OP_NAME, AVX_INTRINSIC, SCALAR_OP, OUT_TENSOR, IN_PLACE, ON_ERR) \
    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32) { \
        ON_ERR("FATAL [%s]: Math operations require FLOAT32.", #OP_NAME); \
    } \
    if (tensor_is_contiguous(A) && tensor_is_contiguous(B) && A->total_size == B->total_size) { \
        if (!IN_PLACE) OUT_TENSOR = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32); \
        size_t limit = A->total_size - (A->total_size % 8); \
        _Pragma("omp parallel for if(limit > 100000)") \
        for (size_t i = 0; i < limit; i += 8) { \
            __m256 a = _mm256_loadu_ps(&F32(A)[i]); \
            __m256 b = _mm256_loadu_ps(&F32(B)[i]); \
            _mm256_storeu_ps(&F32(OUT_TENSOR)[i], AVX_INTRINSIC(a, b)); \
        } \
        for (size_t i = limit; i < A->total_size; i++) { \
            F32(OUT_TENSOR)[i] = F32(A)[i] SCALAR_OP F32(B)[i]; \
        } \
    } else { \
        int out_ndim, out_shape[8]; \
        size_t stride_A[8] = {0}, stride_B[8] = {0}; \
        if (!tensor_broadcast_shapes(A, B, &out_ndim, out_shape, stride_A, stride_B)) { \
            ON_ERR("FATAL: Shapes not broadcastable."); \
        } \
        if (!IN_PLACE) OUT_TENSOR = tensor_create_uninitialized(out_ndim, out_shape, DTYPE_FLOAT32); \
        int idx[8] = {0}; \
        for (size_t i = 0; i < OUT_TENSOR->total_size; i++) { \
            size_t offset_a = 0, offset_b = 0; \
            for (int d = 0; d < out_ndim; d++) { \
                offset_a += idx[d] * stride_A[d]; \
                offset_b += idx[d] * stride_B[d]; \
            } \
            if (IN_PLACE) F32(OUT_TENSOR)[offset_a] = F32(A)[offset_a] SCALAR_OP F32(B)[offset_b]; \
            else F32(OUT_TENSOR)[i] = F32(A)[offset_a] SCALAR_OP F32(B)[offset_b]; \
            for (int d = out_ndim - 1; d >= 0; d--) { \
                idx[d]++; if (idx[d] < out_shape[d]) break; idx[d] = 0; \
            } \
        } \
    }

Tensor* tensor_add(Tensor* A, Tensor* B) { Tensor* out = NULL; TENSOR_BINARY_IMPL(add, _mm256_add_ps, +, out, 0, TENSOR_ERROR); return out; }
Tensor* tensor_sub(Tensor* A, Tensor* B) { Tensor* out = NULL; TENSOR_BINARY_IMPL(sub, _mm256_sub_ps, -, out, 0, TENSOR_ERROR); return out; }
Tensor* tensor_mul(Tensor* A, Tensor* B) { Tensor* out = NULL; TENSOR_BINARY_IMPL(mul, _mm256_mul_ps, *, out, 0, TENSOR_ERROR); return out; }
Tensor* tensor_div(Tensor* A, Tensor* B) { Tensor* out = NULL; TENSOR_BINARY_IMPL(div, _mm256_div_ps, /, out, 0, TENSOR_ERROR); return out; }

void tensor_add_inplace(Tensor* A, Tensor* B) {
    if (!_check_inplace_broadcast(A, B)) TENSOR_ERROR_VOID("FATAL: Cannot inplace-broadcast A over B.");
    TENSOR_BINARY_IMPL(addi, _mm256_add_ps, +, A, 1, TENSOR_ERROR_VOID);
}
void tensor_sub_inplace(Tensor* A, Tensor* B) {
    if (!_check_inplace_broadcast(A, B)) TENSOR_ERROR_VOID("FATAL: Cannot inplace-broadcast A over B.");
    TENSOR_BINARY_IMPL(subi, _mm256_sub_ps, -, A, 1, TENSOR_ERROR_VOID);
}
void tensor_mul_inplace(Tensor* A, Tensor* B) {
    if (!_check_inplace_broadcast(A, B)) TENSOR_ERROR_VOID("FATAL: Cannot inplace-broadcast A over B.");
    TENSOR_BINARY_IMPL(muli, _mm256_mul_ps, *, A, 1, TENSOR_ERROR_VOID);
}
void tensor_div_inplace(Tensor* A, Tensor* B) {
    if (!_check_inplace_broadcast(A, B)) TENSOR_ERROR_VOID("FATAL: Cannot inplace-broadcast A over B.");
    TENSOR_BINARY_IMPL(divi, _mm256_div_ps, /, A, 1, TENSOR_ERROR_VOID);
}

#define TENSOR_SCALAR_IMPL(AVX_INTRINSIC, SCALAR_OP, OUT_TENSOR, IN_PLACE, ON_ERR) \
    if (A->dtype != DTYPE_FLOAT32) ON_ERR("FATAL: Scalar math requires FLOAT32."); \
    if (tensor_is_contiguous(A)) { \
        __m256 v_val = _mm256_set1_ps(val); \
        size_t limit = A->total_size - (A->total_size % 8); \
        _Pragma("omp parallel for if(limit > 100000)") \
        for (size_t i = 0; i < limit; i += 8) { \
            __m256 a = _mm256_loadu_ps(&F32(A)[i]); \
            _mm256_storeu_ps(&F32(OUT_TENSOR)[i], AVX_INTRINSIC(a, v_val)); \
        } \
        for (size_t i = limit; i < A->total_size; i++) { \
            F32(OUT_TENSOR)[i] = F32(A)[i] SCALAR_OP val; \
        } \
    }else { \
        int idx[8] = {0}; \
        for (size_t i = 0; i < OUT_TENSOR->total_size; i++) { \
            size_t offset_a = 0; \
            for (int d = 0; d < A->ndim; d++) offset_a += idx[d] * A->stride[d]; \
            if (IN_PLACE) F32(OUT_TENSOR)[offset_a] = F32(A)[offset_a] SCALAR_OP val; \
            else F32(OUT_TENSOR)[i] = F32(A)[offset_a] SCALAR_OP val; \
            for (int d = A->ndim - 1; d >= 0; d--) { \
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0; \
            } \
        } \
    }

Tensor* tensor_add_scalar(Tensor* A, float val) { Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32); TENSOR_SCALAR_IMPL(_mm256_add_ps, +, out, 0, TENSOR_ERROR); return out; }
Tensor* tensor_mul_scalar(Tensor* A, float val) { Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32); TENSOR_SCALAR_IMPL(_mm256_mul_ps, *, out, 0, TENSOR_ERROR); return out; }
void tensor_add_scalar_inplace(Tensor* A, float val) { TENSOR_SCALAR_IMPL(_mm256_add_ps, +, A, 1, TENSOR_ERROR_VOID); }
void tensor_mul_scalar_inplace(Tensor* A, float val) { TENSOR_SCALAR_IMPL(_mm256_mul_ps, *, A, 1, TENSOR_ERROR_VOID); }

// ============================================================================
// 5. UNARY MATH & LOGICAL
// ============================================================================

static inline float _square(float x) { return x * x; }
static inline float _sign(float x) { return (x > 0.0f) - (x < 0.0f); }
static inline float _sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

#define TENSOR_MATH_UNARY(OP_NAME, MATH_FUNC) \
Tensor* OP_NAME(Tensor* A) { \
    if (A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("FATAL: Unary math requires FLOAT32."); \
    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32); \
    if (tensor_is_contiguous(A)) { \
        _Pragma("omp parallel for simd if(A->total_size > 100000)") \
        for (size_t i = 0; i < A->total_size; i++) F32(out)[i] = MATH_FUNC(F32(A)[i]); \
    } else { \
        int idx[8] = {0}; \
        for (size_t i = 0; i < out->total_size; i++) { \
            size_t offset = 0; \
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d]; \
            F32(out)[i] = MATH_FUNC(F32(A)[offset]); \
            for (int d = A->ndim - 1; d >= 0; d--) { \
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0; \
            } \
        } \
    } \
    return out; \
}

TENSOR_MATH_UNARY(tensor_sqrt, sqrtf)
TENSOR_MATH_UNARY(tensor_square, _square)
TENSOR_MATH_UNARY(tensor_abs, fabsf)
TENSOR_MATH_UNARY(tensor_sign, _sign)
TENSOR_MATH_UNARY(tensor_exp, expf)
TENSOR_MATH_UNARY(tensor_log, logf)
TENSOR_MATH_UNARY(tensor_log1p, log1pf)
TENSOR_MATH_UNARY(tensor_round, roundf)
TENSOR_MATH_UNARY(tensor_floor, floorf)
TENSOR_MATH_UNARY(tensor_ceil, ceilf)
TENSOR_MATH_UNARY(tensor_sigmoid, _sigmoid)
TENSOR_MATH_UNARY(tensor_tanh, tanhf)
TENSOR_MATH_UNARY(tensor_sin, sinf)
TENSOR_MATH_UNARY(tensor_cos, cosf)
TENSOR_MATH_UNARY(tensor_tan, tanf)
TENSOR_MATH_UNARY(tensor_asin, asinf)
TENSOR_MATH_UNARY(tensor_acos, acosf)
TENSOR_MATH_UNARY(tensor_atan, atanf)

Tensor* tensor_pow(Tensor* A, Tensor* B) {
    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    int out_ndim, out_shape[8];
    size_t stride_A[8] = {0}, stride_B[8] = {0};
    if (!tensor_broadcast_shapes(A, B, &out_ndim, out_shape, stride_A, stride_B)) {
        TENSOR_ERROR("FATAL: Shapes not broadcastable.");
    }
    Tensor* out = tensor_create_uninitialized(out_ndim, out_shape, DTYPE_FLOAT32);
    if (tensor_is_contiguous(A) && tensor_is_contiguous(B) && A->total_size == B->total_size) {
        _Pragma("omp simd")
        for (size_t i = 0; i < A->total_size; i++) F32(out)[i] = powf(F32(A)[i], F32(B)[i]);
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < out->total_size; i++) {
            size_t off_a = 0, off_b = 0;
            for (int d = 0; d < out_ndim; d++) {
                off_a += idx[d] * stride_A[d]; off_b += idx[d] * stride_B[d];
            }
            F32(out)[i] = powf(F32(A)[off_a], F32(B)[off_b]);
            for (int d = out_ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < out_shape[d]) break; idx[d] = 0;
            }
        }
    }
    return out;
}

Tensor* tensor_clip(Tensor* A, float min_val, float max_val) {
    if (A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32);
    if (tensor_is_contiguous(A)) {
        _Pragma("omp simd")
        for (size_t i = 0; i < A->total_size; i++) {
            float v = F32(A)[i];
            F32(out)[i] = (v < min_val) ? min_val : ((v > max_val) ? max_val : v);
        }
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < out->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            float v = F32(A)[offset];
            F32(out)[i] = (v < min_val) ? min_val : ((v > max_val) ? max_val : v);
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return out;
}

Tensor* tensor_relu(Tensor* A) { return tensor_clip(A, 0.0f, INFINITY); }

#define TENSOR_LOGICAL_OP(OP_NAME, OP) \
Tensor* OP_NAME(Tensor* A, Tensor* B) { \
    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32."); \
    int out_ndim, out_shape[8]; \
    size_t stride_A[8] = {0}, stride_B[8] = {0}; \
    if (!tensor_broadcast_shapes(A, B, &out_ndim, out_shape, stride_A, stride_B)) { \
        TENSOR_ERROR("FATAL: Shapes not broadcastable."); \
    } \
    Tensor* out = tensor_create_uninitialized(out_ndim, out_shape, DTYPE_FLOAT32); \
    int idx[8] = {0}; \
    for (size_t i = 0; i < out->total_size; i++) { \
        size_t off_a = 0, off_b = 0; \
        for (int d = 0; d < out_ndim; d++) { \
            off_a += idx[d] * stride_A[d]; off_b += idx[d] * stride_B[d]; \
        } \
        F32(out)[i] = (F32(A)[off_a] OP F32(B)[off_b]) ? 1.0f : 0.0f; \
        for (int d = out_ndim - 1; d >= 0; d--) { \
            idx[d]++; if (idx[d] < out_shape[d]) break; idx[d] = 0; \
        } \
    } \
    return out; \
}

TENSOR_LOGICAL_OP(tensor_equal, ==)
TENSOR_LOGICAL_OP(tensor_not_equal, !=)
TENSOR_LOGICAL_OP(tensor_greater, >)
TENSOR_LOGICAL_OP(tensor_greater_equal, >=)
TENSOR_LOGICAL_OP(tensor_less, <)
TENSOR_LOGICAL_OP(tensor_less_equal, <=)

Tensor* tensor_logical_not(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32);
    if (tensor_is_contiguous(A)) {
        for(size_t i = 0; i < A->total_size; i++) F32(out)[i] = (F32(A)[i] == 0.0f) ? 1.0f : 0.0f;
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < out->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            F32(out)[i] = (F32(A)[offset] == 0.0f) ? 1.0f : 0.0f;
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return out;
}

// --- NEW: SCALAR BOOLEAN LOGIC ---
Tensor* tensor_less_scalar_f32(Tensor* A, float val) {
    if (A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32);
    size_t n = A->total_size;
    if (tensor_is_contiguous(A)) {
#ifdef __AVX2__
        __m256 v_val = _mm256_set1_ps(val);
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 a_vec = _mm256_loadu_ps(F32(A) + i);
            __m256 cmp = _mm256_cmp_ps(a_vec, v_val, _CMP_LT_OQ);
            _mm256_storeu_ps(F32(out) + i, _mm256_and_ps(cmp, _mm256_set1_ps(1.0f)));
        }
        for (; i < n; i++) F32(out)[i] = F32(A)[i] < val ? 1.0f : 0.0f;
#else
        for (size_t i = 0; i < n; i++) F32(out)[i] = F32(A)[i] < val ? 1.0f : 0.0f;
#endif
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < n; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            F32(out)[i] = F32(A)[offset] < val ? 1.0f : 0.0f;
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return out;
}

Tensor* tensor_greater_scalar_f32(Tensor* A, float val) {
    if (A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32);
    size_t n = A->total_size;
    if (tensor_is_contiguous(A)) {
#ifdef __AVX2__
        __m256 v_val = _mm256_set1_ps(val);
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 a_vec = _mm256_loadu_ps(F32(A) + i);
            __m256 cmp = _mm256_cmp_ps(a_vec, v_val, _CMP_GT_OQ);
            _mm256_storeu_ps(F32(out) + i, _mm256_and_ps(cmp, _mm256_set1_ps(1.0f)));
        }
        for (; i < n; i++) F32(out)[i] = F32(A)[i] > val ? 1.0f : 0.0f;
#else
        for (size_t i = 0; i < n; i++) F32(out)[i] = F32(A)[i] > val ? 1.0f : 0.0f;
#endif
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < n; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            F32(out)[i] = F32(A)[offset] > val ? 1.0f : 0.0f;
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return out;
}

// ============================================================================
// 6. AXIS & GLOBAL AGGREGATIONS
// ============================================================================

#define TENSOR_AXIS_AGG(OP_NAME, INIT_VAL, OP, FINAL_OP) \
Tensor* OP_NAME(Tensor* A, int axis) { \
    if (A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("FATAL: Aggregation requires FLOAT32."); \
    if (axis < 0 || axis >= A->ndim) return NULL; \
    int out_shape[8]; int out_ndim = 0; \
    for (int i = 0; i < A->ndim; i++) if (i != axis) out_shape[out_ndim++] = A->shape[i]; \
    if (out_ndim == 0) { out_ndim = 1; out_shape[0] = 1; } \
    Tensor* out = tensor_create(out_ndim, out_shape); \
    tensor_fill(out, INIT_VAL); \
    int idx[8] = {0}; \
    for (size_t i = 0; i < A->total_size; i++) { \
        size_t offset_a = 0, offset_out = 0; int out_d = 0; \
        for (int d = 0; d < A->ndim; d++) { \
            offset_a += idx[d] * A->stride[d]; \
            if (d != axis) offset_out += idx[d] * out->stride[out_d++]; \
        } \
        F32(out)[offset_out] = OP(F32(out)[offset_out], F32(A)[offset_a]); \
        for (int d = A->ndim - 1; d >= 0; d--) { \
            idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0; \
        } \
    } \
    FINAL_OP; \
    return out; \
}

#define _ADD_OP(a, b) ((a) + (b))
#define _MAX_OP(a, b) ((a) > (b) ? (a) : (b))
#define _MIN_OP(a, b) ((a) < (b) ? (a) : (b))

TENSOR_AXIS_AGG(tensor_sum_axis, 0.0f, _ADD_OP, )
TENSOR_AXIS_AGG(tensor_mean_axis, 0.0f, _ADD_OP, for (size_t j=0; j<out->total_size; j++) F32(out)[j] /= A->shape[axis])
TENSOR_AXIS_AGG(tensor_max_axis, -INFINITY, _MAX_OP, )
TENSOR_AXIS_AGG(tensor_min_axis, INFINITY, _MIN_OP, )

/*
 * tensor_sum — two-tier strategy:
 *   ≥ 500 K elements : OpenMP parallel reduction (auto-vectorized by -O3 -mavx2)
 *   < 500 K elements : 4-accumulator AVX2 loop (hides latency without thread overhead)
 */
float tensor_sum(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) return 0.0f;
    float sum = 0.0f;
    if (tensor_is_contiguous(A)) {
        const float* data = F32(A);
        size_t n = A->total_size;

        if (n >= 500000) {
            /* Large tensors: let OpenMP+compiler vectorise for us */
            #pragma omp parallel for simd reduction(+:sum) schedule(static)
            for (size_t i = 0; i < n; i++) sum += data[i];
        } else {
#ifdef __AVX2__
            /* Medium tensors: 4-accumulator unrolled AVX2 */
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 31 < n; i += 32) {
                __builtin_prefetch(data + i + 128, 0, 1);
                a0 = _mm256_add_ps(a0, _mm256_loadu_ps(data + i));
                a1 = _mm256_add_ps(a1, _mm256_loadu_ps(data + i +  8));
                a2 = _mm256_add_ps(a2, _mm256_loadu_ps(data + i + 16));
                a3 = _mm256_add_ps(a3, _mm256_loadu_ps(data + i + 24));
            }
            a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            __m128 lo = _mm256_castps256_ps128(a0);
            __m128 hi = _mm256_extractf128_ps(a0, 1);
            __m128 s  = _mm_add_ps(lo, hi);
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            sum = _mm_cvtss_f32(s);
            for (; i < n; i++) sum += data[i];
#else
            for (size_t i = 0; i < n; i++) sum += data[i];
#endif
        }
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < A->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            sum += F32(A)[offset];
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return sum;
}

float tensor_mean(Tensor* A) { return tensor_sum(A) / (float)A->total_size; }

float tensor_product(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) return 0.0f;
    float prod = 1.0f;
    if (tensor_is_contiguous(A)) {
        for(size_t i=0; i<A->total_size; i++) prod *= F32(A)[i];
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < A->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            prod *= F32(A)[offset];
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return prod;
}

// ============================================================================
// SLICING WITH STEPS
// ============================================================================

Tensor* tensor_slice_step(Tensor* t, int axis, int start, int end, int step) {
    if (axis < 0 || axis >= t->ndim) return NULL;
    if (step <= 0) TENSOR_ERROR("FATAL: Step must be > 0.");
    if (start < 0) start = 0;
    if (end > t->shape[axis]) end = t->shape[axis];
    if (start >= end) return NULL;

    int length = (end - start + step - 1) / step; 
    Tensor* v = tensor_view(t);
    v->shape[axis] = length;
    v->stride[axis] = t->stride[axis] * step;
    v->total_size = 1;
    for (int i = 0; i < v->ndim; i++) v->total_size *= v->shape[i];
    v->byte_size = v->total_size * dtype_size(t->dtype);
    v->data = (char*)t->data + (start * t->stride[axis] * dtype_size(t->dtype));
    return v;
}

// ============================================================================
// ZERO-COPY AXIS SWAPPING
// ============================================================================

Tensor* tensor_swapaxes(Tensor* A, int axis1, int axis2) {
    if (axis1 < 0 || axis1 >= A->ndim || axis2 < 0 || axis2 >= A->ndim) {
        TENSOR_ERROR("FATAL [SwapAxes]: Invalid axes.");
    }
    Tensor* v = tensor_view(A);
    int tmp_shape = v->shape[axis1];
    v->shape[axis1] = v->shape[axis2];
    v->shape[axis2] = tmp_shape;
    size_t tmp_stride = v->stride[axis1];
    v->stride[axis1] = v->stride[axis2];
    v->stride[axis2] = tmp_stride;
    return v;
}

// ============================================================================
// CUMULATIVE OPERATIONS
// ============================================================================

Tensor* tensor_cumsum_axis(Tensor* A, int axis) {
    if (axis < 0 || axis >= A->ndim || A->dtype != DTYPE_FLOAT32) return NULL;
    Tensor* out = tensor_create(A->ndim, A->shape);

    int dim_len = A->shape[axis];
    int out_ndim = A->ndim - 1;
    int out_shape[8];
    size_t stride_a[8], stride_out[8];
    
    for (int i = 0, j = 0; i < A->ndim; i++) {
        if (i != axis) {
            out_shape[j] = A->shape[i];
            stride_a[j] = A->stride[i];
            stride_out[j] = out->stride[i];
            j++;
        }
    }

    size_t num_vectors = 1;
    for (int i = 0; i < out_ndim; i++) num_vectors *= out_shape[i];

    int idx[8] = {0};
    for (size_t v = 0; v < num_vectors; v++) {
        size_t base_a = 0, base_out = 0;
        for (int d = 0; d < out_ndim; d++) {
            base_a += idx[d] * stride_a[d]; base_out += idx[d] * stride_out[d];
        }

        float running_sum = 0.0f;
        for (int i = 0; i < dim_len; i++) {
            running_sum += F32(A)[base_a + i * A->stride[axis]];
            F32(out)[base_out + i * out->stride[axis]] = running_sum;
        }

        if (out_ndim > 0) {
            for (int d = out_ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < out_shape[d]) break; idx[d] = 0;
            }
        }
    }
    return out;
}

// ============================================================================
// SORTING AND ARGSORT
// ============================================================================

typedef struct { float val; int idx; } FloatIndex;

static int cmp_float_index(const void* a, const void* b) {
    float va = ((FloatIndex*)a)->val;
    float vb = ((FloatIndex*)b)->val;
    return (va > vb) - (va < vb);
}

Tensor* tensor_argsort(Tensor* A, int axis) {
    if (axis < 0 || axis >= A->ndim || A->dtype != DTYPE_FLOAT32) return NULL;
    // Output is fully overwritten with index values — skip zero-fill.
    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32);
    int dim_len = A->shape[axis];

    FloatIndex* buffer = (FloatIndex*)malloc(dim_len * sizeof(FloatIndex));
    if (!buffer) { tensor_free(out); TENSOR_ERROR("FATAL: malloc failed in argsort."); }

    int out_ndim = A->ndim - 1;
    int out_shape[8]; size_t stride_a[8], stride_out[8];
    
    for (int i = 0, k = 0; i < A->ndim; i++) {
        if (i != axis) {
            out_shape[k] = A->shape[i]; stride_a[k] = A->stride[i]; stride_out[k] = out->stride[i]; k++;
        }
    }

    size_t num_vectors = 1;
    for (int i = 0; i < out_ndim; i++) num_vectors *= out_shape[i];

    int idx[8] = {0};
    for (size_t v = 0; v < num_vectors; v++) {
        size_t base_a = 0, base_out = 0;
        for (int d = 0; d < out_ndim; d++) {
            base_a += idx[d] * stride_a[d]; base_out += idx[d] * stride_out[d];
        }

        for (int i = 0; i < dim_len; i++) {
            buffer[i].val = F32(A)[base_a + i * A->stride[axis]];
            buffer[i].idx = i;
        }

        qsort(buffer, dim_len, sizeof(FloatIndex), cmp_float_index);

        for (int i = 0; i < dim_len; i++) {
            F32(out)[base_out + i * out->stride[axis]] = (float)buffer[i].idx;
        }

        if (out_ndim > 0) {
            for (int d = out_ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < out_shape[d]) break; idx[d] = 0;
            }
        }
    }
    free(buffer);
    return out;
}

/* AVX2 horizontal min/max helper: reduce __m256 to scalar */
#ifdef __AVX2__
static inline float _hmin256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m4 = _mm_min_ps(lo, hi);
    m4 = _mm_min_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_min_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    return _mm_cvtss_f32(m4);
}
static inline float _hmax256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m4 = _mm_max_ps(lo, hi);
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    return _mm_cvtss_f32(m4);
}
#endif

float tensor_min(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) return 0.0f;
    if (tensor_is_contiguous(A)) {
        const float* data = F32(A);
        size_t n = A->total_size;
        float min_val = INFINITY;
        size_t i = 0;
#ifdef __AVX2__
        if (n >= 8) {
            __m256 vmin = _mm256_set1_ps(INFINITY);
            size_t limit = n & ~7UL;
            for (; i < limit; i += 8) {
                __builtin_prefetch(data + i + 64, 0, 1);
                vmin = _mm256_min_ps(vmin, _mm256_loadu_ps(data + i));
            }
            min_val = _hmin256(vmin);
        }
#endif
        for (; i < n; i++) if (data[i] < min_val) min_val = data[i];
        return min_val;
    } else {
        float min_val = INFINITY;
        int idx[8] = {0};
        for (size_t i = 0; i < A->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            if (F32(A)[offset] < min_val) min_val = F32(A)[offset];
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
        return min_val;
    }
}

float tensor_max(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) return 0.0f;
    if (tensor_is_contiguous(A)) {
        const float* data = F32(A);
        size_t n = A->total_size;
        float max_val = -INFINITY;
        size_t i = 0;
#ifdef __AVX2__
        if (n >= 8) {
            __m256 vmax = _mm256_set1_ps(-INFINITY);
            size_t limit = n & ~7UL;
            for (; i < limit; i += 8) {
                __builtin_prefetch(data + i + 64, 0, 1);
                vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(data + i));
            }
            max_val = _hmax256(vmax);
        }
#endif
        for (; i < n; i++) if (data[i] > max_val) max_val = data[i];
        return max_val;
    } else {
        float max_val = -INFINITY;
        int idx[8] = {0};
        for (size_t i = 0; i < A->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            if (F32(A)[offset] > max_val) max_val = F32(A)[offset];
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
        return max_val;
    }
}

int tensor_argmin(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) return 0;
    float min_val = INFINITY; int global_idx = 0, best_idx = 0;
    int idx[8] = {0};
    for (size_t i = 0; i < A->total_size; i++) {
        size_t offset = 0;
        for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
        if(F32(A)[offset] < min_val) { min_val = F32(A)[offset]; best_idx = global_idx; }
        global_idx++;
        for (int d = A->ndim - 1; d >= 0; d--) {
            idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
        }
    }
    return best_idx;
}

int tensor_argmax(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) return 0;
    float max_val = -INFINITY; int global_idx = 0, best_idx = 0;
    int idx[8] = {0};
    for (size_t i = 0; i < A->total_size; i++) {
        size_t offset = 0;
        for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
        if(F32(A)[offset] > max_val) { max_val = F32(A)[offset]; best_idx = global_idx; }
        global_idx++;
        for (int d = A->ndim - 1; d >= 0; d--) {
            idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
        }
    }
    return best_idx;
}

// Welford's online algorithm for precision and safety without 2x memory read
float tensor_variance(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) return 0.0f;
    double mean = 0.0, M2 = 0.0;
    size_t count = 0;
    
    if (tensor_is_contiguous(A)) {
        for(size_t i=0; i<A->total_size; i++) {
            double x = (double)F32(A)[i];
            count++;
            double delta = x - mean;
            mean += delta / count;
            M2 += delta * (x - mean);
        }
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < A->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            double x = (double)F32(A)[offset];
            count++;
            double delta = x - mean;
            mean += delta / count;
            M2 += delta * (x - mean);
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return (count > 0) ? (float)(M2 / count) : 0.0f;
}

float tensor_std(Tensor* A) { return sqrtf(tensor_variance(A)); }

static int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

float tensor_median(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) return 0.0f;
    Tensor* temp = tensor_copy(A); 
    qsort(temp->data, temp->total_size, sizeof(float), cmp_float);
    
    size_t n = temp->total_size;
    float med;
    if (n % 2 == 0 && n > 0) {
        med = (F32(temp)[n/2 - 1] + F32(temp)[n/2]) / 2.0f;
    } else {
        med = F32(temp)[n/2];
    }
    
    tensor_free(temp);
    return med;
}

// ============================================================================
// 7. PREPROCESSING / NORMALIZATION
// ============================================================================

void tensor_standardize_inplace(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) return;
    float mean = tensor_mean(A);
    float std = tensor_std(A);
    float safe_std = std > 1e-8f ? std : 1e-8f;
    tensor_add_scalar_inplace(A, -mean);
    tensor_mul_scalar_inplace(A, 1.0f / safe_std);
}

Tensor* tensor_standardize(Tensor* A) {
    Tensor* out = tensor_copy(A);
    tensor_standardize_inplace(out);
    return out;
}

void tensor_normalize_inplace(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) return;
    float min_v = tensor_min(A);
    float max_v = tensor_max(A);
    float range = (max_v - min_v) > 1e-8f ? (max_v - min_v) : 1e-8f;
    tensor_add_scalar_inplace(A, -min_v);
    tensor_mul_scalar_inplace(A, 1.0f / range);
}

Tensor* tensor_normalize(Tensor* A) {
    Tensor* out = tensor_copy(A);
    tensor_normalize_inplace(out);
    return out;
}

// ============================================================================
// 8. CONCATENATION & MASKING
// ============================================================================

Tensor* tensor_concat(Tensor** tensors, int num_tensors, int axis) {
    if (num_tensors == 0 || axis < 0) return NULL;
    int ndim = tensors[0]->ndim;
    TensorDType dtype = tensors[0]->dtype;

    if (axis >= ndim) return NULL;
    int out_shape[8];
    for (int i = 0; i < ndim; i++) out_shape[i] = tensors[0]->shape[i];
    for (int i = 1; i < num_tensors; i++) {
        if (tensors[i]->ndim != ndim || tensors[i]->dtype != dtype) TENSOR_ERROR("FATAL: Concat shape mismatch.");
        for (int j = 0; j < ndim; j++) {
            if (j != axis && tensors[i]->shape[j] != out_shape[j]) TENSOR_ERROR("FATAL: Concat shape mismatch.");
        }
        out_shape[axis] += tensors[i]->shape[axis];
    }
    
    Tensor* out = tensor_create_dtype(ndim, out_shape, dtype);
    size_t el_size = dtype_size(dtype);
    
    // Fast path: axis=0 concatenation of contiguous tensors
    bool all_contiguous = true;
    for (int i = 0; i < num_tensors; i++) {
        if (!tensor_is_contiguous(tensors[i])) { all_contiguous = false; break; }
    }
    if (all_contiguous && axis == 0) {
        char* dst = (char*)out->data;
        for (int t = 0; t < num_tensors; t++) {
            memcpy(dst, tensors[t]->data, tensors[t]->byte_size);
            dst += tensors[t]->byte_size;
        }
        return out;
    }
    
    int current_axis_offset = 0;
    for (int t = 0; t < num_tensors; t++) {
        Tensor* in = tensors[t];
        int idx[8] = {0};
        for (size_t i = 0; i < in->total_size; i++) {
            size_t offset_in = 0, offset_out = 0;
            for (int d = 0; d < ndim; d++) {
                offset_in += idx[d] * in->stride[d];
                int mapped_idx = (d == axis) ? idx[d] + current_axis_offset : idx[d];
                offset_out += mapped_idx * out->stride[d];
            }
            memcpy((char*)out->data + offset_out * el_size, (char*)in->data + offset_in * el_size, el_size);
            
            for (int d = ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < in->shape[d]) break; idx[d] = 0;
            }
        }
        current_axis_offset += in->shape[axis];
    }
    return out;
}

Tensor* tensor_where(Tensor* condition, Tensor* x, Tensor* y) {
    if (!_tensor_shape_assert(condition, x, "tensor_where") || !_tensor_shape_assert(condition, y, "tensor_where")) return NULL;
    if (x->dtype != DTYPE_FLOAT32 || y->dtype != DTYPE_FLOAT32 || condition->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    
    Tensor* out = tensor_create_uninitialized(x->ndim, x->shape, DTYPE_FLOAT32);
    if (tensor_is_contiguous(condition) && tensor_is_contiguous(x) && tensor_is_contiguous(y)) {
        size_t i = 0;
#ifdef __AVX2__
        for (; i + 7 < out->total_size; i += 8) {
            __m256 cond = _mm256_loadu_ps(&F32(condition)[i]);
            __m256 x_val = _mm256_loadu_ps(&F32(x)[i]);
            __m256 y_val = _mm256_loadu_ps(&F32(y)[i]);
            __m256 threshold = _mm256_set1_ps(0.5f);
            __m256 mask = _mm256_cmp_ps(cond, threshold, _CMP_GT_OQ);
            __m256 blended = _mm256_blendv_ps(y_val, x_val, mask);
            _mm256_storeu_ps(&F32(out)[i], blended);
        }
#endif
        for (; i < out->total_size; i++) F32(out)[i] = (F32(condition)[i] > 0.5f) ? F32(x)[i] : F32(y)[i];
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < out->total_size; i++) {
            size_t off_cond = 0, off_x = 0, off_y = 0;
            for (int d = 0; d < out->ndim; d++) {
                off_cond += idx[d] * condition->stride[d];
                off_x += idx[d] * x->stride[d];
                off_y += idx[d] * y->stride[d];
            }
            F32(out)[i] = (F32(condition)[off_cond] > 0.5f) ? F32(x)[off_x] : F32(y)[off_y];
            for (int d = out->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < out->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return out;
}

// ============================================================================
// BOOLEAN (ADVANCED) INDEXING
// ============================================================================

Tensor* tensor_boolean_index(Tensor* A, Tensor* mask) {
    if (!_tensor_shape_assert(A, mask, "tensor_boolean_index")) return NULL;
    if (A->dtype != DTYPE_FLOAT32 || mask->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    
    int count = 0;
    if (tensor_is_contiguous(mask)) {
        for (size_t i = 0; i < mask->total_size; i++) {
            if (F32(mask)[i] > 0.5f) count++;
        }
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < mask->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < mask->ndim; d++) offset += idx[d] * mask->stride[d];
            if (F32(mask)[offset] > 0.5f) count++;
            for (int d = mask->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < mask->shape[d]) break; idx[d] = 0;
            }
        }
    }

    if (count == 0) return tensor_create(1, (int[]){0});

    Tensor* out = tensor_create(1, &count);
    size_t out_idx = 0;

    if (tensor_is_contiguous(A) && tensor_is_contiguous(mask)) {
        for (size_t i = 0; i < A->total_size; i++) {
            if (F32(mask)[i] > 0.5f) F32(out)[out_idx++] = F32(A)[i];
        }
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < A->total_size; i++) {
            size_t off_a = 0, off_m = 0;
            for (int d = 0; d < A->ndim; d++) {
                off_a += idx[d] * A->stride[d];
                off_m += idx[d] * mask->stride[d];
            }
            if (F32(mask)[off_m] > 0.5f) F32(out)[out_idx++] = F32(A)[off_a];
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return out;
}

// ============================================================================
// 9. LINEAR ALGEBRA & ADVANCED DECOMPOSITIONS
// ============================================================================
float tensor_dot(Tensor* A, Tensor* B) {
    if(A->ndim != 1 || B->ndim != 1 || A->total_size != B->total_size || A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32) {
        TENSOR_ERROR_VAL(0.0f, "FATAL [Dot]: Must be 1D FLOAT32 tensors of identical length.");
    }
    return cblas_sdot(A->total_size, F32(A), A->stride[0], F32(B), B->stride[0]);
}

float tensor_trace(Tensor* A) {
    if(A->ndim != 2 || A->shape[0] != A->shape[1] || A->dtype != DTYPE_FLOAT32) {
        TENSOR_ERROR_VAL(0.0f, "FATAL [Trace]: Must be 2D square matrix FLOAT32.");
    }
    float tr = 0.0f;
    for(int i=0; i<A->shape[0]; i++) {
        tr += F32(A)[i * A->stride[0] + i * A->stride[1]];
    }
    return tr;
}

Tensor* tensor_ref(Tensor* A) {
    if (A->ndim != 2 || A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("FATAL [REF]: Must be 2D FLOAT32.");
    Tensor* out = tensor_copy(A); 
    int m = out->shape[0]; int n = out->shape[1]; int lead = 0;
    
    for (int r = 0; r < m; r++) {
        if (n <= lead) break;
        int i = r;
        while (fabsf(F32(out)[i * n + lead]) < 1e-6f) {
            i++;
            if (m == i) { i = r; lead++; if (n == lead) goto end_ref; }
        }
        if (i != r) { 
            for (int j = 0; j < n; j++) {
                float tmp = F32(out)[i * n + j];
                F32(out)[i * n + j] = F32(out)[r * n + j];
                F32(out)[r * n + j] = tmp;
            }
        }
        for (int i = r + 1; i < m; i++) {
            float ratio = F32(out)[i * n + lead] / F32(out)[r * n + lead];
            for (int j = lead; j < n; j++) F32(out)[i * n + j] -= ratio * F32(out)[r * n + j];
            F32(out)[i * n + lead] = 0.0f; 
        }
        lead++;
    }
end_ref:
    return out;
}

Tensor* tensor_rref(Tensor* A) {
    if (A->ndim != 2 || A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("FATAL [RREF]: Must be 2D FLOAT32.");
    Tensor* out = tensor_copy(A); 
    int m = out->shape[0]; int n = out->shape[1]; int lead = 0;
    
    for (int r = 0; r < m; r++) {
        if (n <= lead) break;
        int i = r;
        while (fabsf(F32(out)[i * n + lead]) < 1e-6f) {
            i++;
            if (m == i) { i = r; lead++; if (n == lead) goto end_rref; }
        }
        if (i != r) { 
            for (int j = 0; j < n; j++) {
                float tmp = F32(out)[i * n + j];
                F32(out)[i * n + j] = F32(out)[r * n + j];
                F32(out)[r * n + j] = tmp;
            }
        }
        float val = F32(out)[r * n + lead];
        if (fabsf(val) > 1e-6f) {
            float inv = 1.0f / val;
            for (int j = 0; j < n; j++) F32(out)[r * n + j] *= inv;
        }
        for (int i = 0; i < m; i++) {
            if (i != r) {
                float val2 = F32(out)[i * n + lead];
                for (int j = 0; j < n; j++) F32(out)[i * n + j] -= val2 * F32(out)[r * n + j];
            }
        }
        lead++;
    }
end_rref:
    return out;
}

Tensor* tensor_matmul(Tensor* A, Tensor* B) {
    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32) TENSOR_ERROR("FATAL [Matmul]: Requires FLOAT32.");
    if (A->ndim != 2 || B->ndim != 2 || A->shape[1] != B->shape[0]) TENSOR_ERROR("FATAL [Matmul]: Inner dimensions do not match.");

    int m = A->shape[0];
    int k = A->shape[1];
    int n = B->shape[1];

    // ----------------------------------------------------------------
    // GEMV fast-path #1: B is a column vector [k, 1]  →  sgemv beats sgemm
    // ----------------------------------------------------------------
    if (n == 1) {
        Tensor* a_c = tensor_is_contiguous(A) ? A : tensor_copy(A);
        Tensor* b_c = tensor_is_contiguous(B) ? B : tensor_copy(B);
        Tensor* out = tensor_create_uninitialized(2, (int[]){m, 1}, DTYPE_FLOAT32);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    m, k, 1.0f,
                    F32(a_c), k,    /* lda = k for row-major [m,k] */
                    F32(b_c), 1,    /* incx = 1 for contiguous [k,1] */
                    0.0f, F32(out), 1);
        if (a_c != A) tensor_free(a_c);
        if (b_c != B) tensor_free(b_c);
        return out;
    }

    // ----------------------------------------------------------------
    // GEMV fast-path #2: A is a row vector [1, k]  →  y = B^T * x
    // ----------------------------------------------------------------
    if (m == 1) {
        Tensor* a_c = tensor_is_contiguous(A) ? A : tensor_copy(A);
        Tensor* b_c = tensor_is_contiguous(B) ? B : tensor_copy(B);
        Tensor* out = tensor_create_uninitialized(2, (int[]){1, n}, DTYPE_FLOAT32);
        /* sgemv(Trans, rows=k, cols=n): out[j] = sum_i B[i,j]*x[i]  ==  x*B */
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    k, n, 1.0f,
                    F32(b_c), n,
                    F32(a_c), 1,
                    0.0f, F32(out), 1);
        if (a_c != A) tensor_free(a_c);
        if (b_c != B) tensor_free(b_c);
        return out;
    }

    // ----------------------------------------------------------------
    // General SGEMM path with transpose / non-contiguous detection
    // ----------------------------------------------------------------
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    int lda = k;
    int ldb = n;

    Tensor* a_work = A;
    Tensor* b_work = B;

    if (A->stride[0] == 1 && A->stride[1] == (size_t)A->shape[0]) {
        transA = CblasTrans;
        lda = A->shape[0];
    } else if (!tensor_is_contiguous(A)) {
        if (A->stride[0] == (size_t)A->shape[1] && A->stride[1] == 1) {
            lda = A->shape[1];
        } else {
            a_work = tensor_copy(A);
            lda = a_work->shape[1];
        }
    } else { lda = A->shape[1]; }

    if (B->stride[0] == 1 && B->stride[1] == (size_t)B->shape[0]) {
        transB = CblasTrans;
        ldb = B->shape[0];
    } else if (!tensor_is_contiguous(B)) {
        if (B->stride[0] == (size_t)B->shape[1] && B->stride[1] == 1) {
            ldb = B->shape[1];
        } else {
            b_work = tensor_copy(B);
            ldb = b_work->shape[1];
        }
    } else { ldb = B->shape[1]; }

    /* beta=0.0 → BLAS overwrites every output element; no zero-fill needed. */
    Tensor* out = tensor_create_uninitialized(2, (int[]){m, n}, DTYPE_FLOAT32);

    cblas_sgemm(CblasRowMajor, transA, transB,
                m, n, k, 1.0f,
                F32(a_work), lda,
                F32(b_work), ldb,
                0.0f, F32(out), n);

    if (a_work != A) tensor_free(a_work);
    if (b_work != B) tensor_free(b_work);
    return out;
}

/*
 * tensor_bmm — batched matrix multiply: A[batch,m,k] × B[batch,k,n] → [batch,m,n]
 *
 * Optimization: for contiguous inputs we bypass all per-batch view/reshape/slice
 * allocations and call cblas_sgemm directly with pointer arithmetic.  5 heap
 * objects per batch iteration → 0.  For batch > 1 and large slices the batch
 * loop is parallelised with OpenMP; BLAS is kept single-threaded in that case
 * to avoid oversubscription.
 */
extern void openblas_set_num_threads(int) __attribute__((weak));

Tensor* tensor_bmm(Tensor* A, Tensor* B) {
    if (A->ndim != 3 || B->ndim != 3 ||
        A->shape[0] != B->shape[0] ||
        A->shape[2] != B->shape[1] ||
        A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32) {
        TENSOR_ERROR("FATAL [BMM]: Invalid dimensions or type.");
    }

    int batch = A->shape[0];
    int m     = A->shape[1];
    int k     = A->shape[2];
    int n     = B->shape[2];

    /* Each slice is fully overwritten by sgemm with beta=0 — skip zero-fill. */
    Tensor* out = tensor_create_uninitialized(3, (int[]){batch, m, n}, DTYPE_FLOAT32);

    /* Compact non-contiguous inputs once (O(N) copy), then use direct offsets. */
    Tensor* a_work = tensor_is_contiguous(A) ? A : tensor_copy(A);
    Tensor* b_work = tensor_is_contiguous(B) ? B : tensor_copy(B);

    bool parallel = (batch > 1) && ((size_t)m * n * k > 10000);

    if (parallel && openblas_set_num_threads)
        openblas_set_num_threads(1); /* prevent BLAS×OMP oversubscription */

    #pragma omp parallel for schedule(dynamic) if(parallel)
    for (int b = 0; b < batch; b++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0f,
                    F32(a_work) + b * m * k, k,
                    F32(b_work) + b * k * n, n,
                    0.0f,
                    F32(out)    + b * m * n, n);
    }

    if (parallel && openblas_set_num_threads)
        openblas_set_num_threads(omp_get_max_threads());

    if (a_work != A) tensor_free(a_work);
    if (b_work != B) tensor_free(b_work);
    return out;
}

Tensor* tensor_inverse(Tensor* A) {
    if(A->ndim != 2 || A->shape[0] != A->shape[1] || A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("FATAL [Inverse]: Must be 2D square FLOAT32.");
    int n = A->shape[0];
    Tensor* out = tensor_copy(A); 
    int* ipiv = (int*)safe_malloc(n * sizeof(int));
    if (!ipiv) { tensor_free(out); TENSOR_ERROR("Malloc failed."); }
    
    int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, F32(out), n, ipiv);
    if (info != 0) {
        free(ipiv); tensor_free(out);
        TENSOR_ERROR("Matrix is singular (sgetrf info=%d).", info);
    }
    info = LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, F32(out), n, ipiv);
    if (info != 0) {
        free(ipiv); tensor_free(out);
        TENSOR_ERROR("Inversion failed (sgetri info=%d).", info);
    }
    free(ipiv);
    return out;
}

// ============================================================================
// 8. ADVANCED DECOMPOSITIONS & REDUCTIONS
// ============================================================================

Tensor* tensor_cholesky(Tensor* A) {
    if(A->ndim != 2 || A->shape[0] != A->shape[1] || A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("FATAL [Cholesky]: Must be 2D square FLOAT32.");
    int n = A->shape[0];
    Tensor* out = tensor_copy(A);
    int info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', n, F32(out), n);
    if (info != 0) { tensor_free(out); TENSOR_ERROR("FATAL [Cholesky]: Matrix is not positive definite."); }
    for(int i = 0; i < n; i++) for(int j = i + 1; j < n; j++) F32(out)[i * n + j] = 0.0f;
    return out;
}

void tensor_lu(Tensor* A, Tensor** P_out, Tensor** L_out, Tensor** U_out) {
    if(A->ndim != 2 || A->dtype != DTYPE_FLOAT32) TENSOR_ERROR_VOID("FATAL [LU]: Must be 2D FLOAT32.");
    int m = A->shape[0]; int n = A->shape[1]; int min_mn = MIN(m, n);
    Tensor* LU = tensor_copy(A); 
    int* ipiv = (int*)safe_malloc(min_mn * sizeof(int));
    if (!ipiv) { tensor_free(LU); TENSOR_ERROR_VOID("Malloc failed."); }
    int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, m, n, F32(LU), n, ipiv);
    if (info < 0) { free(ipiv); tensor_free(LU); TENSOR_ERROR_VOID("FATAL [LU]: Illegal value in LAPACKE."); }
    
    Tensor* L = tensor_zeros(2, (int[]){m, min_mn});
    Tensor* U = tensor_zeros(2, (int[]){min_mn, n});
    Tensor* P = tensor_zeros(2, (int[]){m, m});
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i > j) { if (j < min_mn) F32(L)[i * min_mn + j] = F32(LU)[i * n + j]; } 
            else if (i == j) {
                if (j < min_mn) F32(L)[i * min_mn + j] = 1.0f;
                if (i < min_mn) F32(U)[i * n + j] = F32(LU)[i * n + j];
            } else { if (i < min_mn) F32(U)[i * n + j] = F32(LU)[i * n + j]; }
        }
    }
    
    for (int i = 0; i < m; i++) F32(P)[i * m + i] = 1.0f;
    for (int i = 0; i < min_mn; i++) {
        int ip = ipiv[i] - 1; 
        if (ip != i) {
            for (int j = 0; j < m; j++) {
                float tmp = F32(P)[i * m + j];
                F32(P)[i * m + j] = F32(P)[ip * m + j];
                F32(P)[ip * m + j] = tmp;
            }
        }
    }
    free(ipiv); tensor_free(LU);
    if(P_out) *P_out = P; else tensor_free(P);
    if(L_out) *L_out = L; else tensor_free(L);
    if(U_out) *U_out = U; else tensor_free(U);
}

void tensor_svd(Tensor* A, Tensor** U_out, Tensor** S_out, Tensor** Vt_out) {
    if(A->ndim != 2 || A->dtype != DTYPE_FLOAT32) TENSOR_ERROR_VOID("FATAL [SVD]: Must be 2D FLOAT32.");
    int m = A->shape[0]; int n = A->shape[1]; int min_mn = MIN(m, n);
    Tensor* S = tensor_create(1, &min_mn);
    Tensor* U = tensor_create(2, (int[]){m, m});
    Tensor* Vt = tensor_create(2, (int[]){n, n});
    Tensor* copyA = tensor_copy(A); 
    
    int info = LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'A', m, n, F32(copyA), n, F32(S), F32(U), m, F32(Vt), n);
    if (info != 0) {
        tensor_free(S); tensor_free(U); tensor_free(Vt); tensor_free(copyA);
        TENSOR_ERROR_VOID("FATAL [SVD]: SVD convergence failed.");
    }
    tensor_free(copyA);
    
    if(U_out) *U_out = U; else tensor_free(U);
    if(S_out) *S_out = S; else tensor_free(S);
    if(Vt_out) *Vt_out = Vt; else tensor_free(Vt);
}

void tensor_eigen_sym(Tensor* A, Tensor** EigenVals_out, Tensor** EigenVecs_out) {
    if(A->ndim != 2 || A->shape[0] != A->shape[1] || A->dtype != DTYPE_FLOAT32) TENSOR_ERROR_VOID("Requires symmetric 2D FLOAT32.");
    int n = A->shape[0];
    Tensor* W = tensor_create(1, &n);
    Tensor* V = tensor_copy(A); 
    
    int info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', n, F32(V), n, F32(W));
    if (info != 0) { 
        tensor_free(W); tensor_free(V); 
        TENSOR_ERROR_VOID("FATAL [Eigen]: Computation failed."); 
    }
    
    if(EigenVals_out) *EigenVals_out = W; else tensor_free(W);
    if(EigenVecs_out) *EigenVecs_out = V; else tensor_free(V);
}

// ============================================================================
// SET OPERATIONS
// ============================================================================

static int cmp_float_asc(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

Tensor* tensor_unique(Tensor* A) {
    if(A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    // Fix: Never sort a view of the original data.
    Tensor* temp = tensor_copy(A); 
    temp->ndim = 1;
    temp->shape[0] = (int)temp->total_size;
    _compute_strides(temp);
    
    if (temp->total_size == 0) return temp;
    
    qsort(temp->data, temp->total_size, sizeof(float), cmp_float_asc);
    
    int unique_count = 1;
    for (size_t i = 1; i < temp->total_size; i++) {
        if (fabsf(F32(temp)[i] - F32(temp)[i - 1]) > 1e-6f) unique_count++;
    }
    
    Tensor* out = tensor_create_uninitialized(1, &unique_count, DTYPE_FLOAT32);
    F32(out)[0] = F32(temp)[0];
    int idx = 1;
    for (size_t i = 1; i < temp->total_size; i++) {
        if (fabsf(F32(temp)[i] - F32(temp)[i - 1]) > 1e-6f) {
            F32(out)[idx++] = F32(temp)[i];
        }
    }
    
    tensor_free(temp);
    return out;
}

Tensor* tensor_bincount(Tensor* A) {
    if(A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");

    // First pass: validate all values and find the max.
    // This catches mixed negative/positive arrays that tensor_max would miss.
    float max_f = -INFINITY;
    if (tensor_is_contiguous(A)) {
        for (size_t i = 0; i < A->total_size; i++) {
            float f = F32(A)[i];
            if (fabsf(f - roundf(f)) > 1e-5f) TENSOR_ERROR("Non-integer value %.6f found in bincount.", f);
            int val = (int)roundf(f);
            if (val < 0) TENSOR_ERROR("FATAL [Bincount]: Negative value %d at index %zu.", val, i);
            if (f > max_f) max_f = f;
        }
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < A->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            float f = F32(A)[offset];
            if (fabsf(f - roundf(f)) > 1e-5f) TENSOR_ERROR("Non-integer value %.6f found in bincount.", f);
            int val = (int)roundf(f);
            if (val < 0) TENSOR_ERROR("FATAL [Bincount]: Negative value %d.", val);
            if (f > max_f) max_f = f;
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }

    int max_val = (max_f == -INFINITY) ? 0 : (int)roundf(max_f);
    int size = max_val + 1;
    Tensor* out = tensor_zeros(1, &size);

    // Second pass: accumulate counts (all values validated above).
    if (tensor_is_contiguous(A)) {
        for (size_t i = 0; i < A->total_size; i++)
            F32(out)[(int)roundf(F32(A)[i])] += 1.0f;
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < A->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            F32(out)[(int)roundf(F32(A)[offset])] += 1.0f;
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return out;
}

// ============================================================================
// FANCY INDEXING & PADDING
// ============================================================================

Tensor* tensor_take(Tensor* A, Tensor* indices, int axis) {
    if (axis < 0 || axis >= A->ndim || indices->ndim != 1) return NULL;
    if (A->dtype != DTYPE_FLOAT32 || indices->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    
    int out_shape[8];
    for (int i = 0; i < A->ndim; i++) out_shape[i] = A->shape[i];
    out_shape[axis] = indices->total_size;
    
    Tensor* out = tensor_create_uninitialized(A->ndim, out_shape, DTYPE_FLOAT32);
    int idx[8] = {0};
    
    for (size_t i = 0; i < out->total_size; i++) {
        size_t off_out = 0, off_a = 0;
        
        size_t index_offset = idx[axis] * indices->stride[0];
        int target_idx = (int)F32(indices)[index_offset];
        if (target_idx < 0 || target_idx >= A->shape[axis]) target_idx = 0; 
        
        for (int d = 0; d < A->ndim; d++) {
            off_out += idx[d] * out->stride[d];
            off_a += (d == axis ? target_idx : idx[d]) * A->stride[d];
        }
        
        F32(out)[off_out] = F32(A)[off_a];
        
        for (int d = A->ndim - 1; d >= 0; d--) {
            idx[d]++; if (idx[d] < out_shape[d]) break; idx[d] = 0;
        }
    }
    return out;
}

Tensor* tensor_pad(Tensor* A, int* pad_width, float constant_value) {
    if (A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    int out_shape[8];
    for (int i = 0; i < A->ndim; i++) {
        out_shape[i] = A->shape[i] + pad_width[i * 2] + pad_width[i * 2 + 1];
    }
    
    Tensor* out = tensor_create(A->ndim, out_shape);
    tensor_fill(out, constant_value);
    
    int idx[8] = {0};
    for (size_t i = 0; i < A->total_size; i++) {
        size_t off_a = 0, off_out = 0;
        for (int d = 0; d < A->ndim; d++) {
            off_a += idx[d] * A->stride[d];
            off_out += (idx[d] + pad_width[d * 2]) * out->stride[d];
        }
        F32(out)[off_out] = F32(A)[off_a];
        
        for (int d = A->ndim - 1; d >= 0; d--) {
            idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
        }
    }
    return out;
}

// ============================================================================
// RANDOM SAMPLING
// ============================================================================

Tensor* tensor_random_choice(Tensor* A, int n, bool replace) {
    if (A->ndim != 1) return NULL;
    if (!replace && n > A->total_size) n = A->total_size;
    
    Tensor* out = tensor_create_uninitialized(1, &n, A->dtype);
    size_t el_size = dtype_size(A->dtype);
    
    unsigned int seed = (unsigned int)(1234567 + time(NULL));
    
    if (replace) {
        for (int i = 0; i < n; i++) {
            int r_idx = rand_r(&seed) % A->total_size;
            memcpy((char*)out->data + i * el_size, (char*)A->data + r_idx * A->stride[0] * el_size, el_size);
        }
    } else {
        int* indices = (int*)malloc(A->total_size * sizeof(int));
        if (!indices) { tensor_free(out); TENSOR_ERROR("Malloc failed in random choice."); }
        for (size_t i = 0; i < A->total_size; i++) indices[i] = (int)i;
        for (size_t i = A->total_size - 1; i > 0; i--) {
            int j = rand_r(&seed) % (i + 1);
            int temp = indices[i]; indices[i] = indices[j]; indices[j] = temp;
        }
        for (int i = 0; i < n; i++) {
            memcpy((char*)out->data + i * el_size, (char*)A->data + indices[i] * A->stride[0] * el_size, el_size);
        }
        free(indices);
    }
    return out;
}

Tensor* tensor_random_permutation(Tensor* A) {
    Tensor* out = tensor_copy(A);
    int dim_len = out->shape[0]; 
    
    size_t row_bytes = out->stride[0] * dtype_size(out->dtype);
    void* temp_row = malloc(row_bytes);
    if (!temp_row) { tensor_free(out); TENSOR_ERROR("Malloc failed."); }
    
    unsigned int seed = (unsigned int)(1234567 + time(NULL));
    
    for (int i = dim_len - 1; i > 0; i--) {
        int j = rand_r(&seed) % (i + 1);
        if (i != j) {
            memcpy(temp_row, (char*)out->data + i * row_bytes, row_bytes);
            memcpy((char*)out->data + i * row_bytes, (char*)out->data + j * row_bytes, row_bytes);
            memcpy((char*)out->data + j * row_bytes, temp_row, row_bytes);
        }
    }
    free(temp_row);
    return out;
}

// ============================================================================
// ROBUST LINEAR ALGEBRA
// ============================================================================

Tensor* tensor_solve(Tensor* A, Tensor* B) {
    if (A->ndim != 2 || B->ndim != 2 || A->shape[0] != A->shape[1] || A->shape[0] != B->shape[0] || A->dtype != DTYPE_FLOAT32) {
        TENSOR_ERROR("FATAL [Solve]: Invalid dimensions for Ax=B.");
    }
    
    int n = A->shape[0];
    int nrhs = B->shape[1];
    
    Tensor* copyA = tensor_copy(A); 
    Tensor* out = tensor_copy(B);   
    
    int* ipiv = (int*)safe_malloc(n * sizeof(int));
    if (!ipiv) { tensor_free(copyA); tensor_free(out); TENSOR_ERROR("Malloc failed."); }
    int info = LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, nrhs, F32(copyA), n, ipiv, F32(out), nrhs);
    
    if (info != 0) {
        free(ipiv); tensor_free(copyA); tensor_free(out);
        TENSOR_ERROR("FATAL [Solve]: Matrix A is exactly singular.");
    }
    
    free(ipiv);
    tensor_free(copyA);
    return out;
}

Tensor* tensor_pinv(Tensor* A) {
    if (A->ndim != 2 || A->dtype != DTYPE_FLOAT32) return NULL;
    
    Tensor *U = NULL, *S = NULL, *Vt = NULL;
    tensor_svd(A, &U, &S, &Vt); 
    if (tensor_had_error) return NULL;
    
    int m = A->shape[0];
    int n = A->shape[1];
    
    float max_s = tensor_max(S);
    float tol = (m > n ? m : n) * 1.19209e-07f * max_s; 
    
    Tensor* S_inv = tensor_zeros(2, (int[]){n, m});
    for (size_t i = 0; i < S->total_size; i++) {
        if (F32(S)[i] > tol) {
            F32(S_inv)[i * m + i] = 1.0f / F32(S)[i];
        }
    }
    
    Tensor* V = tensor_transpose_2d(Vt);
    Tensor* U_T = tensor_transpose_2d(U);
    Tensor* V_Sinv = tensor_matmul(V, S_inv);
    Tensor* pinv = tensor_matmul(V_Sinv, U_T);
    
    tensor_free(U); tensor_free(S); tensor_free(Vt);
    tensor_free(S_inv); tensor_free(V); tensor_free(U_T); tensor_free(V_Sinv);
    
    return pinv;
}

// ============================================================================
// BOOLEAN REDUCTIONS
// ============================================================================

bool tensor_any(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) return false;
    if (tensor_is_contiguous(A)) {
        for (size_t i = 0; i < A->total_size; i++) {
            if (fabsf(F32(A)[i]) > 1e-8f) return true;
        }
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < A->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            if (fabsf(F32(A)[offset]) > 1e-8f) return true;
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return false;
}

bool tensor_all(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) return false;
    if (tensor_is_contiguous(A)) {
        for (size_t i = 0; i < A->total_size; i++) {
            if (fabsf(F32(A)[i]) <= 1e-8f) return false;
        }
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < A->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            if (fabsf(F32(A)[offset]) <= 1e-8f) return false;
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return true;
}

// ============================================================================
// MULTI-AXIS REDUCTIONS
// ============================================================================

static int cmp_int_desc(const void* a, const void* b) {
    return (*(const int*)b - *(const int*)a);
}

Tensor* tensor_reduce_multi_axis(Tensor* A, int* axes, int num_axes,
                                  Tensor* (*reduce_fn)(Tensor*, int)) {
    if (num_axes <= 0 || num_axes > A->ndim) return NULL;
    int* sorted_axes = (int*)malloc(num_axes * sizeof(int));
    if (!sorted_axes) TENSOR_ERROR("Malloc failed.");
    memcpy(sorted_axes, axes, num_axes * sizeof(int));
    qsort(sorted_axes, num_axes, sizeof(int), cmp_int_desc);

    Tensor* current = reduce_fn(A, sorted_axes[0]);  // ← no initial copy
    for (int i = 1; i < num_axes; i++) {
        Tensor* next = reduce_fn(current, sorted_axes[i]);
        tensor_free(current);
        current = next;
    }
    free(sorted_axes);
    return current;
}

Tensor* tensor_sum_multi(Tensor* A, int* axes, int num_axes) { return tensor_reduce_multi_axis(A, axes, num_axes, tensor_sum_axis); }
Tensor* tensor_mean_multi(Tensor* A, int* axes, int num_axes) { return tensor_reduce_multi_axis(A, axes, num_axes, tensor_mean_axis); }
Tensor* tensor_max_multi(Tensor* A, int* axes, int num_axes) { return tensor_reduce_multi_axis(A, axes, num_axes, tensor_max_axis); }

// ============================================================================
// MESSY DATA (NaN & Infinity)
// ============================================================================

#define TENSOR_IS_CHECK(OP_NAME, CHECK_FUNC) \
Tensor* OP_NAME(Tensor* A) { \
    if (A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32."); \
    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32); \
    if (tensor_is_contiguous(A)) { \
        for (size_t i = 0; i < A->total_size; i++) F32(out)[i] = CHECK_FUNC(F32(A)[i]) ? 1.0f : 0.0f; \
    } else { \
        int idx[8] = {0}; \
        for (size_t i = 0; i < out->total_size; i++) { \
            size_t offset = 0; \
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d]; \
            F32(out)[i] = CHECK_FUNC(F32(A)[offset]) ? 1.0f : 0.0f; \
            for (int d = A->ndim - 1; d >= 0; d--) { \
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0; \
            } \
        } \
    } \
    return out; \
}

TENSOR_IS_CHECK(tensor_isnan, isnan)
TENSOR_IS_CHECK(tensor_isinf, isinf)

void tensor_nan_to_num_inplace(Tensor* A, float nan_val, float posinf_val, float neginf_val) {
    if (A->dtype != DTYPE_FLOAT32) return;
    if (tensor_is_contiguous(A)) {
        for (size_t i = 0; i < A->total_size; i++) {
            if (isnan(F32(A)[i])) F32(A)[i] = nan_val;
            else if (isinf(F32(A)[i])) F32(A)[i] = F32(A)[i] > 0 ? posinf_val : neginf_val;
        }
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < A->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            if (isnan(F32(A)[offset])) F32(A)[offset] = nan_val;
            else if (isinf(F32(A)[offset])) F32(A)[offset] = F32(A)[offset] > 0 ? posinf_val : neginf_val;
            
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
}

// ============================================================================
// SORTING AND TOP-K
// ============================================================================

Tensor* tensor_sort(Tensor* A, int axis) {
    if (axis < 0 || axis >= A->ndim || A->dtype != DTYPE_FLOAT32) return NULL;
    Tensor* out = tensor_copy(A); 
    int dim_len = out->shape[axis];
    
    int out_ndim = out->ndim - 1;
    int out_shape[8];
    size_t stride_out[8];
    
    for (int i = 0, k = 0; i < out->ndim; i++) {
        if (i != axis) { out_shape[k] = out->shape[i]; stride_out[k] = out->stride[i]; k++; }
    }
    
    size_t num_vectors = 1;
    for (int i = 0; i < out_ndim; i++) num_vectors *= out_shape[i];
    
    float* buffer = (float*)malloc(dim_len * sizeof(float));
    if (!buffer) { tensor_free(out); TENSOR_ERROR("Malloc failed."); }
    int idx[8] = {0};
    
    for (size_t v = 0; v < num_vectors; v++) {
        size_t base = 0;
        for (int d = 0; d < out_ndim; d++) base += idx[d] * stride_out[d];
        
        for (int i = 0; i < dim_len; i++) buffer[i] = F32(out)[base + i * out->stride[axis]];
        qsort(buffer, dim_len, sizeof(float), cmp_float_asc);
        for (int i = 0; i < dim_len; i++) F32(out)[base + i * out->stride[axis]] = buffer[i];
        
        if (out_ndim > 0) {
            for (int d = out_ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < out_shape[d]) break; idx[d] = 0;
            }
        }
    }
    free(buffer);
    return out;
}

Tensor* tensor_topk(Tensor* A, int k, int axis) {
    Tensor* sorted = tensor_sort(A, axis);
    if (!sorted) TENSOR_ERROR("FATAL [TopK]: Sort failed.");
    int start = sorted->shape[axis] - k;
    if (start < 0) start = 0;
    Tensor* top_k_view = tensor_slice(sorted, axis, start, k);
    Tensor* final_out = tensor_copy(top_k_view); 
    
    tensor_free(top_k_view);
    tensor_free(sorted);
    return final_out;
}

// ============================================================================
// DEEP LEARNING (CNN Primitives)
// ============================================================================

/*
 * tensor_im2col — optimised implementation.
 *
 * Loop-order change (b,oh,ow outer; c,kh,kw inner):
 *   • Writes to out_row[col_idx] are now sequential → cache-friendly stores.
 *   • Padding zeros are never written; the tensor is initialised with zeros
 *     and only valid positions are overwritten (branch-free in the common case
 *     after the pad guard is hoisted out of the hot loop).
 *
 * OpenMP: parallelise over batch dimension — each b writes to disjoint rows of
 * the output matrix, so there are no race conditions.
 */
Tensor* tensor_im2col(Tensor* A, int kernel_h, int kernel_w,
                      int stride_h, int stride_w, int pad_h, int pad_w) {
    if (A->ndim != 4 || A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires 4D FLOAT32.");

    int batch    = A->shape[0];
    int channels = A->shape[1];
    int height   = A->shape[2];
    int width    = A->shape[3];
    int out_h    = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w    = (width  + 2 * pad_w - kernel_w) / stride_w + 1;
    int cols     = channels * kernel_h * kernel_w;
    int rows     = batch * out_h * out_w;

    /* Padding regions stay 0 from calloc inside tensor_create */
    Tensor* out = tensor_create(2, (int[]){rows, cols});

    /* Pre-compute per-channel base strides to avoid repeated multiplication */
    size_t as0 = A->stride[0];
    size_t as1 = A->stride[1];
    size_t as2 = A->stride[2];
    /* stride[3] == 1 for contiguous 4D (verified by callers), so we skip it */

    #pragma omp parallel for schedule(static) if(rows > 4096)
    for (int b = 0; b < batch; b++) {
        int base_row = b * (out_h * out_w);

        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int   row_idx = base_row + oh * out_w + ow;
                float* out_row = F32(out) + (size_t)row_idx * cols;

                /* Prefetch the next output row while filling this one */
                if (ow + 1 < out_w)
                    __builtin_prefetch(out_row + cols, 1, 1);

                int col_idx = 0;
                for (int c = 0; c < channels; c++) {
                    size_t base_a = b * as0 + c * as1;
                    for (int kh_i = 0; kh_i < kernel_h; kh_i++) {
                        int im_h = oh * stride_h - pad_h + kh_i;
                        bool h_valid = (im_h >= 0 && im_h < height);
                        for (int kw_i = 0; kw_i < kernel_w; kw_i++, col_idx++) {
                            int im_w = ow * stride_w - pad_w + kw_i;
                            if (h_valid && im_w >= 0 && im_w < width) {
                                out_row[col_idx] =
                                    F32(A)[base_a + (size_t)im_h * as2 + im_w];
                            }
                            /* else: already 0 from tensor_create */
                        }
                    }
                }
            }
        }
    }
    return out;
}

Tensor* tensor_col2im(Tensor* cols_tensor, int batch, int channels, int height, int width, 
                      int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) {
                          
    Tensor* out = tensor_zeros(4, (int[]){batch, channels, height, width});
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int cols = channels * kernel_h * kernel_w;
    
    for (int c = 0; c < channels; c++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int col_idx = c * (kernel_h * kernel_w) + kh * kernel_w + kw;
                float* in_col = F32(cols_tensor) + col_idx; 
                for (int b = 0; b < batch; b++) {
                    for (int oh = 0; oh < out_h; oh++) {
                        for (int ow = 0; ow < out_w; ow++) { 
                            int row_idx = b * (out_h * out_w) + oh * out_w + ow;
                            int im_h = oh * stride_h - pad_h + kh;
                            int im_w = ow * stride_w - pad_w + kw;
                            
                            if (im_h >= 0 && im_h < height && im_w >= 0 && im_w < width) {
                                size_t a_idx = b * out->stride[0] + c * out->stride[1] + im_h * out->stride[2] + im_w * out->stride[3];
                                F32(out)[a_idx] += in_col[row_idx * cols];
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

Tensor* tensor_conv2d(Tensor* X, Tensor* W, Tensor* bias, int stride_h, int stride_w, int pad_h, int pad_w) {
    if (X->ndim != 4 || W->ndim != 4 || X->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires 4D FLOAT32.");
    
    int batch = X->shape[0]; int in_c = X->shape[1]; int in_h = X->shape[2]; int in_w = X->shape[3];
    int out_c = W->shape[0]; int kh = W->shape[2]; int kw = W->shape[3];
    int out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;

    Tensor* X_col = tensor_im2col(X, kh, kw, stride_h, stride_w, pad_h, pad_w);
    int M = batch * out_h * out_w; int K = in_c * kh * kw; int N = out_c;

    // beta=0.0 — BLAS writes all of Y_col; zero-fill is redundant.
    Tensor* Y_col = tensor_create_uninitialized(2, (int[]){M, N}, DTYPE_FLOAT32);
    Tensor* W_contig = tensor_is_contiguous(W) ? W : tensor_copy(W);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f,
                F32(X_col), K, F32(W_contig), K, 0.0f, F32(Y_col), N); 

    if (bias) {
        Tensor* bias_contig = tensor_is_contiguous(bias) ? bias : tensor_copy(bias);
        Tensor* b_reshaped = tensor_reshape(bias_contig, 2, (int[]){1, N});
        tensor_add_inplace(Y_col, b_reshaped); 
        tensor_free(b_reshaped);
        if (!tensor_is_contiguous(bias)) tensor_free(bias_contig);
    }

    Tensor* Y_reshaped = tensor_reshape(Y_col, 4, (int[]){batch, out_h, out_w, out_c});
    Tensor* Y_transposed = tensor_transpose_nd(Y_reshaped, (int[]){0, 3, 1, 2});
    Tensor* Out = tensor_copy(Y_transposed);

    tensor_free(X_col); tensor_free(Y_col);
    tensor_free(Y_reshaped); tensor_free(Y_transposed);
    if (!tensor_is_contiguous(W)) tensor_free(W_contig);

    return Out;
}

Tensor** tensor_conv2d_backward(Tensor* dY, Tensor* X, Tensor* W, int stride_h, int stride_w, int pad_h, int pad_w) {
    int batch = X->shape[0]; int in_c = X->shape[1]; int in_h = X->shape[2]; int in_w = X->shape[3];
    int out_c = W->shape[0]; int kh = W->shape[2]; int kw = W->shape[3];
    int out_h = dY->shape[2]; int out_w = dY->shape[3];
    int M = batch * out_h * out_w; int K = in_c * kh * kw; int N = out_c;

    int axes_to_sum[] = {0, 2, 3};
    Tensor* dbias = tensor_sum_multi(dY, axes_to_sum, 3);

    Tensor* dY_transposed = tensor_transpose_nd(dY, (int[]){0, 2, 3, 1});
    Tensor* dY_contig = tensor_copy(dY_transposed); 
    Tensor* dY_col = tensor_reshape(dY_contig, 2, (int[]){M, N});
    Tensor* X_col = tensor_im2col(X, kh, kw, stride_h, stride_w, pad_h, pad_w);

    // beta=0 — BLAS overwrites entirely; uninitialized is safe and saves a memset pass.
    Tensor* dW_col = tensor_create_uninitialized(2, (int[]){N, K}, DTYPE_FLOAT32);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                N, K, M, 1.0f,
                F32(dY_col), N, F32(X_col), K, 0.0f, F32(dW_col), K);

    Tensor* dW = tensor_reshape(dW_col, 4, (int[]){out_c, in_c, kh, kw});
    dW->owns_data = true;
    dW_col->owns_data = false;

    Tensor* dX_col = tensor_create_uninitialized(2, (int[]){M, K}, DTYPE_FLOAT32);
    Tensor* W_contig = tensor_is_contiguous(W) ? W : tensor_copy(W);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, K, N, 1.0f,
                F32(dY_col), N, F32(W_contig), K, 0.0f, F32(dX_col), K); 

    Tensor* dX = tensor_col2im(dX_col, batch, in_c, in_h, in_w, kh, kw, stride_h, stride_w, pad_h, pad_w);

    tensor_free(dY_transposed); tensor_free(dY_contig); tensor_free(dY_col);
    tensor_free(X_col); tensor_free(dW_col); tensor_free(dX_col);
    if (!tensor_is_contiguous(W)) tensor_free(W_contig);

    Tensor** gradients = (Tensor**)malloc(3 * sizeof(Tensor*));
    if (!gradients) {
        tensor_free(dX); tensor_free(dW); tensor_free(dbias);
        TENSOR_ERROR_VAL(NULL, "FATAL [conv2d_backward]: OOM.");
    }
    gradients[0] = dX; gradients[1] = dW; gradients[2] = dbias;
    return gradients;
}

// ============================================================================
// LLM NLP EMBEDDINGS
// ============================================================================
Tensor* tensor_embedding_lookup(Tensor* tokens, Tensor* weights) {
    if (tokens->dtype != DTYPE_INT32) TENSOR_ERROR("Tokens must be INT32.");
    if (weights->dtype != DTYPE_FLOAT32 || weights->ndim != 2) TENSOR_ERROR("Embedding weights must be 2D FLOAT32.");

    int vocab_size = weights->shape[0];
    int embed_dim = weights->shape[1];
    
    int out_ndim = tokens->ndim + 1;
    int out_shape[8];
    for (int i = 0; i < tokens->ndim; i++) out_shape[i] = tokens->shape[i];
    out_shape[out_ndim - 1] = embed_dim;
    
    Tensor* out = tensor_create_uninitialized(out_ndim, out_shape, DTYPE_FLOAT32); 

    int32_t* tok_data = I32(tokens);
    float* w_data = F32(weights);
    float* out_data = F32(out);
    int error_flag = 0;
    // stride[1]==1 means rows are stored as consecutive floats (contiguous row layout).
    // If stride[1]!=1 (transposed or column-major slice) we must step element-by-element.
    bool w_row_contiguous = (weights->stride[1] == 1);

    #pragma omp parallel for shared(error_flag)
    for (size_t i = 0; i < tokens->total_size; i++) {
        if (error_flag) continue;
        int32_t token_id = tok_data[i];

        if (token_id < 0 || token_id >= vocab_size) {
            #pragma omp atomic write
            error_flag = 1;
            continue;
        }

        if (w_row_contiguous) {
            memcpy(&out_data[i * embed_dim],
                   &w_data[token_id * weights->stride[0]],
                   embed_dim * sizeof(float));
        } else {
            for (int j = 0; j < embed_dim; j++) {
                out_data[i * embed_dim + j] =
                    w_data[token_id * weights->stride[0] + j * weights->stride[1]];
            }
        }
    }
    if (error_flag) {
        tensor_free(out);
        TENSOR_ERROR("Token ID out of bounds.");
    }
    return out;
}

// ============================================================================
// 14. FUSED KERNELS & HARDWARE INFERENCE
// ============================================================================

void tensor_fused_bce_loss_and_grad(Tensor* preds, Tensor* targets, Tensor* grads, float* out_loss) {
    if (!_tensor_shape_assert(preds, targets, "fused_bce")) return;
    if (grads && !_tensor_shape_assert(preds, grads, "fused_bce")) return;
    
    if (!tensor_is_contiguous(preds) || !tensor_is_contiguous(targets) || (grads && !tensor_is_contiguous(grads))) {
        TENSOR_ERROR_VOID("FATAL [Fused BCE]: Tensors must be contiguous for fused execution.");
    }
    
    float total_loss = 0.0f;
    const float epsilon = 1e-7f;
    int size = preds->total_size;

    #pragma omp parallel for simd reduction(+:total_loss)
    for (int i = 0; i < size; i++) {
        float p = F32(preds)[i];
        float y = F32(targets)[i];
        
        p = p < epsilon ? epsilon : (p > 1.0f - epsilon ? 1.0f - epsilon : p);
        total_loss += -(y * logf(p) + (1.0f - y) * logf(1.0f - p));
        
        if (grads != NULL) {
            F32(grads)[i] = (p - y) / (p * (1.0f - p) + epsilon) / size;
        }
    }
    if (out_loss) *out_loss = total_loss / size;
}

void tensor_fused_adam_step(Tensor* param, Tensor* grad, Tensor* m, Tensor* v, float lr, float b1, float b2, float eps, int t) {
    if (t <= 0) {
        TENSOR_ERROR_VOID("FATAL [Fused Adam]: Step t must be >= 1, got %d.", t);
    }
    if (!_tensor_shape_assert(param, grad, "fused_adam") || 
        !_tensor_shape_assert(param, m, "fused_adam") || 
        !_tensor_shape_assert(param, v, "fused_adam")) return;
    
    if (!tensor_is_contiguous(param) || !tensor_is_contiguous(grad) || !tensor_is_contiguous(m) || !tensor_is_contiguous(v)) {
        TENSOR_ERROR_VOID("FATAL [Fused Adam]: Tensors must be contiguous.");
    }
    
    float bias_correction1 = 1.0f - powf(b1, (float)t);
    float bias_correction2 = 1.0f - powf(b2, (float)t);
    float step_size = lr / bias_correction1;
    int size = (int)param->total_size;

    #pragma omp parallel for simd
    for (int i = 0; i < size; i++) {
        float g = F32(grad)[i];
        F32(m)[i] = b1 * F32(m)[i] + (1.0f - b1) * g;
        F32(v)[i] = b2 * F32(v)[i] + (1.0f - b2) * g * g;
        // fmaxf: under -ffast-math, EMA of g² can drift to tiny negatives via FP reassociation.
        float v_hat_sqrt = sqrtf(fmaxf(0.0f, F32(v)[i] / bias_correction2)) + eps;
        F32(param)[i] -= step_size * (F32(m)[i] / v_hat_sqrt);
    }
}

void tensor_hardware_tree_predict(Tensor* X, HardwareNode* nodes, Tensor* out) {
    if (X->ndim != 2 || out->ndim != 1 || X->shape[0] != out->shape[0])
        TENSOR_ERROR_VOID("Invalid shapes.");
    if (!tensor_is_contiguous(X)) TENSOR_ERROR_VOID("X must be contiguous.");
    int num_rows = X->shape[0];
    int num_features = X->shape[1];
    #pragma omp parallel for
    for (int i = 0; i < num_rows; i++) {
        const float* row = &F32(X)[i * num_features];
        int curr = 0, depth = 0;
        while (nodes[curr].left_idx != -1 && depth < 64) {
            int fi = nodes[curr].feature_idx;
            if (fi < 0 || fi >= num_features) break;
            curr = (row[fi] < nodes[curr].threshold)
                ? nodes[curr].left_idx : nodes[curr].right_idx;
            if (curr < 0) break;
            depth++;
        }
        F32(out)[i] = nodes[curr].value;
    }
}

// ============================================================================
// 15. I/O SERIALIZATION
// ============================================================================

void tensor_save_to_file(Tensor* t, const char* filepath) {
    if (!t) return;
    FILE* f = fopen(filepath, "wb");
    if (!f) TENSOR_ERROR_VOID("FATAL [I/O]: Cannot open file for writing.");
    
    uint32_t magic = 0x544E5353; // TENSOR_MAGIC_V2 ("TNSS")
    fwrite(&magic, sizeof(uint32_t), 1, f);
    int dtype_val = (int)t->dtype;
    fwrite(&dtype_val, sizeof(int), 1, f);
    fwrite(&t->ndim, sizeof(int), 1, f);
    fwrite(t->shape, sizeof(int), t->ndim, f);
    
    if (tensor_is_contiguous(t)) {
        fwrite(t->data, dtype_size(t->dtype), t->total_size, f);
    } else {
        Tensor* temp = tensor_copy(t);
        fwrite(temp->data, dtype_size(temp->dtype), temp->total_size, f);
        tensor_free(temp);
    }
    fclose(f);
}

Tensor* tensor_load_from_file(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) TENSOR_ERROR("FATAL [I/O]: Cannot open file for reading.");
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1) { fclose(f); TENSOR_ERROR("FATAL [I/O]: Invalid format."); }
    
    TensorDType dtype = DTYPE_FLOAT32;
    if (magic == 0x544E5353) { // TENSOR_MAGIC_V2
        int dval;
        if (fread(&dval, sizeof(int), 1, f) != 1) {
            fclose(f);
            TENSOR_ERROR("FATAL [I/O]: Corrupt dtype field in V2 magic file.");
        }
        dtype = (TensorDType)dval;
    } else if (magic != 0x544E5352) { // TENSOR_MAGIC_V1 ("TNSR")
        fclose(f); TENSOR_ERROR("FATAL [I/O]: Corrupt Magic Number.");
    }
    
    int ndim;
    if (fread(&ndim, sizeof(int), 1, f) != 1 || ndim < 1 || ndim > 8) { fclose(f); TENSOR_ERROR("FATAL [I/O]: Corrupt dims."); }
    int shape[8] = {0};
    if (fread(shape, sizeof(int), ndim, f) != (size_t)ndim) { fclose(f); TENSOR_ERROR("FATAL [I/O]: Corrupt shape."); }
    
    Tensor* out = tensor_create_uninitialized(ndim, shape, dtype);
    if (fread(out->data, dtype_size(dtype), out->total_size, f) != out->total_size) { 
        fclose(f); tensor_free(out); TENSOR_ERROR("FATAL [I/O]: Corrupt data payload.");
    }
    
    fclose(f);
    return out;
}

int tensor_save_safetensors(const char* filepath, const char* json_header, uint64_t json_len, Tensor** tensors, int num_tensors) {
    FILE* f = fopen(filepath, "wb");
    if (!f) return 0;
    
    fwrite(&json_len, sizeof(uint64_t), 1, f);
    fwrite(json_header, 1, json_len, f);
    
    for (int i = 0; i < num_tensors; i++) {
        Tensor* t = tensors[i];
        if (tensor_is_contiguous(t)) {
            fwrite(t->data, 1, t->byte_size, f);
        } else {
            Tensor* temp = tensor_copy(t);
            fwrite(temp->data, 1, temp->byte_size, f);
            tensor_free(temp);
        }
    }
    
    fclose(f);
    return 1;
}
// ============================================================================
// 16. FUSED NEURAL NETWORK KERNELS (New — Zero-copy, FMA-accelerated)
// ============================================================================

/*
 * tensor_linear — fused fully-connected layer: out = X @ W^T + bias
 *
 *   X    : [m, k]   (or any shape with last-dim k; treated as [m, k])
 *   W    : [n, k]   (weight matrix, transposed convention: rows = output neurons)
 *   bias : [n]      (optional; NULL skips the add)
 *   out  : [m, n]
 *
 * Fusion benefit: single SGEMM call + one AVX2 pass for bias — no temporary
 * tensor allocation between GEMM and add.  Compared to matmul()+add() this
 * saves one full output-buffer allocation and one extra memory pass.
 */
Tensor* tensor_linear(Tensor* X, Tensor* W, Tensor* bias) {
    if (X->dtype != DTYPE_FLOAT32 || W->dtype != DTYPE_FLOAT32)
        TENSOR_ERROR("FATAL [Linear]: Requires FLOAT32.");
    if (W->ndim != 2 || X->shape[X->ndim - 1] != W->shape[1])
        TENSOR_ERROR("FATAL [Linear]: W must be 2D [n,k] with k == X last dim.");

    int k = X->shape[X->ndim - 1];
    int m = (int)(X->total_size / k);
    int n = W->shape[0];

    /* Compact if needed — single O(N) copy, then pure pointer arithmetic */
    Tensor* x_work = tensor_is_contiguous(X) ? X : tensor_copy(X);
    Tensor* w_work = tensor_is_contiguous(W) ? W : tensor_copy(W);

    /* Build output shape: same as X but with last dim replaced by n */
    int out_shape[8];
    for (int i = 0; i < X->ndim - 1; i++) out_shape[i] = X->shape[i];
    out_shape[X->ndim - 1] = n;
    Tensor* out = tensor_create_uninitialized(X->ndim, out_shape, DTYPE_FLOAT32);

    /* SGEMM: out = X * W^T  (W stored as [n,k], so Trans gives [k,n]) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0f,
                F32(x_work), k,
                F32(w_work), k,
                0.0f, F32(out), n);

    /* Fused bias addition: single AVX2 pass over the output matrix */
    if (bias != NULL) {
        if (bias->dtype != DTYPE_FLOAT32)
            TENSOR_ERROR("FATAL [Linear]: bias must be FLOAT32.");
        Tensor* b_work = tensor_is_contiguous(bias) ? bias : tensor_copy(bias);
        float*  b_data = F32(b_work);
        float*  o_data = F32(out);

        #pragma omp parallel for schedule(static) if((size_t)m * n > 100000)
        for (int i = 0; i < m; i++) {
            float* row = o_data + (size_t)i * n;
            size_t j = 0;
#ifdef __AVX2__
            for (; j + 7 < (size_t)n; j += 8) {
                _mm256_storeu_ps(row + j,
                    _mm256_add_ps(_mm256_loadu_ps(row + j),
                                  _mm256_loadu_ps(b_data + j)));
            }
#endif
            for (; j < (size_t)n; j++) row[j] += b_data[j];
        }

        if (b_work != bias) tensor_free(b_work);
    }

    if (x_work != X) tensor_free(x_work);
    if (w_work != W) tensor_free(w_work);
    return out;
}

/*
 * tensor_add_relu — fused elementwise: out = relu(A + B)
 *
 * Saves one full output-buffer allocation + one extra memory pass vs add()+relu().
 * Uses AVX2 _mm256_max_ps for the relu in the same loop as the add.
 */
Tensor* tensor_add_relu(Tensor* A, Tensor* B) {
    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32)
        TENSOR_ERROR("FATAL [AddReLU]: Requires FLOAT32.");
    if (A->total_size != B->total_size)
        TENSOR_ERROR("FATAL [AddReLU]: Shape mismatch.");

    Tensor* a_work = tensor_is_contiguous(A) ? A : tensor_copy(A);
    Tensor* b_work = tensor_is_contiguous(B) ? B : tensor_copy(B);

    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32);
    size_t  n   = A->total_size;
    float*  a   = F32(a_work);
    float*  b   = F32(b_work);
    float*  o   = F32(out);

    size_t i = 0;
#ifdef __AVX2__
    __m256 zero  = _mm256_setzero_ps();
    size_t limit = n & ~7UL;
    _Pragma("omp parallel for schedule(static) if(limit > 100000)")
    for (size_t j = 0; j < limit; j += 8) {
        __builtin_prefetch(a + j + 64, 0, 1);
        __builtin_prefetch(b + j + 64, 0, 1);
        _mm256_storeu_ps(o + j,
            _mm256_max_ps(
                _mm256_add_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j)),
                zero));
    }
    i = limit;
#endif
    for (; i < n; i++) { float v = a[i] + b[i]; o[i] = v > 0.0f ? v : 0.0f; }

    if (a_work != A) tensor_free(a_work);
    if (b_work != B) tensor_free(b_work);
    return out;
}

/*
 * tensor_mul_add — fused FMA: out = A * B + C
 *
 * Uses _mm256_fmadd_ps (single FMA instruction, -mfma flag already set).
 * Saves one full pass over memory vs mul()+add().
 */
Tensor* tensor_mul_add(Tensor* A, Tensor* B, Tensor* C) {
    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32 || C->dtype != DTYPE_FLOAT32)
        TENSOR_ERROR("FATAL [MulAdd]: Requires FLOAT32.");
    if (A->total_size != B->total_size || A->total_size != C->total_size)
        TENSOR_ERROR("FATAL [MulAdd]: Shape mismatch.");

    Tensor* a_work = tensor_is_contiguous(A) ? A : tensor_copy(A);
    Tensor* b_work = tensor_is_contiguous(B) ? B : tensor_copy(B);
    Tensor* c_work = tensor_is_contiguous(C) ? C : tensor_copy(C);

    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32);
    size_t  n   = A->total_size;
    float*  a   = F32(a_work);
    float*  b   = F32(b_work);
    float*  c   = F32(c_work);
    float*  o   = F32(out);

    size_t i = 0;
#ifdef __AVX2__
    size_t limit = n & ~7UL;
    _Pragma("omp parallel for schedule(static) if(limit > 100000)")
    for (size_t j = 0; j < limit; j += 8) {
        __builtin_prefetch(a + j + 64, 0, 1);
        __builtin_prefetch(b + j + 64, 0, 1);
        /* fmadd: a*b + c in one FP instruction */
        _mm256_storeu_ps(o + j,
            _mm256_fmadd_ps(_mm256_loadu_ps(a + j),
                            _mm256_loadu_ps(b + j),
                            _mm256_loadu_ps(c + j)));
    }
    i = limit;
#endif
    for (; i < n; i++) o[i] = a[i] * b[i] + c[i];

    if (a_work != A) tensor_free(a_work);
    if (b_work != B) tensor_free(b_work);
    if (c_work != C) tensor_free(c_work);
    return out;
}

// ============================================================================
// 17. THREADING CONTROL
// ============================================================================

/* Weak-symbol declarations so the code links against any BLAS flavour
   (OpenBLAS, ATLAS, MKL via openblas compat shim) without hard-dep failures. */
extern void openblas_set_num_threads(int) __attribute__((weak));
extern void goto_set_num_threads(int)     __attribute__((weak));   /* older name */

/*
 * tensor_configure_threading — set OpenMP thread count and BLAS thread count
 * independently.  Call once at startup to prevent OpenBLAS × OpenMP
 * oversubscription (common cause of 2× slowdown on multi-core systems).
 *
 * Recommended:
 *   omp_threads  = physical core count
 *   blas_threads = 1  (when outer OpenMP loops are used)
 *   blas_threads = physical core count  (when only BLAS, no outer OpenMP)
 */
void tensor_configure_threading(int omp_threads, int blas_threads) {
    if (omp_threads  > 0) omp_set_num_threads(omp_threads);
    if (blas_threads > 0) {
        if (openblas_set_num_threads) openblas_set_num_threads(blas_threads);
        else if (goto_set_num_threads) goto_set_num_threads(blas_threads);
    }
}
