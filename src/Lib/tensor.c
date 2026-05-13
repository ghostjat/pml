#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
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
// 1. SAFE MEMORY & LIFECYCLE (Size-Class Pools)
// ============================================================================
#define POOL_MAX_BINS 64
#define POOL_BIN_SHIFT 10 // 1KB bins up to 64KB

typedef struct PoolNode {
    struct PoolNode* next;
} PoolNode;

static PoolNode* size_pools[POOL_MAX_BINS] = {NULL};
static omp_lock_t pool_locks[POOL_MAX_BINS];
static bool pools_initialized = false;

static void init_pools() {
    if (!pools_initialized) {
        for (int i = 0; i < POOL_MAX_BINS; i++) omp_init_lock(&pool_locks[i]);
        pools_initialized = true;
    }
}

void* safe_malloc(size_t size) {
    if (size == 0) return NULL;
    void* ptr = malloc(size);
    if (!ptr) TENSOR_ERROR_VAL(NULL, "FATAL: Out of memory (malloc).");
    memset(ptr, 0, size);
    return ptr;
}

void* safe_memalign(size_t alignment, size_t size) {
    if (size == 0) return NULL;
    init_pools();
    
    // Calculate bin index based on a size rounded up to the nearest bin
    int bin = (size + (1 << POOL_BIN_SHIFT) - 1) >> POOL_BIN_SHIFT;
    size_t alloc_size = bin << POOL_BIN_SHIFT;
    if (alloc_size == 0) alloc_size = 1 << POOL_BIN_SHIFT;
    
    if (bin < POOL_MAX_BINS && alignment <= 64) {
        omp_set_lock(&pool_locks[bin]);
        PoolNode* node = size_pools[bin];
        if (node) size_pools[bin] = node->next;
        omp_unset_lock(&pool_locks[bin]);
        if (node) {
            memset(node, 0, size);
            return (void*)node;
        }
        size = alloc_size; // Allocate the full bin size
    }
    
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment > 64 ? alignment : 64, size) != 0 || !ptr) {
        TENSOR_ERROR_VAL(NULL, "FATAL: Memalign failed.");
    }
    memset(ptr, 0, size);
    return ptr;
}

void safe_free_size(void* ptr, size_t size) {
    if (!ptr) return;
    init_pools();
    int bin = (size + (1 << POOL_BIN_SHIFT) - 1) >> POOL_BIN_SHIFT;
    if (bin < POOL_MAX_BINS) {
        omp_set_lock(&pool_locks[bin]);
        PoolNode* node = (PoolNode*)ptr;
        node->next = size_pools[bin];
        size_pools[bin] = node;
        omp_unset_lock(&pool_locks[bin]);
        return;
    }
    free(ptr);
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
    // fall back to a 64-byte sentinel for empty tensors
    size_t alloc_size = t->byte_size > 0 ? t->byte_size : 64;
    void* data = safe_memalign(64, alloc_size);
    if (!data) {
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
        if (t->owns_data) {
            size_t alloc_size = t->byte_size > 0 ? t->byte_size : 64;
            safe_free_size(t->data, alloc_size);
        }
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
#ifdef __AVX512F__
        __m512 v_val = _mm512_set1_ps(val);
        for (; i + 15 < t->total_size; i += 16) _mm512_storeu_ps(&F32(t)[i], v_val);
#elif defined(__AVX2__)
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
        size_t limit = OUT_TENSOR->total_size; \
        /* Precompute base offsets for outer dimensions to avoid modulo per element */ \
        size_t base_stride[8] = {0}; \
        for (int d = out_ndim - 1; d >= 0; d--) { \
            base_stride[d] = (d == out_ndim - 1) ? 1 : base_stride[d+1] * out_shape[d+1]; \
        } \
        _Pragma("omp parallel for schedule(static) if(limit > 100000)") \
        for (size_t i = 0; i < limit; i++) { \
            size_t temp = i; \
            size_t offset_a = 0, offset_b = 0; \
            for (int d = 0; d < out_ndim; d++) { \
                size_t c = temp / base_stride[d]; \
                temp %= base_stride[d]; \
                offset_a += c * stride_A[d]; \
                offset_b += c * stride_B[d]; \
            } \
            if (IN_PLACE) F32(OUT_TENSOR)[offset_a] = F32(A)[offset_a] SCALAR_OP F32(B)[offset_b]; \
            else F32(OUT_TENSOR)[i] = F32(A)[offset_a] SCALAR_OP F32(B)[offset_b]; \
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
        size_t limit = OUT_TENSOR->total_size; \
        _Pragma("omp parallel for schedule(static) if(limit > 100000)") \
        for (size_t i = 0; i < limit; i++) { \
            size_t temp = i; \
            size_t offset_a = 0; \
            for (int d = A->ndim - 1; d >= 0; d--) { \
                size_t c = temp % A->shape[d]; \
                temp /= A->shape[d]; \
                offset_a += c * A->stride[d]; \
            } \
            if (IN_PLACE) F32(OUT_TENSOR)[offset_a] = F32(A)[offset_a] SCALAR_OP val; \
            else F32(OUT_TENSOR)[i] = F32(A)[offset_a] SCALAR_OP val; \
        } \
    }

Tensor* tensor_add_scalar(Tensor* A, float val) { Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32); TENSOR_SCALAR_IMPL(_mm256_add_ps, +, out, 0, TENSOR_ERROR); return out; }
Tensor* tensor_mul_scalar(Tensor* A, float val) { Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32); TENSOR_SCALAR_IMPL(_mm256_mul_ps, *, out, 0, TENSOR_ERROR); return out; }
void tensor_add_scalar_inplace(Tensor* A, float val) { TENSOR_SCALAR_IMPL(_mm256_add_ps, +, A, 1, TENSOR_ERROR_VOID); }
void tensor_mul_scalar_inplace(Tensor* A, float val) { TENSOR_SCALAR_IMPL(_mm256_mul_ps, *, A, 1, TENSOR_ERROR_VOID); }

void tensor_clamp_inplace(Tensor* A, float lo, float hi) {
    if (!A || !A->data) return;
    float* d = (float*)A->data;
    size_t n = A->total_size;
#ifdef __AVX2__
    __m256 vlo = _mm256_set1_ps(lo);
    __m256 vhi = _mm256_set1_ps(hi);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(d + i);
        v = _mm256_max_ps(v, vlo);
        v = _mm256_min_ps(v, vhi);
        _mm256_storeu_ps(d + i, v);
    }
    for (; i < n; i++) { d[i] = d[i] < lo ? lo : (d[i] > hi ? hi : d[i]); }
#else
    for (size_t i = 0; i < n; i++) { d[i] = d[i] < lo ? lo : (d[i] > hi ? hi : d[i]); }
#endif
}

// ============================================================================
// 5. UNARY MATH & LOGICAL
// ============================================================================

static inline float _square(float x) { return x * x; }
static inline float _sign(float x) { return (x > 0.0f) - (x < 0.0f); }
/* Bit-level NaN check — immune to -ffast-math which breaks isnan()/isfinite() */
static inline int _f32_is_nan(float x) {
    uint32_t bits; memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x7FFFFFFFu) > 0x7F800000u;
}
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
        size_t n = A->total_size;
        float* a = F32(A);
        float* o = F32(out);
        size_t i = 0;
#ifdef __AVX512F__
        __m512 vmin = _mm512_set1_ps(min_val);
        __m512 vmax = _mm512_set1_ps(max_val);
        size_t limit = n & ~15UL;
        for (; i < limit; i += 16) {
            __m512 val = _mm512_loadu_ps(a + i);
            val = _mm512_max_ps(vmin, val);
            val = _mm512_min_ps(vmax, val);
            _mm512_storeu_ps(o + i, val);
        }
#elif defined(__AVX2__)
        __m256 vmin = _mm256_set1_ps(min_val);
        __m256 vmax = _mm256_set1_ps(max_val);
        size_t limit = n & ~7UL;
        for (; i < limit; i += 8) {
            __m256 val = _mm256_loadu_ps(a + i);
            val = _mm256_max_ps(vmin, val);
            val = _mm256_min_ps(vmax, val);
            _mm256_storeu_ps(o + i, val);
        }
#endif
        for (; i < n; i++) {
            float v = a[i];
            o[i] = (v < min_val) ? min_val : ((v > max_val) ? max_val : v);
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

Tensor* tensor_relu(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32);
    if (tensor_is_contiguous(A)) {
        size_t n = A->total_size;
        float* a = F32(A);
        float* o = F32(out);
        size_t i = 0;
#ifdef __AVX512F__
        __m512 vzero = _mm512_setzero_ps();
        size_t limit = n & ~15UL;
        for (; i < limit; i += 16) {
            __m512 val = _mm512_loadu_ps(a + i);
            _mm512_storeu_ps(o + i, _mm512_max_ps(vzero, val));
        }
#elif defined(__AVX2__)
        __m256 vzero = _mm256_setzero_ps();
        size_t limit = n & ~7UL;
        for (; i < limit; i += 8) {
            __m256 val = _mm256_loadu_ps(a + i);
            _mm256_storeu_ps(o + i, _mm256_max_ps(vzero, val));
        }
#endif
        for (; i < n; i++) {
            float v = a[i];
            o[i] = v > 0.0f ? v : 0.0f;
        }
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < out->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            float v = F32(A)[offset];
            F32(out)[i] = v > 0.0f ? v : 0.0f;
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return out;
}

#define TENSOR_LOGICAL_OP(OP_NAME, OP, CMP_PRED) \
Tensor* OP_NAME(Tensor* A, Tensor* B) { \
    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32."); \
    int out_ndim, out_shape[8]; \
    size_t stride_A[8] = {0}, stride_B[8] = {0}; \
    if (!tensor_broadcast_shapes(A, B, &out_ndim, out_shape, stride_A, stride_B)) { \
        TENSOR_ERROR("FATAL: Shapes not broadcastable."); \
    } \
    Tensor* out = tensor_create_uninitialized(out_ndim, out_shape, DTYPE_FLOAT32); \
    if (tensor_is_contiguous(A) && tensor_is_contiguous(B) && A->total_size == B->total_size) { \
        size_t n = out->total_size; \
        float* a = F32(A); float* b = F32(B); float* o = F32(out); \
        size_t i = 0; \
        _Pragma("omp parallel for simd if(n > 100000)") \
        for (size_t j = 0; j < n; j++) o[j] = (a[j] OP b[j]) ? 1.0f : 0.0f; \
    } else { \
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
    } \
    return out; \
}

TENSOR_LOGICAL_OP(tensor_equal,         ==, _CMP_EQ_OQ)
TENSOR_LOGICAL_OP(tensor_not_equal,     !=, _CMP_NEQ_OQ)
TENSOR_LOGICAL_OP(tensor_greater,        >, _CMP_GT_OQ)
TENSOR_LOGICAL_OP(tensor_greater_equal, >=, _CMP_GE_OQ)
TENSOR_LOGICAL_OP(tensor_less,           <, _CMP_LT_OQ)
TENSOR_LOGICAL_OP(tensor_less_equal,    <=, _CMP_LE_OQ)

Tensor* tensor_logical_not(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32);
    if (tensor_is_contiguous(A)) {
        size_t n = A->total_size;
        float* a = F32(A); float* o = F32(out);
        size_t i = 0;
#ifdef __AVX2__
        __m256 vzero = _mm256_setzero_ps();
        __m256 vone  = _mm256_set1_ps(1.0f);
        size_t limit = n & ~7UL;
        for (; i < limit; i += 8) {
            __m256 cmp = _mm256_cmp_ps(_mm256_loadu_ps(a + i), vzero, _CMP_EQ_OQ);
            _mm256_storeu_ps(o + i, _mm256_and_ps(cmp, vone));
        }
#endif
        for (; i < n; i++) o[i] = (a[i] == 0.0f) ? 1.0f : 0.0f;
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

/* Fast 2D-specialised sum_axis.
 * axis=1 ([M,N] → [M]): AVX2 horizontal sum per row, OpenMP over rows.
 * axis=0 ([M,N] → [N]): OpenMP over columns, vertical accumulation.
 * All other cases fall through to the generic macro path.             */
Tensor* tensor_sum_axis(Tensor* A, int axis) {
    if (A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("FATAL: Aggregation requires FLOAT32.");
    if (axis < 0 || axis >= A->ndim) return NULL;

    if (A->ndim == 2 && tensor_is_contiguous(A)) {
        int M = A->shape[0], N = A->shape[1];
        if (axis == 1) {
            /* Sum along columns → [M] output */
            int out_shape[1] = { M };
            Tensor* out = tensor_create_dtype(1, out_shape, DTYPE_FLOAT32);
            float* __restrict o = F32(out);
            const float* __restrict a = (const float*)__builtin_assume_aligned(F32(A), 64);
            #pragma omp parallel for schedule(static) if(M > 64)
            for (int r = 0; r < M; r++) {
                const float* __restrict row = a + (size_t)r * N;
                float s = 0.0f;
                int j = 0;
#ifdef __AVX512F__
                { __m512 vs0 = _mm512_setzero_ps(), vs1 = vs0, vs2 = vs0, vs3 = vs0;
                  for (; j <= N - 64; j += 64) {
                      vs0 = _mm512_add_ps(vs0, _mm512_loadu_ps(row + j));
                      vs1 = _mm512_add_ps(vs1, _mm512_loadu_ps(row + j + 16));
                      vs2 = _mm512_add_ps(vs2, _mm512_loadu_ps(row + j + 32));
                      vs3 = _mm512_add_ps(vs3, _mm512_loadu_ps(row + j + 48)); }
                  vs0 = _mm512_add_ps(_mm512_add_ps(vs0,vs1), _mm512_add_ps(vs2,vs3));
                  for (; j <= N - 16; j += 16)
                      vs0 = _mm512_add_ps(vs0, _mm512_loadu_ps(row + j));
                  s = _mm512_reduce_add_ps(vs0); }
#elif defined(__AVX2__)
                { __m256 vs0 = _mm256_setzero_ps(), vs1 = vs0, vs2 = vs0, vs3 = vs0;
                  for (; j <= N - 32; j += 32) {
                      vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(row + j));
                      vs1 = _mm256_add_ps(vs1, _mm256_loadu_ps(row + j +  8));
                      vs2 = _mm256_add_ps(vs2, _mm256_loadu_ps(row + j + 16));
                      vs3 = _mm256_add_ps(vs3, _mm256_loadu_ps(row + j + 24)); }
                  vs0 = _mm256_add_ps(_mm256_add_ps(vs0,vs1), _mm256_add_ps(vs2,vs3));
                  for (; j <= N - 8; j += 8)
                      vs0 = _mm256_add_ps(vs0, _mm256_loadu_ps(row + j));
                  __m128 lo = _mm256_castps256_ps128(vs0);
                  __m128 hi = _mm256_extractf128_ps(vs0, 1);
                  lo = _mm_add_ps(lo, hi);
                  lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
                  s = _mm_cvtss_f32(lo); }
#endif
                for (; j < N; j++) s += row[j];
                o[r] = s;
            }
            return out;
        } else { /* axis == 0 */
            /* Sum along rows → [N] output */
            int out_shape[1] = { N };
            Tensor* out = tensor_create_dtype(1, out_shape, DTYPE_FLOAT32);
            float* __restrict o = F32(out);
            const float* __restrict a = (const float*)__builtin_assume_aligned(F32(A), 64);
            /* Tiled accumulation for better cache reuse */
            const int TILE = 64;
            for (int r0 = 0; r0 < M; r0 += TILE) {
                int r1 = r0 + TILE < M ? r0 + TILE : M;
                int j = 0;
#ifdef __AVX2__
                for (; j <= N - 8; j += 8) {
                    __m256 acc = _mm256_loadu_ps(o + j);
                    for (int r = r0; r < r1; r++)
                        acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + (size_t)r * N + j));
                    _mm256_storeu_ps(o + j, acc);
                }
#endif
                for (; j < N; j++) {
                    float s = o[j];
                    for (int r = r0; r < r1; r++) s += a[(size_t)r * N + j];
                    o[j] = s;
                }
            }
            return out;
        }
    }

    /* Generic N-D fallback (unchanged) */
    int out_shape[8]; int out_ndim = 0;
    for (int i = 0; i < A->ndim; i++) if (i != axis) out_shape[out_ndim++] = A->shape[i];
    if (out_ndim == 0) { out_ndim = 1; out_shape[0] = 1; }
    Tensor* out = tensor_create(out_ndim, out_shape);
    int idx[8] = {0};
    for (size_t i = 0; i < A->total_size; i++) {
        size_t offset_a = 0, offset_out = 0; int out_d = 0;
        for (int d = 0; d < A->ndim; d++) {
            offset_a += idx[d] * A->stride[d];
            if (d != axis) offset_out += idx[d] * out->stride[out_d++];
        }
        F32(out)[offset_out] += F32(A)[offset_a];
        for (int d = A->ndim - 1; d >= 0; d--) {
            idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
        }
    }
    return out;
}

TENSOR_AXIS_AGG(tensor_mean_axis, 0.0f, _ADD_OP, for (size_t j=0; j<out->total_size; j++) F32(out)[j] /= A->shape[axis])
TENSOR_AXIS_AGG(tensor_max_axis, -INFINITY, _MAX_OP, )
TENSOR_AXIS_AGG(tensor_min_axis, INFINITY, _MIN_OP, )

/*
 * tensor_sum — two-tier strategy:
 *   ≥ 500 K elements : OpenMP parallel reduction (auto-vectorized by -O3 -mavx2)
 *   < 500 K elements : 4-accumulator AVX2/AVX512 loop (hides latency without thread overhead)
 */
float tensor_sum(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) return 0.0f;
    float sum = 0.0f;
    if (tensor_is_contiguous(A)) {
        const float* data = (const float*)__builtin_assume_aligned(F32(A), 64);
        size_t n = A->total_size;

        if (n >= 500000) {
            #pragma omp parallel for simd reduction(+:sum) schedule(static)
            for (size_t i = 0; i < n; i++) sum += data[i];
        } else {
#ifdef __AVX512F__
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
            __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
            size_t i = 0;
            for (; i + 63 < n; i += 64) {
                __builtin_prefetch(data + i + 256, 0, 1);
                a0 = _mm512_add_ps(a0, _mm512_loadu_ps(data + i));
                a1 = _mm512_add_ps(a1, _mm512_loadu_ps(data + i + 16));
                a2 = _mm512_add_ps(a2, _mm512_loadu_ps(data + i + 32));
                a3 = _mm512_add_ps(a3, _mm512_loadu_ps(data + i + 48));
            }
            a0 = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
            sum = _mm512_reduce_add_ps(a0);
            for (; i < n; i++) sum += data[i];
#elif defined(__AVX2__)
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
        const float* data = (const float*)__builtin_assume_aligned(F32(A), 64);
        size_t n = A->total_size;
#ifdef __AVX2__
        size_t i = 0;
        if (n >= 32) {
            __m256 p0 = _mm256_set1_ps(1.0f), p1 = _mm256_set1_ps(1.0f);
            __m256 p2 = _mm256_set1_ps(1.0f), p3 = _mm256_set1_ps(1.0f);
            size_t limit = n & ~31UL;
            for (; i < limit; i += 32) {
                __builtin_prefetch(data + i + 128, 0, 1);
                p0 = _mm256_mul_ps(p0, _mm256_loadu_ps(data + i));
                p1 = _mm256_mul_ps(p1, _mm256_loadu_ps(data + i +  8));
                p2 = _mm256_mul_ps(p2, _mm256_loadu_ps(data + i + 16));
                p3 = _mm256_mul_ps(p3, _mm256_loadu_ps(data + i + 24));
            }
            p0 = _mm256_mul_ps(_mm256_mul_ps(p0, p1), _mm256_mul_ps(p2, p3));
            float tmp[8]; _mm256_storeu_ps(tmp, p0);
            prod = tmp[0]*tmp[1]*tmp[2]*tmp[3]*tmp[4]*tmp[5]*tmp[6]*tmp[7];
            for (; i < n; i++) prod *= data[i];
        } else {
            for (; i < n; i++) prod *= data[i];
        }
#else
        for (size_t i = 0; i < n; i++) prod *= data[i];
#endif
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
        const float* data = (const float*)__builtin_assume_aligned(F32(A), 64);
        size_t n = A->total_size;
        float min_val = INFINITY;
        size_t i = 0;
#ifdef __AVX512F__
        if (n >= 16) {
            __m512 vmin = _mm512_set1_ps(INFINITY);
            size_t limit = n & ~15UL;
            for (; i < limit; i += 16) {
                __builtin_prefetch(data + i + 128, 0, 1);
                vmin = _mm512_min_ps(vmin, _mm512_loadu_ps(data + i));
            }
            min_val = _mm512_reduce_min_ps(vmin);
        }
#elif defined(__AVX2__)
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
        const float* data = (const float*)__builtin_assume_aligned(F32(A), 64);
        size_t n = A->total_size;
        float max_val = -INFINITY;
        size_t i = 0;
#ifdef __AVX512F__
        if (n >= 16) {
            __m512 vmax = _mm512_set1_ps(-INFINITY);
            size_t limit = n & ~15UL;
            for (; i < limit; i += 16) {
                __builtin_prefetch(data + i + 128, 0, 1);
                vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(data + i));
            }
            max_val = _mm512_reduce_max_ps(vmax);
        }
#elif defined(__AVX2__)
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
    if (tensor_is_contiguous(A)) {
        float min_val = tensor_min(A);
        const float* data = F32(A);
        size_t n = A->total_size;
        for (size_t i = 0; i < n; i++) {
            if (data[i] == min_val) return (int)i;
        }
        return 0;
    }
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
    if (tensor_is_contiguous(A)) {
        float max_val = tensor_max(A);
        const float* data = F32(A);
        size_t n = A->total_size;
        for (size_t i = 0; i < n; i++) {
            if (data[i] == max_val) return (int)i;
        }
        return 0;
    }
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

#ifdef __AVX2__
static inline float _hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
#endif

static inline float micro_sdot(int n, const float* x, int incx, const float* y, int incy) {
    if (n <= 0) return 0.0f;
#ifdef __AVX512F__
    if (incx == 1 && incy == 1) {
        __m512 vsum0 = _mm512_setzero_ps();
        __m512 vsum1 = _mm512_setzero_ps();
        int i = 0;
        for (; i + 31 < n; i += 32) {
            __m512 vx0 = _mm512_loadu_ps(x + i);
            __m512 vy0 = _mm512_loadu_ps(y + i);
            __m512 vx1 = _mm512_loadu_ps(x + i + 16);
            __m512 vy1 = _mm512_loadu_ps(y + i + 16);
            vsum0 = _mm512_fmadd_ps(vx0, vy0, vsum0);
            vsum1 = _mm512_fmadd_ps(vx1, vy1, vsum1);
        }
        vsum0 = _mm512_add_ps(vsum0, vsum1);
        for (; i + 15 < n; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vy = _mm512_loadu_ps(y + i);
            vsum0 = _mm512_fmadd_ps(vx, vy, vsum0);
        }
        float sum = _mm512_reduce_add_ps(vsum0);
        for (; i < n; i++) sum += x[i] * y[i];
        return sum;
    }
#elif defined(__AVX2__)
    if (incx == 1 && incy == 1) {
        float sum = 0.0f;
        __m256 vsum0 = _mm256_setzero_ps();
        __m256 vsum1 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 15 < n; i += 16) {
            __m256 vx0 = _mm256_loadu_ps(x + i);
            __m256 vy0 = _mm256_loadu_ps(y + i);
            __m256 vx1 = _mm256_loadu_ps(x + i + 8);
            __m256 vy1 = _mm256_loadu_ps(y + i + 8);
            vsum0 = _mm256_fmadd_ps(vx0, vy0, vsum0);
            vsum1 = _mm256_fmadd_ps(vx1, vy1, vsum1);
        }
        vsum0 = _mm256_add_ps(vsum0, vsum1);
        for (; i + 7 < n; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = _mm256_loadu_ps(y + i);
            vsum0 = _mm256_fmadd_ps(vx, vy, vsum0);
        }
        sum = _hsum256(vsum0);
        for (; i < n; i++) sum += x[i] * y[i];
        return sum;
    }
#endif
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i * incx] * y[i * incy];
    return sum;
}

static inline void micro_sgemv(int m, int n, const float* A, int lda, const float* x, float* y) {
    for (int i = 0; i < m; i++) y[i] = 0.0f;
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        const float* row = A + i * lda;
#ifdef __AVX2__
        __m256 vsum0 = _mm256_setzero_ps();
        __m256 vsum1 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 15 < n; j += 16) {
            __m256 vx0 = _mm256_loadu_ps(x + j);
            __m256 vA0 = _mm256_loadu_ps(row + j);
            __m256 vx1 = _mm256_loadu_ps(x + j + 8);
            __m256 vA1 = _mm256_loadu_ps(row + j + 8);
            vsum0 = _mm256_fmadd_ps(vx0, vA0, vsum0);
            vsum1 = _mm256_fmadd_ps(vx1, vA1, vsum1);
        }
        vsum0 = _mm256_add_ps(vsum0, vsum1);
        for (; j + 7 < n; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vA = _mm256_loadu_ps(row + j);
            vsum0 = _mm256_fmadd_ps(vx, vA, vsum0);
        }
        sum = _hsum256(vsum0);
        for (; j < n; j++) sum += row[j] * x[j];
#else
        for (int j = 0; j < n; j++) sum += row[j] * x[j];
#endif
        y[i] = sum;
    }
}

static inline void micro_sgemv_trans(int m, int n, const float* A, int lda, const float* x, float* y) {
    for (int j = 0; j < n; j++) y[j] = 0.0f;
    for (int i = 0; i < m; i++) {
        float xi = x[i];
        const float* row = A + i * lda;
#ifdef __AVX2__
        __m256 vxi = _mm256_set1_ps(xi);
        int j = 0;
        for (; j + 15 < n; j += 16) {
            __m256 vA0 = _mm256_loadu_ps(row + j);
            __m256 vA1 = _mm256_loadu_ps(row + j + 8);
            __m256 vy0 = _mm256_loadu_ps(y + j);
            __m256 vy1 = _mm256_loadu_ps(y + j + 8);
            _mm256_storeu_ps(y + j, _mm256_fmadd_ps(vA0, vxi, vy0));
            _mm256_storeu_ps(y + j + 8, _mm256_fmadd_ps(vA1, vxi, vy1));
        }
        for (; j + 7 < n; j += 8) {
            __m256 vA = _mm256_loadu_ps(row + j);
            __m256 vy = _mm256_loadu_ps(y + j);
            _mm256_storeu_ps(y + j, _mm256_fmadd_ps(vA, vxi, vy));
        }
        for (; j < n; j++) y[j] += row[j] * xi;
#else
        for (int j = 0; j < n; j++) y[j] += row[j] * xi;
#endif
    }
}

static inline void micro_sgemm(int m, int n, int k, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) C[i * ldc + j] = 0.0f;
        for (int p = 0; p < k; p++) {
            float a_ip = A[i * lda + p];
            const float* B_row = B + p * ldb;
            float* C_row = C + i * ldc;
#ifdef __AVX2__
            __m256 va = _mm256_set1_ps(a_ip);
            int j = 0;
            for (; j + 15 < n; j += 16) {
                __m256 vb0 = _mm256_loadu_ps(B_row + j);
                __m256 vb1 = _mm256_loadu_ps(B_row + j + 8);
                __m256 vc0 = _mm256_loadu_ps(C_row + j);
                __m256 vc1 = _mm256_loadu_ps(C_row + j + 8);
                _mm256_storeu_ps(C_row + j, _mm256_fmadd_ps(va, vb0, vc0));
                _mm256_storeu_ps(C_row + j + 8, _mm256_fmadd_ps(va, vb1, vc1));
            }
            for (; j + 7 < n; j += 8) {
                __m256 vb = _mm256_loadu_ps(B_row + j);
                __m256 vc = _mm256_loadu_ps(C_row + j);
                _mm256_storeu_ps(C_row + j, _mm256_fmadd_ps(va, vb, vc));
            }
            for (; j < n; j++) C_row[j] += a_ip * B_row[j];
#else
            for (int j = 0; j < n; j++) C_row[j] += a_ip * B_row[j];
#endif
        }
    }
}

static inline void micro_sgemm_transB(int m, int n, int k, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            const float* a_row = A + i * lda;
            const float* b_row = B + j * ldb;
#ifdef __AVX2__
            __m256 vsum0 = _mm256_setzero_ps();
            __m256 vsum1 = _mm256_setzero_ps();
            int p = 0;
            for (; p + 15 < k; p += 16) {
                __m256 va0 = _mm256_loadu_ps(a_row + p);
                __m256 vb0 = _mm256_loadu_ps(b_row + p);
                __m256 va1 = _mm256_loadu_ps(a_row + p + 8);
                __m256 vb1 = _mm256_loadu_ps(b_row + p + 8);
                vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
                vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
            }
            vsum0 = _mm256_add_ps(vsum0, vsum1);
            for (; p + 7 < k; p += 8) {
                __m256 va = _mm256_loadu_ps(a_row + p);
                __m256 vb = _mm256_loadu_ps(b_row + p);
                vsum0 = _mm256_fmadd_ps(va, vb, vsum0);
            }
            sum = _hsum256(vsum0);
            for (; p < k; p++) sum += a_row[p] * b_row[p];
#else
            for (int p = 0; p < k; p++) sum += a_row[p] * b_row[p];
#endif
            C[i * ldc + j] = sum;
        }
    }
}

float tensor_dot(Tensor* A, Tensor* B) {
    if(A->ndim != 1 || B->ndim != 1 || A->total_size != B->total_size || A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32) {
        TENSOR_ERROR_VAL(0.0f, "FATAL [Dot]: Must be 1D FLOAT32 tensors of identical length.");
    }
    if (A->total_size < 2048) {
        return micro_sdot(A->total_size, F32(A), A->stride[0], F32(B), B->stride[0]);
    }
    return cblas_sdot(A->total_size, F32(A), A->stride[0], F32(B), B->stride[0]);
}

/* Sum of squared elements — equivalent to dot(v,v) but works on any shape.
 * Used for gradient global-norm without allocating an intermediate tensor. */
float tensor_sum_squares(Tensor* A) {
    if (!A || A->dtype != DTYPE_FLOAT32) {
        tensor_set_error("FATAL [SumSquares]: NULL or non-FLOAT32 tensor.");
        return 0.0f;
    }
    size_t n = A->total_size;
    const float* p = F32(A);
    if (tensor_is_contiguous(A) && n <= (size_t)INT_MAX) {
        return cblas_sdot((int)n, p, 1, p, 1);
    }
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (size_t i = 0; i < n; i++) s += p[i] * p[i];
    return s;
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
        if (m <= 128 && k <= 128) {
            micro_sgemv(m, k, F32(a_c), k, F32(b_c), F32(out));
        } else {
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        m, k, 1.0f,
                        F32(a_c), k,    /* lda = k for row-major [m,k] */
                        F32(b_c), 1,    /* incx = 1 for contiguous [k,1] */
                        0.0f, F32(out), 1);
        }
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
        if (n <= 128 && k <= 128) {
            micro_sgemv_trans(n, k, F32(b_c), n, F32(a_c), F32(out));
        } else {
            /* sgemv(Trans, rows=k, cols=n): out[j] = sum_i B[i,j]*x[i]  ==  x*B */
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        k, n, 1.0f,
                        F32(b_c), n,
                        F32(a_c), 1,
                        0.0f, F32(out), 1);
        }
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

    if (m <= 128 && n <= 128 && k <= 128 && transA == CblasNoTrans) {
        if (transB == CblasNoTrans) {
            micro_sgemm(m, n, k, F32(a_work), lda, F32(b_work), ldb, F32(out), n);
        } else if (transB == CblasTrans) {
            micro_sgemm_transB(m, n, k, F32(a_work), lda, F32(b_work), ldb, F32(out), n);
        } else {
            cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, 1.0f, F32(a_work), lda, F32(b_work), ldb, 0.0f, F32(out), n);
        }
    } else {
        cblas_sgemm(CblasRowMajor, transA, transB,
                    m, n, k, 1.0f,
                    F32(a_work), lda,
                    F32(b_work), ldb,
                    0.0f, F32(out), n);
    }

    if (a_work != A) tensor_free(a_work);
    if (b_work != B) tensor_free(b_work);
    return out;
}

/*
 * tensor_bmm — batched matrix multiply: A[batch,m,k] × B[batch,k,n] → [batch,m,n]
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

    Tensor* out = tensor_create_uninitialized(3, (int[]){batch, m, n}, DTYPE_FLOAT32);

    Tensor* a_work = tensor_is_contiguous(A) ? A : tensor_copy(A);
    Tensor* b_work = tensor_is_contiguous(B) ? B : tensor_copy(B);

    bool parallel = (batch > 1) && ((size_t)m * n * k > 10000);

    if (parallel && openblas_set_num_threads)
        openblas_set_num_threads(1); /* prevent BLAS×OMP oversubscription */

    #pragma omp parallel for schedule(dynamic) if(parallel)
    for (int b = 0; b < batch; b++) {
        if (m <= 128 && n <= 128 && k <= 128) {
            micro_sgemm(m, n, k,
                        F32(a_work) + b * m * k, k,
                        F32(b_work) + b * k * n, n,
                        F32(out)    + b * m * n, n);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, 1.0f,
                        F32(a_work) + b * m * k, k,
                        F32(b_work) + b * k * n, n,
                        0.0f,
                        F32(out)    + b * m * n, n);
        }
    }

    if (parallel && openblas_set_num_threads)
        openblas_set_num_threads(omp_get_max_threads());

    if (a_work != A) tensor_free(a_work);
    if (b_work != B) tensor_free(b_work);
    return out;
}

Tensor* tensor_matmul_ex(Tensor* A, Tensor* B, bool transA, bool transB) {
    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32)
        TENSOR_ERROR("FATAL [MatmulEx]: Requires FLOAT32.");
    if (A->ndim != 2 || B->ndim != 2)
        TENSOR_ERROR("FATAL [MatmulEx]: Both inputs must be 2D.");

    int m  = transA ? A->shape[1] : A->shape[0];
    int k  = transA ? A->shape[0] : A->shape[1];
    int k2 = transB ? B->shape[1] : B->shape[0];
    int n  = transB ? B->shape[0] : B->shape[1];

    if (k != k2)
        TENSOR_ERROR("FATAL [MatmulEx]: Inner dims mismatch (%d vs %d).", k, k2);

    int lda = A->shape[1];
    int ldb = B->shape[1];

    Tensor* a_work = tensor_is_contiguous(A) ? A : tensor_copy(A);
    Tensor* b_work = tensor_is_contiguous(B) ? B : tensor_copy(B);

    Tensor* out = tensor_create_uninitialized(2, (int[]){m, n}, DTYPE_FLOAT32);
    if (!out) {
        if (a_work != A) tensor_free(a_work);
        if (b_work != B) tensor_free(b_work);
        return NULL;
    }

    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                m, n, k, 1.0f,
                F32(a_work), lda,
                F32(b_work), ldb,
                0.0f, F32(out), n);

    if (a_work != A) tensor_free(a_work);
    if (b_work != B) tensor_free(b_work);
    return out;
}

void tensor_matmul_into(Tensor* out, Tensor* A, Tensor* B, bool transA, bool transB) {
    if (!out || !A || !B)
        TENSOR_ERROR_VOID("FATAL [MatmulInto]: NULL pointer.");
    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        TENSOR_ERROR_VOID("FATAL [MatmulInto]: Requires FLOAT32.");
    if (A->ndim != 2 || B->ndim != 2 || out->ndim != 2)
        TENSOR_ERROR_VOID("FATAL [MatmulInto]: All tensors must be 2D.");
    if (!tensor_is_contiguous(out))
        TENSOR_ERROR_VOID("FATAL [MatmulInto]: Output tensor must be contiguous.");

    int m  = transA ? A->shape[1] : A->shape[0];
    int k  = transA ? A->shape[0] : A->shape[1];
    int k2 = transB ? B->shape[1] : B->shape[0];
    int n  = transB ? B->shape[0] : B->shape[1];

    if (k != k2)
        TENSOR_ERROR_VOID("FATAL [MatmulInto]: Inner dims mismatch (%d vs %d).", k, k2);
    if (out->shape[0] != m || out->shape[1] != n)
        TENSOR_ERROR_VOID("FATAL [MatmulInto]: Output shape [%d,%d] ≠ expected [%d,%d].",
                          out->shape[0], out->shape[1], m, n);

    int lda = A->shape[1];
    int ldb = B->shape[1];

    Tensor* a_work = tensor_is_contiguous(A) ? A : tensor_copy(A);
    Tensor* b_work = tensor_is_contiguous(B) ? B : tensor_copy(B);

    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                m, n, k, 1.0f,
                F32(a_work), lda,
                F32(b_work), ldb,
                0.0f, F32(out), n);

    if (a_work != A) tensor_free(a_work);
    if (b_work != B) tensor_free(b_work);
}

void tensor_sum_axis_into(Tensor* out, Tensor* A, int axis) {
    if (!out || !A)
        TENSOR_ERROR_VOID("FATAL [SumAxisInto]: NULL pointer.");
    if (A->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        TENSOR_ERROR_VOID("FATAL [SumAxisInto]: Requires FLOAT32.");
    if (axis < 0 || axis >= A->ndim)
        TENSOR_ERROR_VOID("FATAL [SumAxisInto]: Invalid axis %d for ndim=%d.", axis, A->ndim);

    memset(F32(out), 0, out->byte_size);

    /* Fast path: 2-D contiguous, axis=0 */
    if (A->ndim == 2 && axis == 0 && tensor_is_contiguous(A) && tensor_is_contiguous(out)) {
        int rows = A->shape[0];
        int cols = A->shape[1];
        if (out->total_size != (size_t)cols)
            TENSOR_ERROR_VOID("FATAL [SumAxisInto]: Output size %zu ≠ %d (cols).",
                              out->total_size, cols);

        float* ones = (float*)malloc(rows * sizeof(float));
        if (!ones) TENSOR_ERROR_VOID("FATAL [SumAxisInto]: malloc failed.");
        for (int i = 0; i < rows; i++) ones[i] = 1.0f;

        cblas_sgemv(CblasRowMajor, CblasTrans,
                    rows, cols, 1.0f,
                    F32(A), cols, ones, 1,
                    0.0f, F32(out), 1);
        free(ones);
        return;
    }

    /* Generic path */
    int idx[8] = {0};
    for (size_t i = 0; i < A->total_size; i++) {
        size_t offset_a = 0, offset_out = 0;
        int out_d = 0;
        for (int d = 0; d < A->ndim; d++) {
            offset_a += idx[d] * A->stride[d];
            if (d != axis) offset_out += idx[d] * out->stride[out_d++];
        }
        F32(out)[offset_out] += F32(A)[offset_a];
        for (int d = A->ndim - 1; d >= 0; d--) {
            idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
        }
    }
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

/* Economy (thin) SVD — jobz='S':
 *   U  = [m × min_mn]  instead of [m × m]
 *   Vt = [min_mn × n]  instead of [n × n]
 *   S  = [min_mn]      (identical to full SVD)
 *
 * Avoids the O(m²) U allocation that causes SIGSEGV on tall matrices
 * (e.g. [100k × 2879] → U=[100k×100k] = 40 GB with full SVD).
 * Safe for TruncatedSVD callers that only read the top-k rows of Vt.
 * tensor_svd() is unchanged to preserve pinv(), tests, and all other callers. */
void tensor_svd_economy(Tensor* A, Tensor** U_out, Tensor** S_out, Tensor** Vt_out) {
    if(A->ndim != 2 || A->dtype != DTYPE_FLOAT32) TENSOR_ERROR_VOID("FATAL [SVD_ECO]: Must be 2D FLOAT32.");
    int m = A->shape[0]; int n = A->shape[1]; int min_mn = MIN(m, n);
    Tensor* S   = tensor_create(1, &min_mn);
    Tensor* U   = tensor_create(2, (int[]){m, min_mn});        /* [m × min_mn] */
    Tensor* Vt  = tensor_create(2, (int[]){min_mn, n});        /* [min_mn × n] */
    Tensor* copyA = tensor_copy(A);

    /* ldu = min_mn  (columns per row of U in row-major)
     * ldvt = n      (columns per row of Vt in row-major) */
    int info = LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'S', m, n, F32(copyA), n,
                              F32(S), F32(U), min_mn, F32(Vt), n);
    if (info != 0) {
        tensor_free(S); tensor_free(U); tensor_free(Vt); tensor_free(copyA);
        TENSOR_ERROR_VOID("FATAL [SVD_ECO]: SVD convergence failed.");
    }
    tensor_free(copyA);

    if(U_out)  *U_out  = U;  else tensor_free(U);
    if(S_out)  *S_out  = S;  else tensor_free(S);
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
        const float* data = F32(A);
        size_t n = A->total_size;
        size_t i = 0;
#ifdef __AVX2__
        __m256 veps  = _mm256_set1_ps(1e-8f);
        __m256 vsign = _mm256_set1_ps(-0.0f);
        size_t limit = n & ~7UL;
        for (; i < limit; i += 8) {
            __m256 v    = _mm256_loadu_ps(data + i);
            __m256 vabs = _mm256_andnot_ps(vsign, v);
            __m256 cmp  = _mm256_cmp_ps(vabs, veps, _CMP_GT_OQ);
            if (_mm256_movemask_ps(cmp)) return true;
        }
#endif
        for (; i < n; i++) if (fabsf(data[i]) > 1e-8f) return true;
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
        const float* data = F32(A);
        size_t n = A->total_size;
        size_t i = 0;
#ifdef __AVX2__
        __m256 veps  = _mm256_set1_ps(1e-8f);
        __m256 vsign = _mm256_set1_ps(-0.0f);
        size_t limit = n & ~7UL;
        for (; i < limit; i += 8) {
            __m256 v    = _mm256_loadu_ps(data + i);
            __m256 vabs = _mm256_andnot_ps(vsign, v);
            __m256 cmp  = _mm256_cmp_ps(vabs, veps, _CMP_LE_OQ);
            if (_mm256_movemask_ps(cmp)) return false;
        }
#endif
        for (; i < n; i++) if (fabsf(data[i]) <= 1e-8f) return false;
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

Tensor* tensor_isnan(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32);
    if (tensor_is_contiguous(A)) {
        size_t n = A->total_size;
        float* a = F32(A); float* o = F32(out);
        size_t i = 0;
#ifdef __AVX2__
        __m256 vone = _mm256_set1_ps(1.0f);
        size_t limit = n & ~7UL;
        for (; i < limit; i += 8) {
            __m256 v = _mm256_loadu_ps(a + i);
            __m256 cmp = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
            _mm256_storeu_ps(o + i, _mm256_and_ps(cmp, vone));
        }
#endif
        for (; i < n; i++) o[i] = _f32_is_nan(a[i]) ? 1.0f : 0.0f;
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < out->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            F32(out)[i] = _f32_is_nan(F32(A)[offset]) ? 1.0f : 0.0f;
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return out;
}

Tensor* tensor_isinf(Tensor* A) {
    if (A->dtype != DTYPE_FLOAT32) TENSOR_ERROR("Requires FLOAT32.");
    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32);
    if (tensor_is_contiguous(A)) {
        size_t n = A->total_size;
        float* a = F32(A); float* o = F32(out);
        size_t i = 0;
#ifdef __AVX2__
        __m256 vinf  = _mm256_set1_ps(INFINITY);
        __m256 vsign = _mm256_set1_ps(-0.0f);
        __m256 vone  = _mm256_set1_ps(1.0f);
        size_t limit = n & ~7UL;
        for (; i < limit; i += 8) {
            __m256 v    = _mm256_loadu_ps(a + i);
            __m256 vabs = _mm256_andnot_ps(vsign, v);
            __m256 cmp  = _mm256_cmp_ps(vabs, vinf, _CMP_EQ_OQ);
            _mm256_storeu_ps(o + i, _mm256_and_ps(cmp, vone));
        }
#endif
        for (; i < n; i++) o[i] = isinf(a[i]) ? 1.0f : 0.0f;
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < out->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            F32(out)[i] = isinf(F32(A)[offset]) ? 1.0f : 0.0f;
            for (int d = A->ndim - 1; d >= 0; d--) {
                idx[d]++; if (idx[d] < A->shape[d]) break; idx[d] = 0;
            }
        }
    }
    return out;
}

void tensor_nan_to_num_inplace(Tensor* A, float nan_val, float posinf_val, float neginf_val) {
    if (A->dtype != DTYPE_FLOAT32) return;
    if (tensor_is_contiguous(A)) {
        float* data = F32(A);
        size_t n = A->total_size;
        size_t i = 0;
#ifdef __AVX2__
        __m256 vnan     = _mm256_set1_ps(nan_val);
        __m256 vposinf  = _mm256_set1_ps(posinf_val);
        __m256 vneginf  = _mm256_set1_ps(neginf_val);
        __m256 vinf     = _mm256_set1_ps(INFINITY);
        __m256 vsign    = _mm256_set1_ps(-0.0f);
        __m256 vzero    = _mm256_setzero_ps();
        size_t limit = n & ~7UL;
        for (; i < limit; i += 8) {
            __m256 v    = _mm256_loadu_ps(data + i);
            __m256 vabs = _mm256_andnot_ps(vsign, v);
            __m256 nan_mask  = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
            __m256 inf_mask  = _mm256_cmp_ps(vabs, vinf, _CMP_EQ_OQ);
            __m256 pos_mask  = _mm256_and_ps(inf_mask, _mm256_cmp_ps(v, vzero, _CMP_GT_OQ));
            __m256 neg_mask  = _mm256_andnot_ps(pos_mask, inf_mask);
            __m256 result = _mm256_blendv_ps(v,       vneginf, neg_mask);
            result        = _mm256_blendv_ps(result,  vposinf, pos_mask);
            result        = _mm256_blendv_ps(result,  vnan,    nan_mask);
            _mm256_storeu_ps(data + i, result);
        }
#endif
        for (; i < n; i++) {
            if (_f32_is_nan(data[i])) data[i] = nan_val;
            else if (isinf(data[i])) data[i] = data[i] > 0 ? posinf_val : neginf_val;
        }
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < A->total_size; i++) {
            size_t offset = 0;
            for (int d = 0; d < A->ndim; d++) offset += idx[d] * A->stride[d];
            if (_f32_is_nan(F32(A)[offset])) F32(A)[offset] = nan_val;
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

    Tensor* out = tensor_create(2, (int[]){rows, cols});

    size_t as0 = A->stride[0];
    size_t as1 = A->stride[1];
    size_t as2 = A->stride[2];

    #pragma omp parallel for schedule(static) if(rows > 4096)
    for (int b = 0; b < batch; b++) {
        int base_row = b * (out_h * out_w);

        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int   row_idx = base_row + oh * out_w + ow;
                float* out_row = F32(out) + (size_t)row_idx * cols;

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
    int out_w = (width  + 2 * pad_w - kernel_w) / stride_w + 1;
    int cols  = channels * kernel_h * kernel_w;

    size_t os0 = out->stride[0];
    size_t os1 = out->stride[1];
    size_t os2 = out->stride[2];

    for (int b = 0; b < batch; b++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int row_idx = b * (out_h * out_w) + oh * out_w + ow;
                const float* col_row = F32(cols_tensor) + (size_t)row_idx * cols;

                int col_idx = 0;
                for (int c = 0; c < channels; c++) {
                    for (int kh_i = 0; kh_i < kernel_h; kh_i++) {
                        int im_h = oh * stride_h - pad_h + kh_i;
                        bool h_valid = (im_h >= 0 && im_h < height);
                        for (int kw_i = 0; kw_i < kernel_w; kw_i++, col_idx++) {
                            int im_w = ow * stride_w - pad_w + kw_i;
                            if (h_valid && im_w >= 0 && im_w < width) {
                                F32(out)[b * os0 + c * os1 + im_h * os2 + im_w]
                                    += col_row[col_idx];
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
        float v_hat_sqrt = sqrtf(fmaxf(0.0f, F32(v)[i] / bias_correction2)) + eps;
        F32(param)[i] -= step_size * (F32(m)[i] / v_hat_sqrt);
    }
}

// ── Fused SGD step ────────────────────────────────────────────────────────────
// param -= lr * grad  (no momentum, no state)
void tensor_fused_sgd_step(Tensor* param, Tensor* grad, float lr) {
    if (!_tensor_shape_assert(param, grad, "fused_sgd")) return;
    if (!tensor_is_contiguous(param) || !tensor_is_contiguous(grad))
        TENSOR_ERROR_VOID("FATAL [Fused SGD]: Tensors must be contiguous.");

    float* p = F32(param);
    const float* g = F32(grad);
    int n = (int)param->total_size;

    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < n; i++) p[i] -= lr * g[i];
}

// ── Fused RMSProp step ────────────────────────────────────────────────────────
// cache = decay * cache + (1 - decay) * g^2
// param -= lr * g / (sqrt(cache) + eps)
void tensor_fused_rmsprop_step(Tensor* param, Tensor* grad, Tensor* cache,
                                float lr, float decay, float eps) {
    if (!_tensor_shape_assert(param, grad, "fused_rmsprop") ||
        !_tensor_shape_assert(param, cache, "fused_rmsprop")) return;
    if (!tensor_is_contiguous(param) || !tensor_is_contiguous(grad) ||
        !tensor_is_contiguous(cache))
        TENSOR_ERROR_VOID("FATAL [Fused RMSProp]: Tensors must be contiguous.");

    float* p = F32(param);
    const float* g = F32(grad);
    float* c = F32(cache);
    int n = (int)param->total_size;
    float one_minus_decay = 1.0f - decay;

    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < n; i++) {
        c[i] = decay * c[i] + one_minus_decay * g[i] * g[i];
        p[i] -= lr * g[i] / (sqrtf(c[i]) + eps);
    }
}

// ── Fused AdaGrad step ────────────────────────────────────────────────────────
// acc += g^2
// param -= lr * g / (sqrt(acc) + eps)
void tensor_fused_adagrad_step(Tensor* param, Tensor* grad, Tensor* acc,
                                float lr, float eps) {
    if (!_tensor_shape_assert(param, grad, "fused_adagrad") ||
        !_tensor_shape_assert(param, acc, "fused_adagrad")) return;
    if (!tensor_is_contiguous(param) || !tensor_is_contiguous(grad) ||
        !tensor_is_contiguous(acc))
        TENSOR_ERROR_VOID("FATAL [Fused AdaGrad]: Tensors must be contiguous.");

    float* p = F32(param);
    const float* g = F32(grad);
    float* a = F32(acc);
    int n = (int)param->total_size;

    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < n; i++) {
        a[i] += g[i] * g[i];
        p[i] -= lr * g[i] / (sqrtf(a[i]) + eps);
    }
}

// ── Fused AdamW step ──────────────────────────────────────────────────────────
// Weight decay applied directly to weights (decoupled from gradient):
//   param *= (1 - lr * wd)
//   m = b1*m + (1-b1)*g
//   v = b2*v + (1-b2)*g^2
//   param -= step_size * m_hat / (sqrt(v_hat) + eps)
void tensor_fused_adamw_step(Tensor* param, Tensor* grad, Tensor* m, Tensor* v,
                              float lr, float b1, float b2, float eps, int t, float wd) {
    if (t <= 0)
        TENSOR_ERROR_VOID("FATAL [Fused AdamW]: Step t must be >= 1, got %d.", t);
    if (!_tensor_shape_assert(param, grad, "fused_adamw") ||
        !_tensor_shape_assert(param, m, "fused_adamw") ||
        !_tensor_shape_assert(param, v, "fused_adamw")) return;
    if (!tensor_is_contiguous(param) || !tensor_is_contiguous(grad) ||
        !tensor_is_contiguous(m)     || !tensor_is_contiguous(v))
        TENSOR_ERROR_VOID("FATAL [Fused AdamW]: Tensors must be contiguous.");

    float bias_correction1 = 1.0f - powf(b1, (float)t);
    float bias_correction2 = 1.0f - powf(b2, (float)t);
    float step_size = lr / bias_correction1;
    float decay_factor = 1.0f - lr * wd;
    int n = (int)param->total_size;

    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < n; i++) {
        float g = F32(grad)[i];
        F32(param)[i] *= decay_factor;
        F32(m)[i] = b1 * F32(m)[i] + (1.0f - b1) * g;
        F32(v)[i] = b2 * F32(v)[i] + (1.0f - b2) * g * g;
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

Tensor* tensor_linear(Tensor* X, Tensor* W, Tensor* bias) {
    if (X->dtype != DTYPE_FLOAT32 || W->dtype != DTYPE_FLOAT32)
        TENSOR_ERROR("FATAL [Linear]: Requires FLOAT32.");
    if (W->ndim != 2 || X->shape[X->ndim - 1] != W->shape[1])
        TENSOR_ERROR("FATAL [Linear]: W must be 2D [n,k] with k == X last dim.");

    int k = X->shape[X->ndim - 1];
    int m = (int)(X->total_size / k);
    int n = W->shape[0];

    Tensor* x_work = tensor_is_contiguous(X) ? X : tensor_copy(X);
    Tensor* w_work = tensor_is_contiguous(W) ? W : tensor_copy(W);

    int out_shape[8];
    for (int i = 0; i < X->ndim - 1; i++) out_shape[i] = X->shape[i];
    out_shape[X->ndim - 1] = n;
    Tensor* out = tensor_create_uninitialized(X->ndim, out_shape, DTYPE_FLOAT32);

    if (m <= 128 && n <= 128 && k <= 128) {
        micro_sgemm_transB(m, n, k, F32(x_work), k, F32(w_work), k, F32(out), n);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, n, k, 1.0f,
                    F32(x_work), k,
                    F32(w_work), k,
                    0.0f, F32(out), n);
    }

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

extern void openblas_set_num_threads(int) __attribute__((weak));
extern void goto_set_num_threads(int)     __attribute__((weak));

void tensor_configure_threading(int omp_threads, int blas_threads) {
    if (omp_threads  > 0) omp_set_num_threads(omp_threads);
    if (blas_threads > 0) {
        if (openblas_set_num_threads) openblas_set_num_threads(blas_threads);
        else if (goto_set_num_threads) goto_set_num_threads(blas_threads);
    }
}

// ============================================================================
// 18. TRANSFORMER INFERENCE PRIMITIVES
// ============================================================================

/* fast_expf and fast_logf (Cephes based) */
static inline float fast_expf(float x) {
    if (__builtin_expect(x < -88.3762626647949f, 0)) return 0.0f;
    float fx = floorf(x * 1.44269504088896341f + 0.5f);
    x -= fx * 0.693359375f;
    x -= fx * (-2.12194440e-4f);
    float z = x * x;
    float y = 1.9875691500e-4f;
    y = y * x + 1.3981999507e-3f;
    y = y * x + 8.3334519073e-3f;
    y = y * x + 4.1665795894e-2f;
    y = y * x + 1.6666665459e-1f;
    y = y * x + 5.0000001201e-1f;
    y = y * z + x + 1.0f;
    int fi = (int)fx;
    int bits = (fi + 127) << 23;
    float pw2; memcpy(&pw2, &bits, sizeof pw2);
    return y * pw2;
}

static inline float fast_logf(float x) {
    if (__builtin_expect(x <= 0.0f, 0)) return -INFINITY;
    int e;
    float m = frexpf(x, &e);
    float m1 = m - 1.0f;
    float z = m1 / (m + 1.0f);
    float z2 = z * z;
    float y = 0.66666662693e-1f;
    y = y * z2 + 0.3999999940e-1f;
    y = y * z2 + 0.2857142874e-1f;
    y = y * z2 + 0.2222219846e-1f;
    y = y * z2 + 0.1818357216e-1f;
    y = y * z2 + 0.1531383766e-1f;
    y = y * z2 + 0.1479819861e-1f;
    y = y * z2 + 0.2222219846e-1f;
    return 2.0f * z * (1.0f + y) + (float)e * 0.6931471805599453f;
}

#ifdef __AVX512F__
/* Forward-declared here so softmax_rows can use it; defined again near Mamba section. */
static inline __m512 avx512_expf(__m512 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.3762626647949f));
    x = _mm512_min_ps(x, _mm512_set1_ps( 88.3762626647949f));
    __m512 fx = _mm512_floor_ps(
        _mm512_fmadd_ps(x, _mm512_set1_ps(1.44269504088896341f),
                           _mm512_set1_ps(0.5f)));
    x = _mm512_fnmadd_ps(fx, _mm512_set1_ps( 0.693359375f),    x);
    x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f),  x);
    __m512 z = _mm512_mul_ps(x, x);
    __m512 y = _mm512_set1_ps(1.9875691500e-4f);
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507e-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073e-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894e-2f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459e-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201e-1f));
    y = _mm512_fmadd_ps(y, z, _mm512_add_ps(x, _mm512_set1_ps(1.0f)));
    __m512i n = _mm512_add_epi32(_mm512_cvttps_epi32(fx), _mm512_set1_epi32(127));
    return _mm512_mul_ps(y, _mm512_castsi512_ps(_mm512_slli_epi32(n, 23)));
}
#endif

#ifdef __AVX2__
static inline __m256 avx2_expf(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
    x = _mm256_min_ps(x, _mm256_set1_ps( 88.3762626647949f));
    __m256 fx = _mm256_floor_ps(
        _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f),
                           _mm256_set1_ps(0.5f)));
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps( 0.693359375f),    x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f),  x);
    __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, z, _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
    __m256i n = _mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127));
    return _mm256_mul_ps(y, _mm256_castsi256_ps(_mm256_slli_epi32(n, 23)));
}
#endif

/* AVX2 fast sigmoid: 1/(1+exp(-x))  using avx2_expf */
#ifdef __AVX2__
static inline __m256 avx2_sigmoidf(__m256 x) {
    __m256 neg = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
    __m256 e   = avx2_expf(neg);
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, e));
}

/* AVX2 fast tanh: (e^2x - 1)/(e^2x + 1)  using avx2_expf */
static inline __m256 avx2_tanhf(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-9.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps( 9.0f));
    __m256 e2x = avx2_expf(_mm256_add_ps(x, x));
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(_mm256_sub_ps(e2x, one),
                         _mm256_add_ps(e2x, one));
}
#endif

static inline float dot_f32(const float* __restrict a,
                            const float* __restrict b, int n) {
    int i = 0;
    float sum = 0.0f;
#ifdef __AVX512F__
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    for (; i <= n - 32; i += 32) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i),    _mm512_loadu_ps(b+i),    s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16), s1);
    }
    s0 = _mm512_add_ps(s0, s1);
    for (; i <= n - 16; i += 16)
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i), _mm512_loadu_ps(b+i), s0);
    sum = _mm512_reduce_add_ps(s0);
#elif defined(__AVX2__)
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    for (; i <= n - 32; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i),    s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+ 8), _mm256_loadu_ps(b+i+ 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), s3);
    }
    s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    for (; i <= n - 8; i += 8)
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), s0);
    {
        __m128 lo = _mm256_castps256_ps128(s0);
        __m128 hi = _mm256_extractf128_ps(s0, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum = _mm_cvtss_f32(lo);
    }
#endif
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static void softmax_rows(float* __restrict data, int rows, int cols) {
#pragma omp parallel for schedule(static, 16) if(rows > 64)
    for (int r = 0; r < rows; r++) {
        float* __restrict row = data + (size_t)r * cols;

        float vmax = -FLT_MAX;
        int j = 0;
#ifdef __AVX512F__
        {
            __m512 mx0 = _mm512_set1_ps(-FLT_MAX), mx1 = mx0;
            for (; j <= cols - 32; j += 32) {
                mx0 = _mm512_max_ps(mx0, _mm512_loadu_ps(row + j));
                mx1 = _mm512_max_ps(mx1, _mm512_loadu_ps(row + j + 16));
            }
            mx0 = _mm512_max_ps(mx0, mx1);
            for (; j <= cols - 16; j += 16)
                mx0 = _mm512_max_ps(mx0, _mm512_loadu_ps(row + j));
            vmax = _mm512_reduce_max_ps(mx0);
        }
#elif defined(__AVX2__)
        {
            __m256 mx0 = _mm256_set1_ps(-FLT_MAX);
            __m256 mx1 = mx0, mx2 = mx0, mx3 = mx0;
            for (; j <= cols - 32; j += 32) {
                mx0 = _mm256_max_ps(mx0, _mm256_loadu_ps(row + j));
                mx1 = _mm256_max_ps(mx1, _mm256_loadu_ps(row + j +  8));
                mx2 = _mm256_max_ps(mx2, _mm256_loadu_ps(row + j + 16));
                mx3 = _mm256_max_ps(mx3, _mm256_loadu_ps(row + j + 24));
            }
            mx0 = _mm256_max_ps(_mm256_max_ps(mx0, mx1), _mm256_max_ps(mx2, mx3));
            for (; j <= cols - 8; j += 8)
                mx0 = _mm256_max_ps(mx0, _mm256_loadu_ps(row + j));
            __m128 lo = _mm256_castps256_ps128(mx0);
            __m128 hi = _mm256_extractf128_ps(mx0, 1);
            lo = _mm_max_ps(lo, hi);
            lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
            lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
            vmax = _mm_cvtss_f32(lo);
        }
#endif
        for (; j < cols; j++) if (row[j] > vmax) vmax = row[j];

        float sum = 0.0f;
        j = 0;
#ifdef __AVX512F__
        {
            __m512 vs0 = _mm512_setzero_ps(), vs1 = vs0;
            __m512 vm  = _mm512_set1_ps(vmax);
            for (; j <= cols - 32; j += 32) {
                __m512 e0 = avx512_expf(_mm512_sub_ps(_mm512_loadu_ps(row + j),      vm));
                __m512 e1 = avx512_expf(_mm512_sub_ps(_mm512_loadu_ps(row + j + 16), vm));
                vs0 = _mm512_add_ps(vs0, e0); vs1 = _mm512_add_ps(vs1, e1);
                _mm512_storeu_ps(row + j,      e0);
                _mm512_storeu_ps(row + j + 16, e1);
            }
            vs0 = _mm512_add_ps(vs0, vs1);
            for (; j <= cols - 16; j += 16) {
                __m512 e = avx512_expf(_mm512_sub_ps(_mm512_loadu_ps(row + j), vm));
                _mm512_storeu_ps(row + j, e);
                vs0 = _mm512_add_ps(vs0, e);
            }
            sum = _mm512_reduce_add_ps(vs0);
        }
#elif defined(__AVX2__)
        {
            __m256 vs0 = _mm256_setzero_ps(), vs1 = vs0, vs2 = vs0, vs3 = vs0;
            __m256 vm  = _mm256_set1_ps(vmax);
            for (; j <= cols - 32; j += 32) {
                __m256 e0 = avx2_expf(_mm256_sub_ps(_mm256_loadu_ps(row + j),      vm));
                __m256 e1 = avx2_expf(_mm256_sub_ps(_mm256_loadu_ps(row + j +  8), vm));
                __m256 e2 = avx2_expf(_mm256_sub_ps(_mm256_loadu_ps(row + j + 16), vm));
                __m256 e3 = avx2_expf(_mm256_sub_ps(_mm256_loadu_ps(row + j + 24), vm));
                _mm256_storeu_ps(row + j,      e0);
                _mm256_storeu_ps(row + j +  8, e1);
                _mm256_storeu_ps(row + j + 16, e2);
                _mm256_storeu_ps(row + j + 24, e3);
                vs0 = _mm256_add_ps(vs0, e0); vs1 = _mm256_add_ps(vs1, e1);
                vs2 = _mm256_add_ps(vs2, e2); vs3 = _mm256_add_ps(vs3, e3);
            }
            vs0 = _mm256_add_ps(_mm256_add_ps(vs0, vs1), _mm256_add_ps(vs2, vs3));
            for (; j <= cols - 8; j += 8) {
                __m256 e = avx2_expf(_mm256_sub_ps(_mm256_loadu_ps(row + j), vm));
                _mm256_storeu_ps(row + j, e);
                vs0 = _mm256_add_ps(vs0, e);
            }
            __m128 lo = _mm256_castps256_ps128(vs0);
            __m128 hi = _mm256_extractf128_ps(vs0, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            sum = _mm_cvtss_f32(lo);
        }
#endif
        for (; j < cols; j++) { float e = fast_expf(row[j] - vmax); row[j] = e; sum += e; }

        float inv_sum = 1.0f / sum;
        j = 0;
#ifdef __AVX512F__
        {
            __m512 vi = _mm512_set1_ps(inv_sum);
            for (; j <= cols - 32; j += 32) {
                _mm512_storeu_ps(row + j,      _mm512_mul_ps(_mm512_loadu_ps(row + j),      vi));
                _mm512_storeu_ps(row + j + 16, _mm512_mul_ps(_mm512_loadu_ps(row + j + 16), vi));
            }
            for (; j <= cols - 16; j += 16)
                _mm512_storeu_ps(row + j, _mm512_mul_ps(_mm512_loadu_ps(row + j), vi));
        }
#elif defined(__AVX2__)
        {
            __m256 vi = _mm256_set1_ps(inv_sum);
            for (; j <= cols - 32; j += 32) {
                _mm256_storeu_ps(row + j,      _mm256_mul_ps(_mm256_loadu_ps(row + j),      vi));
                _mm256_storeu_ps(row + j +  8, _mm256_mul_ps(_mm256_loadu_ps(row + j +  8), vi));
                _mm256_storeu_ps(row + j + 16, _mm256_mul_ps(_mm256_loadu_ps(row + j + 16), vi));
                _mm256_storeu_ps(row + j + 24, _mm256_mul_ps(_mm256_loadu_ps(row + j + 24), vi));
            }
            for (; j <= cols - 8; j += 8)
                _mm256_storeu_ps(row + j, _mm256_mul_ps(_mm256_loadu_ps(row + j), vi));
        }
#endif
        for (; j < cols; j++) row[j] *= inv_sum;
    }
}

void tensor_rmsnorm(Tensor* x, float eps) {
    if (!x || x->dtype != DTYPE_FLOAT32) {
        tensor_set_error("FATAL [RMSNorm]: Requires FLOAT32."); return;
    }
    if (!tensor_is_contiguous(x)) {
        tensor_set_error("FATAL [RMSNorm]: Input must be contiguous."); return;
    }

    int    ndim  = x->ndim;
    int    cols  = x->shape[ndim - 1];
    size_t rows  = x->total_size / (size_t)cols;
    float  inv_n = 1.0f / (float)cols;
    float* __restrict data = (float*)__builtin_assume_aligned(F32(x), 64);

#pragma omp parallel for schedule(static) if(rows > 64)
    for (size_t r = 0; r < rows; r++) {
        float* __restrict row = data + r * (size_t)cols;

        float ss = 0.0f;
        int j = 0;
#ifdef __AVX512F__
        {
            __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
            for (; j <= cols - 64; j += 64) {
                __m512 v0 = _mm512_loadu_ps(row + j);
                __m512 v1 = _mm512_loadu_ps(row + j + 16);
                __m512 v2 = _mm512_loadu_ps(row + j + 32);
                __m512 v3 = _mm512_loadu_ps(row + j + 48);
                a0 = _mm512_fmadd_ps(v0, v0, a0);
                a1 = _mm512_fmadd_ps(v1, v1, a1);
                a2 = _mm512_fmadd_ps(v2, v2, a2);
                a3 = _mm512_fmadd_ps(v3, v3, a3);
            }
            a0 = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
            for (; j <= cols - 16; j += 16) {
                __m512 v = _mm512_loadu_ps(row + j);
                a0 = _mm512_fmadd_ps(v, v, a0);
            }
            ss = _mm512_reduce_add_ps(a0);
        }
#elif defined(__AVX2__)
        {
            __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
            for (; j <= cols - 32; j += 32) {
                __m256 v0 = _mm256_loadu_ps(row + j);
                __m256 v1 = _mm256_loadu_ps(row + j +  8);
                __m256 v2 = _mm256_loadu_ps(row + j + 16);
                __m256 v3 = _mm256_loadu_ps(row + j + 24);
                a0 = _mm256_fmadd_ps(v0, v0, a0);
                a1 = _mm256_fmadd_ps(v1, v1, a1);
                a2 = _mm256_fmadd_ps(v2, v2, a2);
                a3 = _mm256_fmadd_ps(v3, v3, a3);
            }
            a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            for (; j <= cols - 8; j += 8) {
                __m256 v = _mm256_loadu_ps(row + j);
                a0 = _mm256_fmadd_ps(v, v, a0);
            }
            __m128 lo = _mm256_castps256_ps128(a0);
            __m128 hi = _mm256_extractf128_ps(a0, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            ss = _mm_cvtss_f32(lo);
        }
#endif
        for (; j < cols; j++) { float v = row[j]; ss += v * v; }

        float scale = 1.0f / sqrtf(ss * inv_n + eps);

        j = 0;
#ifdef __AVX512F__
        {
            __m512 vs = _mm512_set1_ps(scale);
            for (; j <= cols - 64; j += 64) {
                _mm512_storeu_ps(row + j,      _mm512_mul_ps(_mm512_loadu_ps(row + j),      vs));
                _mm512_storeu_ps(row + j + 16, _mm512_mul_ps(_mm512_loadu_ps(row + j + 16), vs));
                _mm512_storeu_ps(row + j + 32, _mm512_mul_ps(_mm512_loadu_ps(row + j + 32), vs));
                _mm512_storeu_ps(row + j + 48, _mm512_mul_ps(_mm512_loadu_ps(row + j + 48), vs));
            }
            for (; j <= cols - 16; j += 16)
                _mm512_storeu_ps(row + j, _mm512_mul_ps(_mm512_loadu_ps(row + j), vs));
        }
#elif defined(__AVX2__)
        {
            __m256 vs = _mm256_set1_ps(scale);
            for (; j <= cols - 32; j += 32) {
                _mm256_storeu_ps(row + j,      _mm256_mul_ps(_mm256_loadu_ps(row + j),      vs));
                _mm256_storeu_ps(row + j +  8, _mm256_mul_ps(_mm256_loadu_ps(row + j +  8), vs));
                _mm256_storeu_ps(row + j + 16, _mm256_mul_ps(_mm256_loadu_ps(row + j + 16), vs));
                _mm256_storeu_ps(row + j + 24, _mm256_mul_ps(_mm256_loadu_ps(row + j + 24), vs));
            }
            for (; j <= cols - 8; j += 8)
                _mm256_storeu_ps(row + j, _mm256_mul_ps(_mm256_loadu_ps(row + j), vs));
        }
#endif
        for (; j < cols; j++) row[j] *= scale;
    }
}

#define ROPE_MAX_HALF 128
static __thread float rope_freqs[ROPE_MAX_HALF];
static __thread int   rope_freq_hd   = -1;
static __thread float rope_freq_base = -1.0f;

static inline void rope_ensure_freqs(int head_dim, float base_freq) {
    if (__builtin_expect(head_dim == rope_freq_hd && base_freq == rope_freq_base, 1)) return;
    int half = head_dim / 2;
    for (int i = 0; i < half; i++)
        rope_freqs[i] = 1.0f / powf(base_freq, (2.0f * i) / (float)head_dim);
    rope_freq_hd   = head_dim;
    rope_freq_base = base_freq;
}

void tensor_apply_rope(Tensor* q, Tensor* k, int head_dim, int pos,
                       float base_freq, float scale) {
    if (!q || !k || q->dtype != DTYPE_FLOAT32 || k->dtype != DTYPE_FLOAT32) {
        tensor_set_error("FATAL [RoPE]: Requires FLOAT32."); return;
    }
    if (!tensor_is_contiguous(q) || !tensor_is_contiguous(k)) {
        tensor_set_error("FATAL [RoPE]: Inputs must be contiguous."); return;
    }
    if (head_dim <= 0 || head_dim % 2 != 0 || head_dim > ROPE_MAX_HALF * 2) {
        tensor_set_error("FATAL [RoPE]: head_dim must be positive, even, <= 256."); return;
    }
    if (base_freq <= 0.0f) {
        tensor_set_error("FATAL [RoPE]: base_freq must be > 0."); return;
    }

    rope_ensure_freqs(head_dim, base_freq);

    int half = head_dim / 2;
    float cos_buf[ROPE_MAX_HALF];
    float sin_buf[ROPE_MAX_HALF];
    float scaled_pos = (float)pos * scale;
    for (int i = 0; i < half; i++) {
        float angle = scaled_pos * rope_freqs[i];
        cos_buf[i] = cosf(angle);
        sin_buf[i] = sinf(angle);
    }

    {
        size_t nrows = q->total_size / (size_t)head_dim;
        float* __restrict d = F32(q);
        for (size_t r = 0; r < nrows; r++) {
            float* __restrict row = d + r * (size_t)head_dim;
            for (int i = 0; i < half; i++) {
                float c = cos_buf[i], s = sin_buf[i];
                float x0 = row[2*i], x1 = row[2*i+1];
                row[2*i]   = x0 * c - x1 * s;
                row[2*i+1] = x0 * s + x1 * c;
            }
        }
    }

    {
        size_t nrows = k->total_size / (size_t)head_dim;
        float* __restrict d = F32(k);
        for (size_t r = 0; r < nrows; r++) {
            float* __restrict row = d + r * (size_t)head_dim;
            for (int i = 0; i < half; i++) {
                float c = cos_buf[i], s = sin_buf[i];
                float x0 = row[2*i], x1 = row[2*i+1];
                row[2*i]   = x0 * c - x1 * s;
                row[2*i+1] = x0 * s + x1 * c;
            }
        }
    }
}

void tensor_softmax_inplace(Tensor* x) {
    if (!x || x->dtype != DTYPE_FLOAT32) {
        tensor_set_error("FATAL [SoftmaxInplace]: Requires FLOAT32."); return;
    }
    if (!tensor_is_contiguous(x)) {
        tensor_set_error("FATAL [SoftmaxInplace]: Input must be contiguous."); return;
    }
    int ndim = x->ndim;
    softmax_rows(F32(x), (int)(x->total_size / (size_t)x->shape[ndim-1]),
                         x->shape[ndim-1]);
}

void tensor_attention(Tensor* out, Tensor* q, Tensor* k, Tensor* v) {
    if (!out || !q || !k || !v) {
        tensor_set_error("FATAL [Attention]: NULL pointer."); return;
    }
    if (q->dtype  != DTYPE_FLOAT32 || k->dtype  != DTYPE_FLOAT32 ||
        v->dtype  != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32) {
        tensor_set_error("FATAL [Attention]: Requires FLOAT32."); return;
    }
    if (!tensor_is_contiguous(q) || !tensor_is_contiguous(k) ||
        !tensor_is_contiguous(v) || !tensor_is_contiguous(out)) {
        tensor_set_error("FATAL [Attention]: All tensors must be contiguous."); return;
    }
    if (q->ndim != 2 || k->ndim != 2 || v->ndim != 2 || out->ndim != 2) {
        tensor_set_error("FATAL [Attention]: Expects 2D [seq_len, head_dim]."); return;
    }

    int seq = q->shape[0];
    int hd  = q->shape[1];

    if (k->shape[0] != seq || k->shape[1] != hd ||
        v->shape[0] != seq || v->shape[1] != hd ||
        out->shape[0] != seq || out->shape[1] != hd) {
        tensor_set_error("FATAL [Attention]: Shape mismatch."); return;
    }
    if (__builtin_expect(seq == 0, 0)) return;

    const float* __restrict Q = (const float*)__builtin_assume_aligned(F32(q), 64);
    const float* __restrict K = (const float*)__builtin_assume_aligned(F32(k), 64);
    const float* __restrict V = (const float*)__builtin_assume_aligned(F32(v), 64);
    float*       __restrict O = (float*)__builtin_assume_aligned(F32(out), 64);
    float scale = 1.0f / sqrtf((float)hd);

#pragma omp parallel for schedule(static)
    for (int qi = 0; qi < seq; qi++) {
        const float* __restrict q_row = Q + (size_t)qi * hd;
        float*       __restrict o_row = O + (size_t)qi * hd;

        float s0 = dot_f32(q_row, K, hd) * scale;
        float m = s0, denom = 1.0f;
        memcpy(o_row, V, (size_t)hd * sizeof(float));

        for (int j = 1; j < seq; j++) {
            const float* __restrict kj = K + (size_t)j * hd;
            const float* __restrict vj = V + (size_t)j * hd;
            __builtin_prefetch(K + (size_t)(j + 2) * hd, 0, 1);
            __builtin_prefetch(V + (size_t)(j + 2) * hd, 0, 1);

            float s     = dot_f32(q_row, kj, hd) * scale;
            float m_new = s > m ? s : m;
            float alpha = fast_expf(m     - m_new);
            float beta  = fast_expf(s     - m_new);

            int d = 0;
#ifdef __AVX512F__
            {
                __m512 va = _mm512_set1_ps(alpha);
                __m512 vb = _mm512_set1_ps(beta);
                for (; d <= hd - 16; d += 16)
                    _mm512_storeu_ps(o_row + d,
                        _mm512_fmadd_ps(va, _mm512_loadu_ps(o_row + d),
                                            _mm512_mul_ps(vb, _mm512_loadu_ps(vj + d))));
            }
#elif defined(__AVX2__)
            {
                __m256 va = _mm256_set1_ps(alpha);
                __m256 vb = _mm256_set1_ps(beta);
                for (; d <= hd - 8; d += 8)
                    _mm256_storeu_ps(o_row + d,
                        _mm256_fmadd_ps(va, _mm256_loadu_ps(o_row + d),
                                            _mm256_mul_ps(vb, _mm256_loadu_ps(vj + d))));
            }
#endif
            for (; d < hd; d++) o_row[d] = o_row[d] * alpha + beta * vj[d];

            denom = denom * alpha + beta;
            m     = m_new;
        }

        float inv_d = 1.0f / denom;
        int d = 0;
#ifdef __AVX512F__
        {
            __m512 vi = _mm512_set1_ps(inv_d);
            for (; d <= hd - 16; d += 16)
                _mm512_storeu_ps(o_row + d,
                    _mm512_mul_ps(_mm512_loadu_ps(o_row + d), vi));
        }
#elif defined(__AVX2__)
        {
            __m256 vi = _mm256_set1_ps(inv_d);
            for (; d <= hd - 8; d += 8)
                _mm256_storeu_ps(o_row + d,
                    _mm256_mul_ps(_mm256_loadu_ps(o_row + d), vi));
        }
#endif
        for (; d < hd; d++) o_row[d] *= inv_d;
    }
}

// ============================================================================
// 19. KV CACHE — interleaved layout, zero-copy streaming attention
// ============================================================================

KVCache* kvcache_create(int cap, int head_dim) {
    if (cap <= 0 || head_dim <= 0) {
        tensor_set_error("FATAL [kvcache_create]: cap and head_dim must be > 0.");
        return NULL;
    }
    KVCache* c = (KVCache*)safe_malloc(sizeof(KVCache));
    if (!c) return NULL;
    c->data = (float*)safe_memalign(64, (size_t)cap * 2 * head_dim * sizeof(float));
    if (!c->data) { free(c); return NULL; }
    c->len      = 0;
    c->cap      = cap;
    c->head_dim = head_dim;
    return c;
}

void kvcache_free(KVCache* c) {
    if (!c) return;
    free(c->data);
    free(c);
}

void kvcache_reset(KVCache* c) {
    if (c) c->len = 0;
}

int kvcache_len(const KVCache* c) {
    return c ? c->len : 0;
}

void kvcache_append(KVCache* c, const Tensor* k, const Tensor* v) {
    if (!c || !k || !v) {
        tensor_set_error("FATAL [kvcache_append]: NULL pointer."); return;
    }
    if (k->dtype != DTYPE_FLOAT32 || v->dtype != DTYPE_FLOAT32) {
        tensor_set_error("FATAL [kvcache_append]: Requires FLOAT32."); return;
    }
    if (!tensor_is_contiguous(k) || !tensor_is_contiguous(v)) {
        tensor_set_error("FATAL [kvcache_append]: Inputs must be contiguous."); return;
    }

    int hd  = c->head_dim;
    int n   = (int)(k->total_size / (size_t)hd);

    if (c->len + n > c->cap) {
        tensor_set_error("FATAL [kvcache_append]: Cache capacity exceeded."); return;
    }

    const float* __restrict ks  = (const float*)__builtin_assume_aligned(F32(k), 64);
    const float* __restrict vs  = (const float*)__builtin_assume_aligned(F32(v), 64);
    float*       __restrict dst = (float*)__builtin_assume_aligned(c->data, 64) + (size_t)c->len * 2 * hd;

    for (int i = 0; i < n; i++) {
        memcpy(dst,      ks + (size_t)i * hd, (size_t)hd * sizeof(float));
        memcpy(dst + hd, vs + (size_t)i * hd, (size_t)hd * sizeof(float));
        dst += 2 * hd;
    }
    c->len += n;
}

void tensor_attention_kv(Tensor* out, Tensor* q, const KVCache* kvc) {
    if (!out || !q || !kvc) {
        tensor_set_error("FATAL [AttentionKV]: NULL pointer."); return;
    }
    if (q->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32) {
        tensor_set_error("FATAL [AttentionKV]: Requires FLOAT32."); return;
    }
    if (!tensor_is_contiguous(q) || !tensor_is_contiguous(out)) {
        tensor_set_error("FATAL [AttentionKV]: Tensors must be contiguous."); return;
    }

    int hd     = kvc->head_dim;
    int seq_q  = (int)(q->total_size  / (size_t)hd);
    int seq_kv = kvc->len;
    int stride = 2 * hd;

    if (q->shape[q->ndim - 1] != hd || out->shape[out->ndim - 1] != hd ||
        (int)(out->total_size / (size_t)hd) != seq_q) {
        tensor_set_error("FATAL [AttentionKV]: Shape mismatch."); return;
    }
    if (__builtin_expect(seq_kv == 0, 0)) {
        memset(F32(out), 0, (size_t)seq_q * hd * sizeof(float)); return;
    }

    float* __restrict       Q  = (float*)__builtin_assume_aligned(F32(q), 64);
    float* __restrict       O  = (float*)__builtin_assume_aligned(F32(out), 64);
    const float* __restrict KV = (const float*)__builtin_assume_aligned(kvc->data, 64);
    float scale = 1.0f / sqrtf((float)hd);

#pragma omp parallel for schedule(static)
    for (int qi = 0; qi < seq_q; qi++) {
        const float* __restrict q_row = Q + (size_t)qi * hd;
        float*       __restrict o_row = O + (size_t)qi * hd;

        const float* kv0 = KV;
        float s0 = dot_f32(q_row, kv0, hd) * scale;
        float m = s0, denom = 1.0f;
        memcpy(o_row, kv0 + hd, (size_t)hd * sizeof(float));

        for (int j = 1; j < seq_kv; j++) {
            const float* __restrict kvj = KV + (size_t)j * stride;

            __builtin_prefetch(kvj + stride,      0, 1);
            __builtin_prefetch(kvj + stride + 32, 0, 1);

            float s     = dot_f32(q_row, kvj, hd) * scale;
            float m_new = s > m ? s : m;
            float alpha = fast_expf(m - m_new);
            float beta  = fast_expf(s - m_new);

            const float* __restrict vj = kvj + hd;
            int d = 0;
#ifdef __AVX512F__
            {
                __m512 va = _mm512_set1_ps(alpha);
                __m512 vb = _mm512_set1_ps(beta);
                for (; d <= hd - 16; d += 16)
                    _mm512_storeu_ps(o_row + d,
                        _mm512_fmadd_ps(va, _mm512_loadu_ps(o_row + d),
                                            _mm512_mul_ps(vb, _mm512_loadu_ps(vj + d))));
            }
#elif defined(__AVX2__)
            {
                __m256 va = _mm256_set1_ps(alpha);
                __m256 vb = _mm256_set1_ps(beta);
                for (; d <= hd - 8; d += 8)
                    _mm256_storeu_ps(o_row + d,
                        _mm256_fmadd_ps(va, _mm256_loadu_ps(o_row + d),
                                            _mm256_mul_ps(vb, _mm256_loadu_ps(vj + d))));
            }
#endif
            for (; d < hd; d++) o_row[d] = o_row[d] * alpha + beta * vj[d];

            denom = denom * alpha + beta;
            m     = m_new;
        }

        float inv_d = 1.0f / denom;
        int d = 0;
#ifdef __AVX512F__
        {
            __m512 vi = _mm512_set1_ps(inv_d);
            for (; d <= hd - 16; d += 16)
                _mm512_storeu_ps(o_row + d,
                    _mm512_mul_ps(_mm512_loadu_ps(o_row + d), vi));
        }
#elif defined(__AVX2__)
        {
            __m256 vi = _mm256_set1_ps(inv_d);
            for (; d <= hd - 8; d += 8)
                _mm256_storeu_ps(o_row + d,
                    _mm256_mul_ps(_mm256_loadu_ps(o_row + d), vi));
        }
#endif
        for (; d < hd; d++) o_row[d] *= inv_d;
    }
}

void tensor_copy_from(Tensor* dest, Tensor* src) {
    if (dest->total_size != src->total_size || dest->total_size == 0) return;

    if (tensor_is_contiguous(dest) && tensor_is_contiguous(src)) {
        size_t bytes = dest->total_size * sizeof(float);
        
        if (dest->total_size > 500000) {
            float* __restrict d_ptr = (float*)__builtin_assume_aligned(F32(dest), 64);
            float* __restrict s_ptr = (float*)__builtin_assume_aligned(F32(src), 64);
            #pragma omp parallel for simd
            for (size_t i = 0; i < dest->total_size; i++) {
                d_ptr[i] = s_ptr[i];
            }
        } else {
            memcpy(F32(dest), F32(src), bytes);
        }
        return;
    }

    if (dest->ndim == 0) {
        F32(dest)[0] = F32(src)[0];
        return;
    }

    int ndim = dest->ndim;
    int* shape = dest->shape;
    int inner_len = shape[ndim - 1];

    if (tensor_is_contiguous(dest)) {
        int s_inner_stride = src->stride[ndim - 1];
        size_t total_outer = dest->total_size / inner_len;

        float* __restrict d_ptr = (float*)__builtin_assume_aligned(F32(dest), 64);
        float* s_base = (float*)__builtin_assume_aligned(F32(src), 64);
        int s_idx[8] = {0};

        for (size_t i = 0; i < total_outer; i++) {
            float* __restrict s_inner = s_base;

            if (s_inner_stride == 1) {
                memcpy(d_ptr, s_inner, inner_len * sizeof(float));
                d_ptr += inner_len;
            } else {
                int j = 0;
#ifdef __AVX2__
                __m256i v_indices = _mm256_set_epi32(
                    7 * s_inner_stride, 6 * s_inner_stride, 5 * s_inner_stride, 4 * s_inner_stride,
                    3 * s_inner_stride, 2 * s_inner_stride, 1 * s_inner_stride, 0
                );
                for (; j <= inner_len - 8; j += 8) {
                    __m256 v_val = _mm256_i32gather_ps(s_inner, v_indices, 4);
                    _mm256_storeu_ps(d_ptr, v_val);
                    s_inner += 8 * s_inner_stride;
                    d_ptr += 8;
                }
#endif
                for (; j < inner_len; j++) {
                    *d_ptr++ = *s_inner;
                    s_inner += s_inner_stride;
                }
            }

            for (int d = ndim - 2; d >= 0; d--) {
                s_idx[d]++;
                if (s_idx[d] < shape[d]) {
                    s_base += src->stride[d];
                    break;
                } else {
                    s_idx[d] = 0;
                    s_base -= src->stride[d] * (shape[d] - 1);
                }
            }
        }
        return;
    }

    int s_inner_stride = src->stride[ndim - 1];
    int d_inner_stride = dest->stride[ndim - 1];
    size_t total_outer = dest->total_size / inner_len;

    float* s_base = (float*)__builtin_assume_aligned(F32(src), 64);
    float* d_base = (float*)__builtin_assume_aligned(F32(dest), 64);
    int idx[8] = {0};

    for (size_t i = 0; i < total_outer; i++) {
        float* __restrict s_inner = s_base;
        float* __restrict d_inner = d_base;

        for (int j = 0; j < inner_len; j++) {
            *d_inner = *s_inner;
            d_inner += d_inner_stride;
            s_inner += s_inner_stride;
        }

        for (int d = ndim - 2; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < shape[d]) {
                s_base += src->stride[d];
                d_base += dest->stride[d];
                break;
            } else {
                idx[d] = 0;
                s_base -= src->stride[d] * (shape[d] - 1);
                d_base -= dest->stride[d] * (shape[d] - 1);
            }
        }
    }
}

// ============================================================================
// 20. ADVANCED INFERENCE & TRAINING PRIMITIVES
// ============================================================================

Tensor* tensor_from_mmap(const char* filepath, size_t byte_offset,
                         int ndim, const int* shape, int dtype_int) {
    if (!filepath || !shape || ndim <= 0 || ndim > 8) {
        tensor_set_error("FATAL [mmap]: Invalid arguments."); return NULL;
    }
    TensorDType dtype = (TensorDType)dtype_int;
    size_t elem_size  = dtype_size(dtype);

    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            tensor_set_error("FATAL [mmap]: shape element <= 0."); return NULL;
        }
        total *= (size_t)shape[i];
    }
    size_t map_bytes = total * elem_size;

    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        tensor_set_error("FATAL [mmap]: Cannot open file."); return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0 || (size_t)st.st_size < byte_offset + map_bytes) {
        tensor_set_error("FATAL [mmap]: File too small for requested region.");
        close(fd); return NULL;
    }

    long page_sz      = sysconf(_SC_PAGESIZE);
    size_t page_off   = byte_offset % (size_t)page_sz;
    size_t map_off    = byte_offset - page_off;
    size_t map_total  = map_bytes + page_off;

    void* mapped = mmap(NULL, map_total, PROT_READ, MAP_PRIVATE, fd, (off_t)map_off);
    close(fd);
    if (mapped == MAP_FAILED) {
        tensor_set_error("FATAL [mmap]: mmap() failed."); return NULL;
    }
    madvise(mapped, map_total, MADV_WILLNEED);

    Tensor* t = (Tensor*)safe_malloc(sizeof(Tensor));
    if (!t) { munmap(mapped, map_total); return NULL; }

    t->ndim       = ndim;
    t->total_size = total;
    t->byte_size  = map_bytes;
    t->owns_data  = false;
    t->is_arena   = false;
    t->dtype      = dtype;
    t->data       = (uint8_t*)mapped + page_off;

    t->stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        t->stride[i] = t->stride[i + 1] * (size_t)shape[i + 1];
    for (int i = 0; i < ndim; i++)
        t->shape[i] = shape[i];

    return t;
}

void tensor_mmap_free(Tensor* t) {
    if (!t) return;
    if (t->data) {
        long page_sz = sysconf(_SC_PAGESIZE);
        uintptr_t addr     = (uintptr_t)t->data;
        uintptr_t page_off = addr % (uintptr_t)page_sz;
        void*     base     = (void*)(addr - page_off);
        size_t    total    = t->byte_size + page_off;
        munmap(base, total);
    }
    free(t);
}

Tensor* tensor_silu(Tensor* A) {
    if (!A || A->dtype != DTYPE_FLOAT32) {
        TENSOR_ERROR("FATAL [SiLU]: Requires FLOAT32 tensor.");
    }
    if (!tensor_is_contiguous(A)) {
        TENSOR_ERROR("FATAL [SiLU]: Input must be contiguous.");
    }

    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32);
    if (!out) return NULL;

    size_t n = A->total_size;
    const float* __restrict src = (const float*)__builtin_assume_aligned(F32(A), 64);
    float*       __restrict dst = (float*)__builtin_assume_aligned(F32(out), 64);

    size_t i = 0;
#ifdef __AVX2__
    {
        __m256 ones = _mm256_set1_ps(1.0f);
        for (; i + 8 <= n; i += 8) {
            __m256 x   = _mm256_loadu_ps(src + i);
            __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), x);
            __m256 e   = avx2_expf(neg);
            __m256 sig = _mm256_div_ps(ones, _mm256_add_ps(ones, e));
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(x, sig));
        }
    }
#endif
    for (; i < n; i++) {
        float x  = src[i];
        float ex = fast_expf(-x);
        dst[i]   = x / (1.0f + ex);
    }

    return out;
}

Tensor* tensor_swiglu(Tensor* gate, Tensor* up) {
    if (!gate || !up || gate->dtype != DTYPE_FLOAT32 || up->dtype != DTYPE_FLOAT32) {
        TENSOR_ERROR("FATAL [SwiGLU]: Requires FLOAT32 tensors.");
    }
    if (!tensor_is_contiguous(gate) || !tensor_is_contiguous(up)) {
        TENSOR_ERROR("FATAL [SwiGLU]: Inputs must be contiguous.");
    }
    if (gate->total_size != up->total_size) {
        TENSOR_ERROR("FATAL [SwiGLU]: gate and up must have the same total size.");
    }

    Tensor* out = tensor_create_uninitialized(gate->ndim, gate->shape, DTYPE_FLOAT32);
    if (!out) return NULL;

    size_t n = gate->total_size;
    const float* __restrict g   = (const float*)__builtin_assume_aligned(F32(gate), 64);
    const float* __restrict u   = (const float*)__builtin_assume_aligned(F32(up), 64);
    float*       __restrict dst = (float*)__builtin_assume_aligned(F32(out), 64);

    size_t i = 0;
#ifdef __AVX2__
    {
        __m256 ones = _mm256_set1_ps(1.0f);
        for (; i + 8 <= n; i += 8) {
            __m256 gv  = _mm256_loadu_ps(g + i);
            __m256 uv  = _mm256_loadu_ps(u + i);
            __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), gv);
            __m256 e   = avx2_expf(neg);
            __m256 sig = _mm256_div_ps(ones, _mm256_add_ps(ones, e));
            __m256 silu_g = _mm256_mul_ps(gv, sig);
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(silu_g, uv));
        }
    }
#endif
    for (; i < n; i++) {
        float gx  = g[i];
        float ex  = fast_expf(-gx);
        float silu = gx / (1.0f + ex);
        dst[i]    = silu * u[i];
    }

    return out;
}

void tensor_fused_cross_entropy_loss_and_grad(Tensor* logits, Tensor* target_ids,
                                               Tensor* grads,  float*  out_loss) {
    if (!logits || !target_ids || !grads || !out_loss) {
        tensor_set_error("FATAL [CrossEntropy]: NULL pointer."); return;
    }
    if (logits->dtype != DTYPE_FLOAT32 || grads->dtype != DTYPE_FLOAT32) {
        tensor_set_error("FATAL [CrossEntropy]: logits/grads must be FLOAT32."); return;
    }
    if (target_ids->dtype != DTYPE_INT32) {
        tensor_set_error("FATAL [CrossEntropy]: target_ids must be INT32."); return;
    }
    if (!tensor_is_contiguous(logits) || !tensor_is_contiguous(target_ids) ||
        !tensor_is_contiguous(grads)) {
        tensor_set_error("FATAL [CrossEntropy]: All tensors must be contiguous."); return;
    }
    if (logits->ndim < 2) {
        tensor_set_error("FATAL [CrossEntropy]: logits must be at least 2D."); return;
    }

    int ndim  = logits->ndim;
    int vocab = logits->shape[ndim - 1];
    int batch = (int)(logits->total_size / (size_t)vocab);

    if ((int)(grads->total_size / (size_t)vocab) != batch ||
        (int)target_ids->total_size != batch) {
        tensor_set_error("FATAL [CrossEntropy]: Shape mismatch."); return;
    }

    const float* __restrict L  = (const float*)__builtin_assume_aligned(F32(logits), 64);
    float*       __restrict G  = (float*)__builtin_assume_aligned(F32(grads), 64);
    const int*   __restrict T  = (const int*)target_ids->data;

    float total_loss = 0.0f;
    int   n_valid    = 0;

    /* tid == -1 signals a padding position: zero the gradient row and skip loss.
     * This lets callers pack variable-length sequences into a padded batch without
     * contaminating the loss or gradients. */
#pragma omp parallel for schedule(static) reduction(+:total_loss) reduction(+:n_valid)
    for (int b = 0; b < batch; b++) {
        const float* __restrict row_l = L + (size_t)b * vocab;
        float*       __restrict row_g = G + (size_t)b * vocab;
        int tid = T[b];

        if (tid < 0) {
            memset(row_g, 0, (size_t)vocab * sizeof(float));
            continue;
        }
        n_valid++;

        float mx = row_l[0];
        for (int i = 1; i < vocab; i++)
            if (row_l[i] > mx) mx = row_l[i];

        float sum = 0.0f;
        for (int i = 0; i < vocab; i++) {
            float e = fast_expf(row_l[i] - mx);
            row_g[i] = e;
            sum += e;
        }

        float inv_sum = 1.0f / sum;
        for (int i = 0; i < vocab; i++)
            row_g[i] *= inv_sum;

        float p_target = row_g[tid];
        if (p_target < 1e-12f) p_target = 1e-12f;
        total_loss += -fast_logf(p_target);

        row_g[tid] -= 1.0f;
    }

    *out_loss = n_valid > 0 ? total_loss / (float)n_valid : 0.0f;
}

Tensor* tensor_rmsnorm_backward(Tensor* dY, Tensor* X, Tensor* weights, float eps) {
    if (!dY || !X || !weights) {
        TENSOR_ERROR("FATAL [RMSNormBwd]: NULL pointer.");
    }
    if (dY->dtype != DTYPE_FLOAT32 || X->dtype != DTYPE_FLOAT32 ||
        weights->dtype != DTYPE_FLOAT32) {
        TENSOR_ERROR("FATAL [RMSNormBwd]: All tensors must be FLOAT32.");
    }
    if (!tensor_is_contiguous(dY) || !tensor_is_contiguous(X) ||
        !tensor_is_contiguous(weights)) {
        TENSOR_ERROR("FATAL [RMSNormBwd]: All tensors must be contiguous.");
    }
    if (dY->ndim < 1 || X->ndim < 1 || weights->ndim != 1) {
        TENSOR_ERROR("FATAL [RMSNormBwd]: Unexpected dimensionality.");
    }

    int ndim = X->ndim;
    int d    = X->shape[ndim - 1];
    int rows = (int)(X->total_size / (size_t)d);

    if ((int)weights->total_size != d ||
        X->total_size != dY->total_size) {
        TENSOR_ERROR("FATAL [RMSNormBwd]: Shape mismatch.");
    }

    Tensor* dX = tensor_create_uninitialized(X->ndim, X->shape, DTYPE_FLOAT32);
    if (!dX) return NULL;

    const float* __restrict dy_ptr = (const float*)__builtin_assume_aligned(F32(dY), 64);
    const float* __restrict x_ptr  = (const float*)__builtin_assume_aligned(F32(X), 64);
    const float* __restrict w_ptr  = (const float*)__builtin_assume_aligned(F32(weights), 64);
    float*       __restrict dx_ptr = (float*)__builtin_assume_aligned(F32(dX), 64);

#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float* __restrict dy = dy_ptr + (size_t)r * d;
        const float* __restrict xr = x_ptr  + (size_t)r * d;
        float*       __restrict dx = dx_ptr + (size_t)r * d;

        float ss0 = 0.0f, ss1 = 0.0f, ss2 = 0.0f, ss3 = 0.0f;
        int i = 0;
#ifdef __AVX2__
        {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (; i + 16 <= d; i += 16) {
                __m256 v0 = _mm256_loadu_ps(xr + i);
                __m256 v1 = _mm256_loadu_ps(xr + i + 8);
                acc0 = _mm256_fmadd_ps(v0, v0, acc0);
                acc1 = _mm256_fmadd_ps(v1, v1, acc1);
            }
            __m256 acc = _mm256_add_ps(acc0, acc1);
            __m128 lo  = _mm256_castps256_ps128(acc);
            __m128 hi  = _mm256_extractf128_ps(acc, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
            ss0 = _mm_cvtss_f32(sum);
        }
#endif
        for (; i < d; i++) { float v = xr[i]; ss0 += v * v; }
        float rms = 1.0f / sqrtf(ss0 / (float)d + eps);
        float r3  = rms * rms * rms;

        float S = 0.0f;
        i = 0;
#ifdef __AVX2__
        {
            __m256 sA = _mm256_setzero_ps();
            for (; i + 8 <= d; i += 8) {
                __m256 wi  = _mm256_loadu_ps(w_ptr + i);
                __m256 dyi = _mm256_loadu_ps(dy + i);
                __m256 xi  = _mm256_loadu_ps(xr + i);
                sA = _mm256_fmadd_ps(_mm256_mul_ps(wi, dyi), xi, sA);
            }
            __m128 lo  = _mm256_castps256_ps128(sA);
            __m128 hi  = _mm256_extractf128_ps(sA, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
            S   = _mm_cvtss_f32(sum);
        }
#endif
        for (; i < d; i++) S += w_ptr[i] * dy[i] * xr[i];

        float coeff = S * r3 / (float)d;

        i = 0;
#ifdef __AVX2__
        {
            __m256 vrms  = _mm256_set1_ps(rms);
            __m256 vcoef = _mm256_set1_ps(coeff);
            for (; i + 8 <= d; i += 8) {
                __m256 wi  = _mm256_loadu_ps(w_ptr + i);
                __m256 dyi = _mm256_loadu_ps(dy + i);
                __m256 xi  = _mm256_loadu_ps(xr + i);
                __m256 term1 = _mm256_mul_ps(vrms, _mm256_mul_ps(wi, dyi));
                __m256 term2 = _mm256_mul_ps(xi, vcoef);
                _mm256_storeu_ps(dx + i, _mm256_sub_ps(term1, term2));
            }
        }
#endif
        for (; i < d; i++)
            dx[i] = rms * w_ptr[i] * dy[i] - xr[i] * coeff;
    }

    return dX;
}

void tensor_embedding_backward(Tensor* dY, Tensor* token_ids, Tensor* dWeights) {
    if (!dY || !token_ids || !dWeights) {
        tensor_set_error("FATAL [EmbeddingBwd]: NULL pointer."); return;
    }
    if (dY->dtype != DTYPE_FLOAT32 || dWeights->dtype != DTYPE_FLOAT32) {
        tensor_set_error("FATAL [EmbeddingBwd]: dY and dWeights must be FLOAT32."); return;
    }
    if (token_ids->dtype != DTYPE_INT32) {
        tensor_set_error("FATAL [EmbeddingBwd]: token_ids must be INT32."); return;
    }
    if (!tensor_is_contiguous(dY) || !tensor_is_contiguous(token_ids) ||
        !tensor_is_contiguous(dWeights)) {
        tensor_set_error("FATAL [EmbeddingBwd]: All tensors must be contiguous."); return;
    }

    int seq_len   = (int)token_ids->total_size;
    int embed_dim = dWeights->ndim >= 2 ? dWeights->shape[dWeights->ndim - 1]
                                        : (int)dWeights->total_size;
    int vocab     = (int)(dWeights->total_size / (size_t)embed_dim);

    if ((int)(dY->total_size / (size_t)embed_dim) != seq_len) {
        tensor_set_error("FATAL [EmbeddingBwd]: dY shape does not match seq_len/embed_dim."); return;
    }

    const float* __restrict dy  = (const float*)__builtin_assume_aligned(F32(dY), 64);
    float*       __restrict dW  = (float*)__builtin_assume_aligned(F32(dWeights), 64);
    const int*   __restrict ids = (const int*)token_ids->data;

#pragma omp parallel for schedule(static)
    for (int s = 0; s < seq_len; s++) {
        int tid = ids[s];
        if (tid < 0 || tid >= vocab) continue;

        const float* __restrict dy_row = dy + (size_t)s * embed_dim;
        float*       __restrict dw_row = dW + (size_t)tid * embed_dim;

        for (int i = 0; i < embed_dim; i++) {
#pragma omp atomic
            dw_row[i] += dy_row[i];
        }
    }
}

// ============================================================================
// 21. MAMBA / SELECTIVE SSM ENGINE
//
// ZOH recurrence (for each batch b, time t, feature d, state n):
//   Ā          = exp(delta[b,t,d] * A_log[d,n])
//   h[t,d,n]  = Ā * h[t-1,d,n]  +  delta[b,t,d] * B[b,t,n] * x[b,t,d]
//   y[b,t,d]  = Σ_n  C[b,t,n] * h[t,d,n]  +  D_skip[d] * x[b,t,d]
//
// Parallelism: OMP collapse(B,D) — T is serial (data dependency).
//              Tile-scan path for small B*D + large T.
// SIMD: AVX512 (16-wide) > AVX2 (8-wide) > scalar fallback over N.
// All intermediates on stack (N <= MAMBA_MAX_N = 128).
// ============================================================================

#define MAMBA_DT_MIN     1e-4f
#define MAMBA_DT_MAX     1.0f
#define MAMBA_MAX_N      128
#define MAMBA_TILE_SCAN_THRESH  128   /* use tile-scan when T >= this AND B*D < nthreads*4 */

/* _hsum256 already defined above — reuse existing implementation */

/* avx512_expf defined earlier near avx2_expf — no re-definition needed. */

/* ---------------------------------------------------------------------------
 * FORWARD INNER STEP — updates h[0..N) in-place for one (b,d,t) triple.
 *
 * Returns y_d (scalar output contribution from SSM state).
 * Optionally writes h into cache_slot[0..N).
 * ------------------------------------------------------------------------- */
__attribute__((always_inline))
static inline float _mamba_fwd_step(
        float* __restrict h,             /* [N] state — read+write      */
        const float* __restrict A_d,     /* [N] A_log values             */
        const float* __restrict B_t,     /* [N] B at time t              */
        const float* __restrict C_t,     /* [N] C at time t              */
        float delta_d, float x_d,
        float* __restrict cache_slot,    /* [N] or NULL                  */
        float* __restrict A_prod,        /* [N] or NULL                  */
        int N)
{
    /* Clamp delta for numerical safety */
    delta_d = delta_d < MAMBA_DT_MIN ? MAMBA_DT_MIN :
              delta_d > MAMBA_DT_MAX ? MAMBA_DT_MAX : delta_d;
    const float xd_delta = x_d * delta_d;

    float yd = 0.0f;
    int n = 0;

#ifdef __AVX512F__
    {
        __m512 vd   = _mm512_set1_ps(delta_d);
        __m512 vxdd = _mm512_set1_ps(xd_delta);
        __m512 vy   = _mm512_setzero_ps();
        for (; n + 16 <= N; n += 16) {
            __builtin_prefetch(A_d + n + 16, 0, 1);
            __builtin_prefetch(B_t + n + 16, 0, 1);
            __builtin_prefetch(C_t + n + 16, 0, 1);
            
            __m512 vA    = _mm512_load_ps(A_d + n);
            __m512 vZ    = _mm512_max_ps(_mm512_mul_ps(vd, vA), _mm512_set1_ps(-20.0f));
            __m512 vAbar = avx512_expf(vZ);
            __m512 vh    = _mm512_load_ps(h + n);
            __m512 vB    = _mm512_load_ps(B_t + n);
            __m512 vC    = _mm512_load_ps(C_t + n);
            vh = _mm512_fmadd_ps(vAbar, vh, _mm512_mul_ps(vxdd, vB));
            _mm512_store_ps(h + n, vh);
            
            if (cache_slot) _mm512_store_ps(cache_slot + n, vh);
            if (A_prod) {
                __m512 vAp = _mm512_load_ps(A_prod + n);
                _mm512_store_ps(A_prod + n, _mm512_mul_ps(vAp, vAbar));
            }
            
            vy = _mm512_fmadd_ps(vC, vh, vy);
        }
        yd = _mm512_reduce_add_ps(vy);
    }
#elif defined(__AVX2__)
    {
        __m256 vd   = _mm256_set1_ps(delta_d);
        __m256 vxdd = _mm256_set1_ps(xd_delta);
        __m256 vy   = _mm256_setzero_ps();
        /* 2x unroll for better ILP when N=16 (two 8-wide iters = full N) */
        for (; n + 16 <= N; n += 16) {
            __builtin_prefetch(A_d + n + 16, 0, 1);
            __builtin_prefetch(B_t + n + 16, 0, 1);
            __builtin_prefetch(C_t + n + 16, 0, 1);
            
            __m256 vA0   = _mm256_load_ps(A_d + n);
            __m256 vA1   = _mm256_load_ps(A_d + n + 8);
            __m256 vZ0   = _mm256_max_ps(_mm256_mul_ps(vd, vA0), _mm256_set1_ps(-20.0f));
            __m256 vZ1   = _mm256_max_ps(_mm256_mul_ps(vd, vA1), _mm256_set1_ps(-20.0f));
            __m256 vAb0  = avx2_expf(vZ0);
            __m256 vAb1  = avx2_expf(vZ1);
            __m256 vh0   = _mm256_load_ps(h + n);
            __m256 vh1   = _mm256_load_ps(h + n + 8);
            __m256 vB0   = _mm256_load_ps(B_t + n);
            __m256 vB1   = _mm256_load_ps(B_t + n + 8);
            __m256 vC0   = _mm256_load_ps(C_t + n);
            __m256 vC1   = _mm256_load_ps(C_t + n + 8);
            vh0 = _mm256_fmadd_ps(vAb0, vh0, _mm256_mul_ps(vxdd, vB0));
            vh1 = _mm256_fmadd_ps(vAb1, vh1, _mm256_mul_ps(vxdd, vB1));
            _mm256_store_ps(h + n,     vh0);
            _mm256_store_ps(h + n + 8, vh1);
            
            if (cache_slot) {
                _mm256_store_ps(cache_slot + n,     vh0);
                _mm256_store_ps(cache_slot + n + 8, vh1);
            }
            if (A_prod) {
                __m256 vAp0 = _mm256_load_ps(A_prod + n);
                __m256 vAp1 = _mm256_load_ps(A_prod + n + 8);
                _mm256_store_ps(A_prod + n, _mm256_mul_ps(vAp0, vAb0));
                _mm256_store_ps(A_prod + n + 8, _mm256_mul_ps(vAp1, vAb1));
            }
            
            vy = _mm256_fmadd_ps(vC0, vh0, _mm256_fmadd_ps(vC1, vh1, vy));
        }
        for (; n + 8 <= N; n += 8) {
            __m256 vA   = _mm256_load_ps(A_d + n);
            __m256 vZ   = _mm256_max_ps(_mm256_mul_ps(vd, vA), _mm256_set1_ps(-20.0f));
            __m256 vAb  = avx2_expf(vZ);
            __m256 vh   = _mm256_load_ps(h + n);
            __m256 vB   = _mm256_load_ps(B_t + n);
            __m256 vC   = _mm256_load_ps(C_t + n);
            vh = _mm256_fmadd_ps(vAb, vh, _mm256_mul_ps(vxdd, vB));
            _mm256_store_ps(h + n, vh);
            
            if (cache_slot) _mm256_store_ps(cache_slot + n, vh);
            if (A_prod) {
                __m256 vAp = _mm256_load_ps(A_prod + n);
                _mm256_store_ps(A_prod + n, _mm256_mul_ps(vAp, vAb));
            }
            
            vy = _mm256_fmadd_ps(vC, vh, vy);
        }
        yd = _hsum256(vy);
    }
#endif
    /* Scalar tail (also the sole path without AVX) */
    #pragma omp simd aligned(h, A_d, B_t, C_t : 64) reduction(+:yd)
    for (int i = n; i < N; i++) {
        float z = delta_d * A_d[i];
        z = z < -20.0f ? -20.0f : z;
        float Abar = fast_expf(z);
        h[i] = Abar * h[i] + xd_delta * B_t[i];
        if (cache_slot) cache_slot[i] = h[i];
        if (A_prod) A_prod[i] *= Abar;
        yd += C_t[i] * h[i];
    }
    return yd;
}

/* ---------------------------------------------------------------------------
 * TILE-SCAN FORWARD — parallel over T for small B*D workloads.
 *
 * Parallel log(N) prefix-scan correctly merges tile boundaries.
 * ------------------------------------------------------------------------- */
#define TILE_MAX_THREADS 1024

static void _mamba_tile_scan_bd(
        float* __restrict h,             /* [N] state (initial on entry)  */
        const float* __restrict x_bd,    /* x[b, 0..T-1, d]  stride D     */
        const float* __restrict A_d,     /* [N]                           */
        const float* __restrict B_bt,    /* B[b, 0, :]  stride N per step */
        const float* __restrict C_bt,    /* C[b, 0, :]  stride N per step */
        float D_d,
        const float* __restrict dt_bd,   /* delta[b, 0..T-1, d]  stride D */
        float* __restrict out_bd,        /* out[b, 0..T-1, d]  stride D   */
        float* __restrict cache_bd,      /* cache[b, d, 0..T-1, :]        */
        int T, int N, int D_stride, int BN_stride)
{
    int nchunks = omp_get_max_threads();
    if (nchunks > T) nchunks = T;
    if (nchunks > TILE_MAX_THREADS) nchunks = TILE_MAX_THREADS;
    int chunk = (T + nchunks - 1) / nchunks;

    /* Static thread-local buffers (zero heap allocation, safe per OpenMP thread team) */
    static __thread float tls_carry_A[TILE_MAX_THREADS * MAMBA_MAX_N] __attribute__((aligned(64)));
    static __thread float tls_carry_b[TILE_MAX_THREADS * MAMBA_MAX_N] __attribute__((aligned(64)));
    static __thread float tls_boundary[TILE_MAX_THREADS * MAMBA_MAX_N] __attribute__((aligned(64)));

    float* carry_A  = tls_carry_A;
    float* carry_b  = tls_carry_b;
    float* boundary = tls_boundary;

    /* ---- Pass 1: independent tiles, start from h=0 ---- */
#pragma omp parallel for schedule(static, 1)
    for (int c = 0; c < nchunks; c++) {
        int t0 = c * chunk;
        int t1 = t0 + chunk < T ? t0 + chunk : T;
        float h_local[MAMBA_MAX_N] __attribute__((aligned(64)));
        float A_prod[MAMBA_MAX_N]  __attribute__((aligned(64)));
        memset(h_local, 0, N * sizeof(float));
        for (int n = 0; n < N; n++) A_prod[n] = 1.0f;

        for (int t = t0; t < t1; t++) {
            float dt   = dt_bd[t * D_stride];
            float xd   = x_bd[t * D_stride];
            const float* Bt = B_bt + (size_t)t * BN_stride;
            const float* Ct = C_bt + (size_t)t * BN_stride;
            float* slot = cache_bd ? cache_bd + (size_t)t * N : NULL;
            
            float yd = _mamba_fwd_step(h_local, A_d, Bt, Ct, dt, xd, slot, A_prod, N);
            out_bd[t * D_stride] = yd + D_d * xd;
        }
        memcpy(boundary + c * N, h_local, N * sizeof(float));
        memcpy(carry_A  + c * N, A_prod,  N * sizeof(float));
    }

    /* ---- Pass 2: Parallel Prefix-Scan (Log N) ---- */
    if (nchunks >= 4) {
        int step = 1;
        while (step < nchunks) {
            #pragma omp parallel for schedule(static)
            for (int c = nchunks - 1; c >= step; c--) {
                int prev = c - step;
                float temp_A[MAMBA_MAX_N] __attribute__((aligned(64)));
                float temp_b[MAMBA_MAX_N] __attribute__((aligned(64)));
                
                #pragma omp simd aligned(temp_A, temp_b : 64)
                for (int n = 0; n < N; n++) {
                    temp_A[n] = carry_A[c * N + n] * carry_A[prev * N + n];
                    temp_b[n] = carry_A[c * N + n] * boundary[prev * N + n] + boundary[c * N + n];
                }
                
                #pragma omp simd aligned(temp_A, temp_b : 64)
                for (int n = 0; n < N; n++) {
                    carry_A[c * N + n] = temp_A[n];
                    boundary[c * N + n] = temp_b[n];
                }
            }
            step *= 2;
        }
    } else {
        for (int c = 1; c < nchunks; c++) {
            #pragma omp simd
            for (int n = 0; n < N; n++) {
                boundary[c * N + n] = carry_A[c * N + n] * boundary[(c - 1) * N + n] + boundary[c * N + n];
                carry_A[c * N + n] = carry_A[c * N + n] * carry_A[(c - 1) * N + n];
            }
        }
    }

    /* ---- Pass 3: Apply initial state h and offset outputs ---- */
#pragma omp parallel for schedule(static, 1)
    for (int c = nchunks - 1; c >= 0; c--) {
        #pragma omp simd
        for (int n = 0; n < N; n++) {
            carry_b[c * N + n] = (c == 0) ? h[n] : (carry_A[(c - 1) * N + n] * h[n] + boundary[(c - 1) * N + n]);
        }
    }

    #pragma omp simd
    for (int n = 0; n < N; n++) {
        h[n] = carry_A[(nchunks - 1) * N + n] * h[n] + boundary[(nchunks - 1) * N + n];
    }

#pragma omp parallel for schedule(static, 1)
    for (int c = 0; c < nchunks; c++) {
        int t0 = c * chunk;
        int t1 = t0 + chunk < T ? t0 + chunk : T;
        float* cb = carry_b + c * N;

        float h_local[MAMBA_MAX_N] __attribute__((aligned(64)));
        memcpy(h_local, cb, N * sizeof(float));

        for (int t = t0; t < t1; t++) {
            float dt = dt_bd[t * D_stride];
            float xd = x_bd[t * D_stride];
            const float* Bt = B_bt + (size_t)t * BN_stride;
            const float* Ct = C_bt + (size_t)t * BN_stride;
            float* slot = cache_bd ? cache_bd + (size_t)t * N : NULL;
            
            float yd = _mamba_fwd_step(h_local, A_d, Bt, Ct, dt, xd, slot, NULL, N);
            out_bd[t * D_stride] = yd + D_d * xd;
        }
    }
}

/* ---------------------------------------------------------------------------
 * PUBLIC: tensor_mamba_forward
 * ------------------------------------------------------------------------- */
void tensor_mamba_forward(Tensor* x,      Tensor* A_log,
                          Tensor* B_proj, Tensor* C_proj,
                          Tensor* D_skip, Tensor* delta,
                          Tensor* state,  Tensor* out,
                          Tensor* cache,  int training)
{
    if (!x || !A_log || !B_proj || !C_proj || !delta || !state || !out) {
        TENSOR_ERROR_VOID("mamba_forward: NULL required tensor.");
    }
    if (x->ndim != 3 || A_log->ndim != 2 || B_proj->ndim != 3 ||
        C_proj->ndim != 3 || delta->ndim != 3 || state->ndim != 3 || out->ndim != 3) {
        TENSOR_ERROR_VOID("mamba_forward: unexpected ndim.");
    }

    const int Bsz = x->shape[0];
    const int T   = x->shape[1];
    const int D   = x->shape[2];
    const int N   = A_log->shape[1];

    if (A_log->shape[0] != D || B_proj->shape[0] != Bsz || B_proj->shape[1] != T ||
        B_proj->shape[2] != N || C_proj->shape[0] != Bsz || C_proj->shape[1] != T ||
        C_proj->shape[2] != N || delta->shape[0] != Bsz || delta->shape[1] != T ||
        delta->shape[2] != D || state->shape[0] != Bsz || state->shape[1] != D ||
        state->shape[2] != N || out->shape[0] != Bsz || out->shape[1] != T ||
        out->shape[2] != D) {
        TENSOR_ERROR_VOID("mamba_forward: shape mismatch.");
    }
    if (N > MAMBA_MAX_N) {
        TENSOR_ERROR_VOID("mamba_forward: d_state > MAMBA_MAX_N (128).");
    }

    const float* __restrict xp  = (const float*)__builtin_assume_aligned(F32(x),      64);
    const float* __restrict Ap  = (const float*)__builtin_assume_aligned(F32(A_log),  64);
    const float* __restrict Bp  = (const float*)__builtin_assume_aligned(F32(B_proj), 64);
    const float* __restrict Cp  = (const float*)__builtin_assume_aligned(F32(C_proj), 64);
    const float* __restrict Dp  = D_skip ? (const float*)F32(D_skip) : NULL;
    const float* __restrict dtp = (const float*)__builtin_assume_aligned(F32(delta),  64);
    float* __restrict sp  = (float*)__builtin_assume_aligned(F32(state), 64);
    float* __restrict op  = (float*)__builtin_assume_aligned(F32(out),   64);
    float* __restrict cp  = cache ? (float*)F32(cache) : NULL;

    const int nthreads = omp_get_max_threads();
    const int use_tile = training && cache && T >= MAMBA_TILE_SCAN_THRESH
                         && Bsz * D < nthreads * 4;

    if (use_tile) {
        for (int b = 0; b < Bsz; b++) {
            for (int d = 0; d < D; d++) {
                float* __restrict h_bd    = sp + (size_t)b * D * N + (size_t)d * N;
                const float* __restrict x_bd  = xp  + (size_t)b * T * D + d;
                const float* __restrict dt_bd = dtp + (size_t)b * T * D + d;
                const float* __restrict B_bt  = Bp  + (size_t)b * T * N;
                const float* __restrict C_bt  = Cp  + (size_t)b * T * N;
                float* __restrict out_bd  = op + (size_t)b * T * D + d;
                float D_d = Dp ? Dp[d] : 0.0f;
                
                /* Updated cache striding to [B, D, T, N] */
                float* __restrict cache_bd = cp ? cp + ((size_t)b * D * T + (size_t)d * T) * N : NULL;
                
                _mamba_tile_scan_bd(h_bd, x_bd, Ap + (size_t)d * N, B_bt, C_bt,
                                    D_d, dt_bd, out_bd, cache_bd,
                                    T, N, D, N);
            }
        }
    } else {
#pragma omp parallel for collapse(2) schedule(static)
        for (int b = 0; b < Bsz; b++) {
            for (int d = 0; d < D; d++) {
                float* __restrict h = sp + (size_t)b * D * N + (size_t)d * N;
                const float* __restrict A_d = Ap + (size_t)d * N;
                float D_d = Dp ? Dp[d] : 0.0f;
                
                /* Updated cache striding to [B, D, T, N] */
                float* __restrict cache_bd = cp ? cp + ((size_t)b * D * T + (size_t)d * T) * N : NULL;

                for (int t = 0; t < T; t++) {
                    const size_t td_off = (size_t)b * T * D + (size_t)t * D + d;
                    float delta_d = dtp[td_off];
                    float x_d     = xp[td_off];
                    const float* __restrict B_t = Bp + (size_t)b * T * N + (size_t)t * N;
                    const float* __restrict C_t = Cp + (size_t)b * T * N + (size_t)t * N;
                    float* __restrict slot = cache_bd ? cache_bd + (size_t)t * N : NULL;
                    
                    float yd = _mamba_fwd_step(h, A_d, B_t, C_t, delta_d, x_d, slot, NULL, N);
                    op[td_off] = yd + D_d * x_d;
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * BACKWARD INNER STEP — computes all gradients for one (b,d,t) triple.
 *
 * dh[0..N):  accumulated gradient w.r.t. h_t from future steps (in/out).
 * On return, dh holds gradient w.r.t. h_{t-1}.
 * h_t:       h after this step (from cache).
 * h_prev:    h before this step (cache[t-1] or h0 at t=0).
 * Accumulates: dA_d, dB_t, dC_t (+=).
 * Returns:   scalar dx contribution and ddelta contribution.
 * ------------------------------------------------------------------------- */
__attribute__((always_inline))
static inline void _mamba_bwd_step(
        float* __restrict dh,            /* [N] in/out                   */
        const float* __restrict h_t,     /* [N] h after step t           */
        const float* __restrict h_prev,  /* [N] h before step t          */
        const float* __restrict A_d,     /* [N]                          */
        const float* __restrict B_t,     /* [N]                          */
        const float* __restrict C_t,     /* [N]                          */
        float dout_d, float delta_d, float x_d, float D_d,
        float* __restrict dC_t,          /* [N] +=                       */
        float* __restrict dB_t,          /* [N] +=                       */
        float* __restrict dA_d,          /* [N] +=                       */
        float* __restrict dx_acc,        /* scalar +=                    */
        float* __restrict dD_acc,        /* scalar +=                    */
        float* __restrict ddelta_acc,    /* scalar +=                    */
        int N)
{
    delta_d = delta_d < MAMBA_DT_MIN ? MAMBA_DT_MIN :
              delta_d > MAMBA_DT_MAX ? MAMBA_DT_MAX : delta_d;

    float s_dx = 0.0f, s_dd = 0.0f;
    *dD_acc += dout_d * x_d;

    int n = 0;
#ifdef __AVX512F__
    {
        __m512 vdout  = _mm512_set1_ps(dout_d);
        __m512 vdelta = _mm512_set1_ps(delta_d);
        __m512 vxd    = _mm512_set1_ps(x_d);
        __m512 vxdd   = _mm512_set1_ps(delta_d * x_d);
        __m512 vs_dx  = _mm512_setzero_ps();
        __m512 vs_dd  = _mm512_setzero_ps();
        for (; n + 16 <= N; n += 16) {
            __builtin_prefetch(A_d + n + 16, 0, 1);
            __builtin_prefetch(B_t + n + 16, 0, 1);
            __builtin_prefetch(C_t + n + 16, 0, 1);
            __builtin_prefetch(h_t + n + 16, 0, 1);
            __builtin_prefetch(h_prev + n + 16, 0, 1);
            
            __m512 vdh  = _mm512_load_ps(dh + n);
            __m512 vht  = _mm512_load_ps(h_t + n);
            __m512 vhp  = _mm512_load_ps(h_prev + n);
            __m512 vA   = _mm512_load_ps(A_d + n);
            __m512 vB   = _mm512_load_ps(B_t + n);
            __m512 vC   = _mm512_load_ps(C_t + n);
            __m512 vZ   = _mm512_max_ps(_mm512_mul_ps(vdelta, vA), _mm512_set1_ps(-20.0f));
            __m512 vAbar = avx512_expf(vZ);
            
            vdh = _mm512_fmadd_ps(vdout, vC, vdh);
            __m512 vdC = _mm512_load_ps(dC_t + n);
            _mm512_store_ps(dC_t + n, _mm512_fmadd_ps(vdout, vht, vdC));
            __m512 vdA = _mm512_load_ps(dA_d + n);
            _mm512_store_ps(dA_d + n, _mm512_fmadd_ps(
                vdh, _mm512_mul_ps(vhp, _mm512_mul_ps(vAbar, vdelta)), vdA));
            __m512 vdB = _mm512_load_ps(dB_t + n);
            _mm512_store_ps(dB_t + n, _mm512_fmadd_ps(vdh, vxdd, vdB));
            vs_dx = _mm512_fmadd_ps(vdh, _mm512_mul_ps(vdelta, vB), vs_dx);
            
            __m512 dA_term = _mm512_mul_ps(vhp, _mm512_mul_ps(vAbar, vA));
            __m512 dB_term = _mm512_mul_ps(vB, vxd);
            vs_dd = _mm512_fmadd_ps(vdh, _mm512_add_ps(dA_term, dB_term), vs_dd);
            _mm512_store_ps(dh + n, _mm512_mul_ps(vdh, vAbar));
        }
        s_dx += _mm512_reduce_add_ps(vs_dx);
        s_dd += _mm512_reduce_add_ps(vs_dd);
    }
#elif defined(__AVX2__)
    {
        __m256 vdout  = _mm256_set1_ps(dout_d);
        __m256 vdelta = _mm256_set1_ps(delta_d);
        __m256 vxd    = _mm256_set1_ps(x_d);
        __m256 vxdd   = _mm256_set1_ps(delta_d * x_d);
        __m256 vs_dx  = _mm256_setzero_ps();
        __m256 vs_dd  = _mm256_setzero_ps();
        for (; n + 8 <= N; n += 8) {
            __builtin_prefetch(A_d + n + 8, 0, 1);
            __builtin_prefetch(B_t + n + 8, 0, 1);
            __builtin_prefetch(C_t + n + 8, 0, 1);
            __builtin_prefetch(h_t + n + 8, 0, 1);
            __builtin_prefetch(h_prev + n + 8, 0, 1);
            
            __m256 vdh   = _mm256_load_ps(dh + n);
            __m256 vht   = _mm256_load_ps(h_t + n);
            __m256 vhp   = _mm256_load_ps(h_prev + n);
            __m256 vA    = _mm256_load_ps(A_d + n);
            __m256 vB    = _mm256_load_ps(B_t + n);
            __m256 vC    = _mm256_load_ps(C_t + n);
            __m256 vZ    = _mm256_max_ps(_mm256_mul_ps(vdelta, vA), _mm256_set1_ps(-20.0f));
            __m256 vAbar = avx2_expf(vZ);
            
            vdh = _mm256_fmadd_ps(vdout, vC, vdh);
            __m256 vdC = _mm256_load_ps(dC_t + n);
            _mm256_store_ps(dC_t + n, _mm256_fmadd_ps(vdout, vht, vdC));
            __m256 vdA = _mm256_load_ps(dA_d + n);
            _mm256_store_ps(dA_d + n, _mm256_fmadd_ps(
                vdh, _mm256_mul_ps(vhp, _mm256_mul_ps(vAbar, vdelta)), vdA));
            __m256 vdB = _mm256_load_ps(dB_t + n);
            _mm256_store_ps(dB_t + n, _mm256_fmadd_ps(vdh, vxdd, vdB));
            vs_dx = _mm256_fmadd_ps(vdh, _mm256_mul_ps(vdelta, vB), vs_dx);
            
            __m256 dA_term = _mm256_mul_ps(vhp, _mm256_mul_ps(vAbar, vA));
            __m256 dB_term = _mm256_mul_ps(vB, vxd);
            vs_dd = _mm256_fmadd_ps(vdh, _mm256_add_ps(dA_term, dB_term), vs_dd);
            _mm256_store_ps(dh + n, _mm256_mul_ps(vdh, vAbar));
        }
        s_dx += _hsum256(vs_dx);
        s_dd += _hsum256(vs_dd);
    }
#endif
    /* Scalar tail */
    #pragma omp simd aligned(dh, h_t, h_prev, A_d, B_t, C_t, dC_t, dB_t, dA_d : 64) reduction(+:s_dx, s_dd)
    for (int i = n; i < N; i++) {
        float z = delta_d * A_d[i];
        z = z < -20.0f ? -20.0f : z;
        float Abar  = fast_expf(z);
        float dh_n  = dh[i] + dout_d * C_t[i];
        dC_t[i]    += dout_d * h_t[i];
        dA_d[i]    += dh_n * h_prev[i] * Abar * delta_d;
        dB_t[i]    += dh_n * delta_d * x_d;
        s_dx       += dh_n * delta_d * B_t[i];
        s_dd       += dh_n * (h_prev[i] * Abar * A_d[i] + B_t[i] * x_d);
        dh[i]       = dh_n * Abar;
    }

    *dx_acc      += s_dx + dout_d * D_d;
    *ddelta_acc  += s_dd;
}

/* ---------------------------------------------------------------------------
 * PUBLIC: tensor_mamba_backward
 * ------------------------------------------------------------------------- */
void tensor_mamba_backward(Tensor* dout,   Tensor* x,
                           Tensor* A_log,  Tensor* B_proj, Tensor* C_proj,
                           Tensor* D_skip, Tensor* delta,
                           Tensor* h0,     Tensor* cache,
                           Tensor* dx,     Tensor* dA,
                           Tensor* dB,     Tensor* dC,
                           Tensor* dD,     Tensor* ddelta)
{
    if (!dout || !x || !A_log || !B_proj || !C_proj || !delta ||
        !h0 || !dx || !dA || !dB || !dC || !ddelta) {
        TENSOR_ERROR_VOID("mamba_backward: NULL required tensor.");
    }

    const int Bsz = x->shape[0];
    const int T   = x->shape[1];
    const int D   = x->shape[2];
    const int N   = A_log->shape[1];

    const float* __restrict doutp = (const float*)__builtin_assume_aligned(F32(dout), 64);
    const float* __restrict xp    = (const float*)__builtin_assume_aligned(F32(x), 64);
    const float* __restrict Ap    = (const float*)__builtin_assume_aligned(F32(A_log), 64);
    const float* __restrict Bp    = (const float*)__builtin_assume_aligned(F32(B_proj), 64);
    const float* __restrict Cp    = (const float*)__builtin_assume_aligned(F32(C_proj), 64);
    const float* __restrict Dp    = D_skip ? (const float*)F32(D_skip) : NULL;
    const float* __restrict dtp   = (const float*)__builtin_assume_aligned(F32(delta), 64);
    const float* __restrict h0p   = (const float*)__builtin_assume_aligned(F32(h0), 64);
    const float* __restrict cachep = cache ? (const float*)__builtin_assume_aligned(F32(cache), 64) : NULL;

    float* __restrict dxp  = (float*)__builtin_assume_aligned(F32(dx), 64);
    float* __restrict dAp  = (float*)__builtin_assume_aligned(F32(dA), 64);
    float* __restrict dBp  = (float*)__builtin_assume_aligned(F32(dB), 64);
    float* __restrict dCp  = (float*)__builtin_assume_aligned(F32(dC), 64);
    float* __restrict dDp  = dD ? (float*)__builtin_assume_aligned(F32(dD), 64) : NULL;
    float* __restrict ddtp = (float*)__builtin_assume_aligned(F32(ddelta), 64);

    int max_threads = omp_get_max_threads();
    int D_pad = (D + 15) & ~15; // Pad D to 64 bytes to eliminate false sharing

    // Thread-local buffers. dB and dC are dramatically shrunk to O(threads * T * N)
    float* thread_dD = (float*)safe_memalign(64, max_threads * D_pad * sizeof(float));
    float* thread_dA = (float*)safe_memalign(64, max_threads * D_pad * N * sizeof(float));
    float* thread_dB = (float*)safe_memalign(64, max_threads * T * N * sizeof(float));
    float* thread_dC = (float*)safe_memalign(64, max_threads * T * N * sizeof(float));
    float* thread_recompute = cachep ? NULL : (float*)safe_memalign(64, max_threads * T * N * sizeof(float));
    
    if (!thread_dD || !thread_dA || !thread_dB || !thread_dC || (!cachep && !thread_recompute)) {
        safe_free((void**)&thread_dD); safe_free((void**)&thread_dA); 
        safe_free((void**)&thread_dB); safe_free((void**)&thread_dC); safe_free((void**)&thread_recompute);
        TENSOR_ERROR_VOID("mamba_backward: OOM allocating thread-local buffers.");
    }

    memset(thread_dD, 0, max_threads * D_pad * sizeof(float));
    memset(thread_dA, 0, max_threads * D_pad * N * sizeof(float));

    /* Outer loop is over batch dimension — parallelized across threads internally */
    for (int b = 0; b < Bsz; b++) {
        memset(thread_dB, 0, max_threads * T * N * sizeof(float));
        memset(thread_dC, 0, max_threads * T * N * sizeof(float));

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            float* __restrict my_dA = thread_dA + tid * D_pad * N;
            float* __restrict my_dD = thread_dD + tid * D_pad;
            float* __restrict my_dB = thread_dB + tid * T * N;
            float* __restrict my_dC = thread_dC + tid * T * N;
            float* __restrict my_cache = thread_recompute ? thread_recompute + tid * T * N : NULL;

            /* Parallelizing deeply over D to efficiently populate the T*N buffers */
            #pragma omp for schedule(static)
            for (int d = 0; d < D; d++) {
                float dh[MAMBA_MAX_N] __attribute__((aligned(64)));
                memset(dh, 0, N * sizeof(float));

                const float* __restrict A_d = Ap + (size_t)d * N;
                float D_d = Dp ? Dp[d] : 0.0f;
                float dD_bd = 0.0f;

                /* Updated cache striding: [B, D, T, N] */
                const float* __restrict cache_bd = cachep ? cachep + ((size_t)b * D * T + (size_t)d * T) * N : NULL;

                // Lightweight Recompute Path if Cache is absent
                if (my_cache) {
                    float h_fwd[MAMBA_MAX_N] __attribute__((aligned(64)));
                    memcpy(h_fwd, h0p + (size_t)b * D * N + (size_t)d * N, N * sizeof(float));
                    for (int f_t = 0; f_t < T; f_t++) {
                        float dt = dtp[b * T * D + f_t * D + d];
                        float xd = xp[b * T * D + f_t * D + d];
                        _mamba_fwd_step(h_fwd, A_d, Bp + (size_t)b * T * N + (size_t)f_t * N, 
                                        Cp + (size_t)b * T * N + (size_t)f_t * N, dt, xd, 
                                        my_cache + (size_t)f_t * N, NULL, N);
                    }
                }

                for (int t = T - 1; t >= 0; t--) {
                    const size_t td_off = (size_t)b * T * D + (size_t)t * D + d;
                    float dout_d  = doutp[td_off];
                    float delta_d = dtp[td_off];
                    float x_d     = xp[td_off];

                    const float* __restrict B_t   = Bp    + (size_t)b * T * N + (size_t)t * N;
                    const float* __restrict C_t   = Cp    + (size_t)b * T * N + (size_t)t * N;
                    
                    /* Access sequentially along T axis using cache_bd offset */
                    const float* __restrict h_t   = my_cache ? my_cache + (size_t)t * N 
                                                             : cache_bd + (size_t)t * N;
                    const float* __restrict h_prev = (t > 0)
                        ? (my_cache ? my_cache + (size_t)(t-1) * N : cache_bd + (size_t)(t-1) * N)
                        : h0p + (size_t)b * D * N + (size_t)d * N;

                    float* __restrict my_dB_t = my_dB + (size_t)t * N;
                    float* __restrict my_dC_t = my_dC + (size_t)t * N;

                    float dx_step = 0.0f, dD_step = 0.0f, dd_step = 0.0f;

                    _mamba_bwd_step(dh, h_t, h_prev, A_d, B_t, C_t,
                                    dout_d, delta_d, x_d, D_d,
                                    my_dC_t, my_dB_t,
                                    my_dA + (size_t)d * N,
                                    &dx_step, &dD_step, &dd_step, N);

                    dxp[td_off]  += dx_step;
                    ddtp[td_off] += dd_step;
                    dD_bd        += dD_step;
                }
                my_dD[d] += dD_bd;
            }

            /* Local reduction specifically for this `b` index — no atomics, totally parallel */
            #pragma omp for collapse(2) schedule(static)
            for (int t = 0; t < T; t++) {
                for (int n = 0; n < N; n++) {
                    float sum_dB = 0.0f;
                    float sum_dC = 0.0f;
                    for (int th = 0; th < max_threads; th++) {
                        sum_dB += thread_dB[th * T * N + t * N + n];
                        sum_dC += thread_dC[th * T * N + t * N + n];
                    }
                    dBp[(size_t)b * T * N + (size_t)t * N + n] += sum_dB;
                    dCp[(size_t)b * T * N + (size_t)t * N + n] += sum_dC;
                }
            }
        }
    }

    // Final Thread-Local Reduction Loops (zero contention)
    #pragma omp parallel for schedule(static)
    for (int d = 0; d < D; d++) {
        float sum_dD = 0.0f;
        float* __restrict global_dA = dAp + (size_t)d * N;
        
        for (int t = 0; t < max_threads; t++) {
            sum_dD += thread_dD[t * D_pad + d];
            float* __restrict t_dA = thread_dA + t * D_pad * N + d * N;
            
            #pragma omp simd aligned(global_dA, t_dA : 64)
            for (int n = 0; n < N; n++) {
                global_dA[n] += t_dA[n];
            }
        }
        if (dDp) dDp[d] += sum_dD;
    }

    if (thread_recompute) safe_free((void**)&thread_recompute);
    safe_free((void**)&thread_dA);
    safe_free((void**)&thread_dD);
    safe_free((void**)&thread_dB);
    safe_free((void**)&thread_dC);
}
/* ---------------------------------------------------------------------------
 * Convenience allocators
 * ------------------------------------------------------------------------- */

Tensor* tensor_mamba_alloc_state(int batch, int d_model, int d_state) {
    int shape[3] = {batch, d_model, d_state};
    return tensor_create_dtype(3, shape, DTYPE_FLOAT32);
}

Tensor* tensor_mamba_alloc_cache(int batch, int seq_len, int d_model, int d_state) {
    int shape[4] = {batch, seq_len, d_model, d_state};
    return tensor_create_dtype(4, shape, DTYPE_FLOAT32);
}
// ============================================================================
// SECTION 22 — CLASSICAL ML EXTENSIONS
// ============================================================================

static int _s22_cmp_f32(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

// ─── 22.1  ARGMAX ALONG AXIS  ─────────────────────────────────────────────────
// Returns FLOAT32 tensor of argmax indices, shape = A->shape with axis removed.
Tensor* tensor_argmax_axis(Tensor* A, int axis) {
    if (A->dtype != DTYPE_FLOAT32 || axis < 0 || axis >= A->ndim)
        TENSOR_ERROR("FATAL [argmax_axis]: Invalid input.");

    int out_ndim = A->ndim - 1;
    if (out_ndim < 1) out_ndim = 1;

    int out_shape[8]; int j = 0;
    for (int i = 0; i < A->ndim; i++)
        if (i != axis) out_shape[j++] = A->shape[i];
    if (j < 1) { out_shape[0] = 1; j = 1; }

    Tensor* out = tensor_create_uninitialized(j, out_shape, DTYPE_FLOAT32);
    if (!out) return NULL;

    int   dim_len = A->shape[axis];
    size_t num_vecs = out->total_size;

    int idx[8] = {0};
    for (size_t v = 0; v < num_vecs; v++) {
        /* build base offset into A, inserting 0 at the axis position */
        size_t base = 0; int ai = 0;
        for (int d = 0; d < A->ndim; d++) {
            if (d == axis) continue;
            base += (size_t)idx[ai++] * A->stride[d];
        }
        float best = -FLT_MAX; int bi = 0;
        for (int i = 0; i < dim_len; i++) {
            float val = F32(A)[base + (size_t)i * A->stride[axis]];
            if (val > best) { best = val; bi = i; }
        }
        F32(out)[v] = (float)bi;
        for (int d = j - 1; d >= 0; d--) {
            idx[d]++; if (idx[d] < out_shape[d]) break; idx[d] = 0;
        }
    }
    return out;
}

// ─── 22.2  PAIRWISE SQUARED L2 DISTANCE  ──────────────────────────────────────
// D[i,j] = ||A[i]-B[j]||²  A:[m,k] B:[n,k] → out:[m,n]
// Uses SGEMM trick: D = -2*(A@B^T) + sqA[:,None] + sqB[None,:]
Tensor* tensor_pairwise_sq_l2(Tensor* A, Tensor* B) {
    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32)
        TENSOR_ERROR("FATAL [pairwise_sq_l2]: Requires FLOAT32.");
    if (A->ndim != 2 || B->ndim != 2 || A->shape[1] != B->shape[1])
        TENSOR_ERROR("FATAL [pairwise_sq_l2]: Both must be 2D with equal column count.");

    int m = A->shape[0], n = B->shape[0], k = A->shape[1];
    Tensor* a_c = tensor_is_contiguous(A) ? A : tensor_copy(A);
    Tensor* b_c = tensor_is_contiguous(B) ? B : tensor_copy(B);

    Tensor* out = tensor_create_uninitialized(2, (int[]){m, n}, DTYPE_FLOAT32);

    /* Step 1: out = -2 * A @ B^T  (single SGEMM) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, -2.0f,
                F32(a_c), k, F32(b_c), k,
                0.0f, F32(out), n);

    /* Step 2: precompute ||B[j]||^2 for all j  (reused across all i) */
    float* sq_b = (float*)malloc((size_t)n * sizeof(float));
    if (!sq_b) { if (a_c!=A) tensor_free(a_c); if (b_c!=B) tensor_free(b_c); tensor_free(out); TENSOR_ERROR("OOM sq_b"); }
    for (int j2 = 0; j2 < n; j2++)
        sq_b[j2] = cblas_sdot(k, F32(b_c) + (size_t)j2*k, 1, F32(b_c) + (size_t)j2*k, 1);

    /* Step 3: add ||A[i]||^2 to row i, add sq_b[j] to column j */
    #pragma omp parallel for schedule(static) if(m > 64)
    for (int i = 0; i < m; i++) {
        float sq_a = cblas_sdot(k, F32(a_c)+(size_t)i*k, 1, F32(a_c)+(size_t)i*k, 1);
        float* row = F32(out) + (size_t)i * n;
        #pragma omp simd
        for (int j2 = 0; j2 < n; j2++) {
            row[j2] += sq_a + sq_b[j2];
            if (row[j2] < 0.0f) row[j2] = 0.0f; /* clamp numeric error */
        }
    }

    free(sq_b);
    if (a_c != A) tensor_free(a_c);
    if (b_c != B) tensor_free(b_c);
    return out;
}

// ─── 22.3  INPLACE UNARY OPS  ─────────────────────────────────────────────────

void tensor_exp_inplace(Tensor* A) {
    if (!A || A->dtype != DTYPE_FLOAT32 || !tensor_is_contiguous(A)) return;
    size_t n = A->total_size; float* a = F32(A); size_t i = 0;
#ifdef __AVX2__
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(a+i, avx2_expf(_mm256_loadu_ps(a+i)));
#endif
    for (; i < n; i++) a[i] = fast_expf(a[i]);
}

void tensor_log_inplace(Tensor* A) {
    if (!A || A->dtype != DTYPE_FLOAT32 || !tensor_is_contiguous(A)) return;
    size_t n = A->total_size; float* a = F32(A);
    #pragma omp simd
    for (size_t i = 0; i < n; i++) a[i] = logf(a[i]);
}

void tensor_sqrt_inplace(Tensor* A) {
    if (!A || A->dtype != DTYPE_FLOAT32 || !tensor_is_contiguous(A)) return;
    size_t n = A->total_size; float* a = F32(A); size_t i = 0;
#ifdef __AVX2__
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(a+i, _mm256_sqrt_ps(_mm256_loadu_ps(a+i)));
#endif
    for (; i < n; i++) a[i] = sqrtf(a[i]);
}

/* sigmoid: explicit AVX2 fast path using avx2_expf */
void tensor_sigmoid_inplace(Tensor* A) {
    if (!A || A->dtype != DTYPE_FLOAT32 || !tensor_is_contiguous(A)) return;
    size_t n = A->total_size; float* a = F32(A); size_t i = 0;
#ifdef __AVX2__
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(a+i, avx2_sigmoidf(_mm256_loadu_ps(a+i)));
#endif
    for (; i < n; i++) a[i] = 1.0f / (1.0f + fast_expf(-a[i]));
}

/* tanh: explicit AVX2 fast path using avx2_tanhf */
void tensor_tanh_inplace(Tensor* A) {
    if (!A || A->dtype != DTYPE_FLOAT32 || !tensor_is_contiguous(A)) return;
    size_t n = A->total_size; float* a = F32(A); size_t i = 0;
#ifdef __AVX2__
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(a+i, avx2_tanhf(_mm256_loadu_ps(a+i)));
#endif
    for (; i < n; i++) a[i] = tanhf(a[i]);
}

void tensor_relu_inplace(Tensor* A) {
    if (!A || A->dtype != DTYPE_FLOAT32 || !tensor_is_contiguous(A)) return;
    size_t n = A->total_size; float* a = F32(A); size_t i = 0;
#ifdef __AVX512F__
    { __m512 vz = _mm512_setzero_ps();
      for (; i + 15 < n; i += 16)
          _mm512_storeu_ps(a+i, _mm512_max_ps(_mm512_loadu_ps(a+i), vz)); }
#elif defined(__AVX2__)
    { __m256 vz = _mm256_setzero_ps();
      for (; i + 7 < n; i += 8)
          _mm256_storeu_ps(a+i, _mm256_max_ps(_mm256_loadu_ps(a+i), vz)); }
#endif
    for (; i < n; i++) { if (a[i] < 0.0f) a[i] = 0.0f; }
}

// ─── 22.4  ROW-WISE SOFTMAX IN-PLACE  ─────────────────────────────────────────
// Delegates to the fully AVX2/AVX512-accelerated softmax_rows() helper.
void tensor_row_softmax_inplace(Tensor* A) {
    if (!A || A->dtype != DTYPE_FLOAT32 || A->ndim < 2 || !tensor_is_contiguous(A)) return;
    int rows = (int)(A->total_size / (size_t)A->shape[A->ndim-1]);
    int cols = A->shape[A->ndim-1];
    softmax_rows(F32(A), rows, cols);
}

// ─── 22.5  GBDT ENGINE  ───────────────────────────────────────────────────────
// Histogram-based GBDT (LightGBM-style).
// All per-sample operations happen in C; PHP only loops over O(2^D * T) nodes.

/* Compute per-feature quantile bin boundaries.
 * X: [N,D]  Q: number of bins (boundaries = Q-1 per feature)
 * Returns [D, Q-1] FLOAT32 boundary matrix. */
Tensor* tensor_gbdt_compute_boundaries(Tensor* X, int Q) {
    if (!X || X->dtype != DTYPE_FLOAT32 || X->ndim != 2 || Q < 2)
        TENSOR_ERROR("FATAL [gbdt_boundaries]: X must be [N,D] FLOAT32, Q>=2.");
    int N = X->shape[0], D = X->shape[1];
    int nb = Q - 1; /* number of boundary points per feature */
    Tensor* out = tensor_create_uninitialized(2, (int[]){D, nb}, DTYPE_FLOAT32);
    if (!out) return NULL;
    Tensor* a_c = tensor_is_contiguous(X) ? X : tensor_copy(X);
    float* col_buf = (float*)malloc((size_t)N * sizeof(float));
    if (!col_buf) { if (a_c!=X) tensor_free(a_c); tensor_free(out); TENSOR_ERROR("OOM col_buf"); }

    for (int d = 0; d < D; d++) {
        /* extract column d */
        for (int i = 0; i < N; i++) col_buf[i] = F32(a_c)[(size_t)i*D + d];
        /* partial sort to find quantile positions */
        /* use qsort on a copy for simplicity; N usually small at fit time */
        float* sorted = (float*)malloc((size_t)N * sizeof(float));
        if (!sorted) { free(col_buf); if (a_c!=X) tensor_free(a_c); tensor_free(out); TENSOR_ERROR("OOM sorted"); }
        memcpy(sorted, col_buf, (size_t)N * sizeof(float));
        /* insertion for small N, qsort otherwise */
        if (N <= 256) {
            for (int i = 1; i < N; i++) {
                float key = sorted[i]; int j = i-1;
                while (j >= 0 && sorted[j] > key) { sorted[j+1] = sorted[j]; j--; }
                sorted[j+1] = key;
            }
        } else {
            qsort(sorted, (size_t)N, sizeof(float), _s22_cmp_f32);
        }
        /* extract nb evenly spaced quantiles */
        for (int b = 0; b < nb; b++) {
            float pos = (float)(b + 1) / (float)(nb + 1) * (float)(N - 1);
            int lo = (int)pos; int hi = lo + 1;
            if (hi >= N) hi = N - 1;
            float frac = pos - (float)lo;
            F32(out)[(size_t)d * nb + b] = sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
        }
        free(sorted);
    }
    free(col_buf);
    if (a_c != X) tensor_free(a_c);
    return out;
}

/* Assign bin indices to each sample.
 * X: [N,D]  boundaries: [D, Q-1]  Q: num_bins
 * Returns [N,D] INT32 bin indices in [0, Q-1]. */
Tensor* tensor_gbdt_bin_samples(Tensor* X, Tensor* boundaries, int Q) {
    if (!X || !boundaries || X->ndim != 2 || boundaries->ndim != 2)
        TENSOR_ERROR("FATAL [gbdt_bin]: Invalid tensors.");
    int N = X->shape[0], D = X->shape[1], nb = Q - 1;
    if (boundaries->shape[0] != D || boundaries->shape[1] != nb)
        TENSOR_ERROR("FATAL [gbdt_bin]: boundaries shape mismatch.");
    Tensor* x_c  = tensor_is_contiguous(X) ? X : tensor_copy(X);
    Tensor* b_c  = tensor_is_contiguous(boundaries) ? boundaries : tensor_copy(boundaries);
    Tensor* out  = tensor_create_uninitialized(2, (int[]){N, D}, DTYPE_INT32);
    if (!out) { if (x_c!=X) tensor_free(x_c); if (b_c!=boundaries) tensor_free(b_c); return NULL; }

    #pragma omp parallel for collapse(2) schedule(static) if(N*D > 100000)
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < D; d++) {
            float val = F32(x_c)[(size_t)i*D + d];
            const float* bdry = F32(b_c) + (size_t)d * nb;
            /* binary search for bin index */
            int lo = 0, hi = nb - 1, bin = 0;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                if (val <= bdry[mid]) { bin = mid; hi = mid - 1; }
                else { bin = mid + 1; lo = mid + 1; }
            }
            if (bin > Q-1) bin = Q-1;
            I32(out)[(size_t)i*D + d] = bin;
        }
    }
    if (x_c != X) tensor_free(x_c);
    if (b_c != boundaries) tensor_free(b_c);
    return out;
}

/* MSE gradient: g[i] = pred[i]-y[i],  h[i] = 1.0 */
void tensor_gbdt_mse_grad_hess(Tensor* preds, Tensor* y, Tensor* out_g, Tensor* out_h) {
    if (!preds||!y||!out_g||!out_h) return;
    size_t n = preds->total_size;
    float* p = F32(preds); float* yt = F32(y);
    float* g = F32(out_g); float* h = F32(out_h);
    #pragma omp simd
    for (size_t i = 0; i < n; i++) { g[i] = p[i] - yt[i]; h[i] = 1.0f; }
}

/* Log-loss gradient: g[i] = sigmoid(pred)-y,  h[i] = p*(1-p) */
void tensor_gbdt_logloss_grad_hess(Tensor* preds, Tensor* y, Tensor* out_g, Tensor* out_h) {
    if (!preds||!y||!out_g||!out_h) return;
    size_t n = preds->total_size;
    float* p = F32(preds); float* yt = F32(y);
    float* g = F32(out_g); float* h = F32(out_h);
    for (size_t i = 0; i < n; i++) {
        float prob = 1.0f / (1.0f + expf(-p[i]));
        g[i] = prob - yt[i];
        h[i] = prob * (1.0f - prob) + 1e-16f;
    }
}

/* Build gradient histogram for a node.
 * bins:[N,D] INT32; g,h,mask:[N] FLOAT32; Q: num bins
 * hist_g,hist_h: [D,Q] FLOAT32 (caller pre-allocates and pre-zeros) */
void tensor_gbdt_histogram(Tensor* bins, Tensor* g, Tensor* h, Tensor* mask,
                            int Q, Tensor* hist_g, Tensor* hist_h) {
    if (!bins||!g||!h||!mask||!hist_g||!hist_h) return;
    int N = bins->shape[0], D = bins->shape[1];
    int32_t* bp  = I32(bins);
    float*   gp  = F32(g); float* hp = F32(h); float* mp = F32(mask);
    float*   hgp = F32(hist_g); float* hhp = F32(hist_h);

    for (int i = 0; i < N; i++) {
        float mi = mp[i];
        if (mi < 0.5f) continue;
        float gi = gp[i], hi2 = hp[i];
        for (int d = 0; d < D; d++) {
            int bin = bp[(size_t)i*D + d];
            hgp[(size_t)d*Q + bin] += gi;
            hhp[(size_t)d*Q + bin] += hi2;
        }
    }
}

/* Find best split: scans all (feature, bin) pairs using prefix-sum over histogram.
 * Sets out_feat, out_bin, out_gain.  gain<0 means no profitable split found. */
void tensor_gbdt_best_split(Tensor* hist_g, Tensor* hist_h, int Q,
                             float sum_g, float sum_h, int node_n,
                             float lambda, float gamma,
                             int* out_feat, int* out_bin, float* out_gain) {
    (void)node_n;
    *out_feat = -1; *out_bin = -1; *out_gain = -1.0f;
    if (!hist_g || !hist_h) return;
    int D = hist_g->shape[0];
    float* hgp = F32(hist_g); float* hhp = F32(hist_h);
    float root_score = (sum_g * sum_g) / (sum_h + lambda);
    float best = gamma; /* minimum gain threshold */

    for (int d = 0; d < D; d++) {
        float gl = 0.0f, hl = 0.0f;
        for (int b = 0; b < Q - 1; b++) {
            gl += hgp[(size_t)d*Q + b];
            hl += hhp[(size_t)d*Q + b];
            float gr = sum_g - gl, hr = sum_h - hl;
            if (hl < 1e-6f || hr < 1e-6f) continue;
            float gain = 0.5f * ((gl*gl)/(hl+lambda) + (gr*gr)/(hr+lambda) - root_score);
            if (gain > best) { best = gain; *out_feat = d; *out_bin = b; *out_gain = gain; }
        }
    }
}

/* Split node mask into left/right masks based on (feat, bin) split.
 * bins:[N,D] INT32; mask:[N]; out_left,out_right:[N] (pre-allocated) */
void tensor_gbdt_split_node(Tensor* bins, Tensor* mask, int feat, int bin,
                             Tensor* out_left, Tensor* out_right) {
    if (!bins||!mask||!out_left||!out_right) return;
    int N = bins->shape[0], D = bins->shape[1];
    int32_t* bp = I32(bins);
    float* mp = F32(mask); float* lp = F32(out_left); float* rp = F32(out_right);
    #pragma omp simd
    for (int i = 0; i < N; i++) {
        float m = mp[i];
        int in_left = (m > 0.5f) && (bp[(size_t)i*D + feat] <= bin);
        lp[i] = in_left ? 1.0f : 0.0f;
        rp[i] = (m > 0.5f && !in_left) ? 1.0f : 0.0f;
    }
}

/* Compute leaf value (-sum_g/(sum_h+lambda)), update preds += lr*leaf_value.
 * Returns the leaf value. */
float tensor_gbdt_leaf_update(Tensor* preds, Tensor* mask,
                               float sum_g, float sum_h,
                               float lr, float lambda) {
    float leaf = -sum_g / (sum_h + lambda);
    if (!preds||!mask) return leaf;
    int N = (int)preds->total_size;
    float* pp = F32(preds); float* mp = F32(mask);
    float delta = lr * leaf;
    #pragma omp simd
    for (int i = 0; i < N; i++) pp[i] += mp[i] > 0.5f ? delta : 0.0f;
    return leaf;
}

/* Prediction with all T trees packed as flat Tensor columns.
 * X_bins: [N,D] INT32  (binned test data)
 * feats:      [T, max_nodes] FLOAT32  (feature index; -1 = leaf)
 * thresholds: [T, max_nodes] FLOAT32  (bin index for splits; leaf value for leaves)
 * lefts:      [T, max_nodes] FLOAT32  (left child index; -1 at leaf)
 * rights:     [T, max_nodes] FLOAT32  (right child index; -1 at leaf)
 * tree_sizes: [T] FLOAT32             (node count per tree)
 * base_score: initial prediction
 * Returns [N] FLOAT32 predictions. */
Tensor* tensor_gbdt_predict_all(Tensor* X_bins, Tensor* feats, Tensor* thresholds,
                                 Tensor* lefts, Tensor* rights, Tensor* tree_sizes,
                                 float base_score) {
    if (!X_bins||!feats||!thresholds||!lefts||!rights||!tree_sizes)
        TENSOR_ERROR("FATAL [gbdt_predict]: NULL tensor.");
    int N = X_bins->shape[0], D = X_bins->shape[1];
    int T = feats->shape[0], M = feats->shape[1]; /* max_nodes */

    Tensor* xb_c = tensor_is_contiguous(X_bins) ? X_bins : tensor_copy(X_bins);
    Tensor* out  = tensor_create_uninitialized(1, (int[]){N}, DTYPE_FLOAT32);
    if (!out) { if (xb_c!=X_bins) tensor_free(xb_c); return NULL; }
    float* op = F32(out);
    for (int i = 0; i < N; i++) op[i] = base_score;

    int32_t* bp = I32(xb_c);
    float* fp = F32(feats); float* tp = F32(thresholds);
    float* lp = F32(lefts); float* rp = F32(rights);
    float* sp = F32(tree_sizes);

    #pragma omp parallel for schedule(static) if(N > 1000)
    for (int i = 0; i < N; i++) {
        const int32_t* xi = bp + (size_t)i * D;
        for (int t = 0; t < T; t++) {
            const float* tf = fp + (size_t)t * M;
            const float* tt = tp + (size_t)t * M;
            const float* tl = lp + (size_t)t * M;
            const float* tr = rp + (size_t)t * M;
            int sz = (int)sp[t];
            int node = 0;
            while (node < sz) {
                int feat = (int)tf[node];
                if (feat < 0) { op[i] += tt[node]; break; } /* leaf */
                int bin_split = (int)tt[node];
                node = (xi[feat] <= bin_split) ? (int)tl[node] : (int)tr[node];
                if (node < 0) break;
            }
        }
    }
    if (xb_c != X_bins) tensor_free(xb_c);
    return out;
}

// ─── 22.5b  GBDT LEAF-WISE ENGINE (LightGBM-style)  ──────────────────────────
// Histogram subtraction, priority-queue leaf-wise growth, L1+L2 regularisation.
// All per-sample work stays in C; PHP outer loop over T trees calls one function.

// ── L1+L2 score & leaf-weight helpers ────────────────────────────────────────

static inline float _gbdt_score_l1(float g, float h, float lam, float alpha) {
    float ag = fabsf(g);
    float cl = ag > alpha ? ag - alpha : 0.0f;
    return cl * cl / (h + lam);
}

static inline float _gbdt_leaf_val_l1(float g, float h, float lam, float alpha) {
    float ag = fabsf(g);
    float cl = ag > alpha ? ag - alpha : 0.0f;
    return -(g >= 0.0f ? cl : -cl) / (h + lam);
}

// ── Public histogram subtraction: out_g = parent_g - sibling_g  (same for h) ─
// All tensors must be [D*Q] FLOAT32 and same size.
void tensor_gbdt_hist_subtract(Tensor* parent_g, Tensor* parent_h,
                                Tensor* sibling_g, Tensor* sibling_h,
                                Tensor* out_g,     Tensor* out_h) {
    if (!parent_g || !parent_h || !sibling_g || !sibling_h || !out_g || !out_h) return;
    size_t n = parent_g->total_size;
    float* pg = F32(parent_g);  float* ph = F32(parent_h);
    float* sg = F32(sibling_g); float* sh = F32(sibling_h);
    float* og = F32(out_g);     float* oh = F32(out_h);

    size_t i = 0;
#ifdef __AVX512F__
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(og + i, _mm512_sub_ps(_mm512_loadu_ps(pg + i), _mm512_loadu_ps(sg + i)));
        _mm512_storeu_ps(oh + i, _mm512_sub_ps(_mm512_loadu_ps(ph + i), _mm512_loadu_ps(sh + i)));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(og + i, _mm256_sub_ps(_mm256_loadu_ps(pg + i), _mm256_loadu_ps(sg + i)));
        _mm256_storeu_ps(oh + i, _mm256_sub_ps(_mm256_loadu_ps(ph + i), _mm256_loadu_ps(sh + i)));
    }
#endif
    for (; i < n; i++) { og[i] = pg[i] - sg[i]; oh[i] = ph[i] - sh[i]; }
}

// ── Priority queue (max-heap by best_gain) ─────────────────────────────────

typedef struct {
    int   node_id;
    int   hist_slot;
    int   idx_start;
    int   idx_count;
    float sum_g;
    float sum_h;
    float best_gain;
    int   best_feat;
    int   best_bin;
} _GBDTLeaf;

static void _pq_push(_GBDTLeaf* heap, int* sz, _GBDTLeaf e) {
    int i = (*sz)++;
    heap[i] = e;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (heap[p].best_gain >= heap[i].best_gain) break;
        _GBDTLeaf tmp = heap[p]; heap[p] = heap[i]; heap[i] = tmp;
        i = p;
    }
}

static _GBDTLeaf _pq_pop(_GBDTLeaf* heap, int* sz) {
    _GBDTLeaf top = heap[0];
    heap[0] = heap[--(*sz)];
    int i = 0, n = *sz;
    for (;;) {
        int l = 2*i+1, r = 2*i+2, best = i;
        if (l < n && heap[l].best_gain > heap[best].best_gain) best = l;
        if (r < n && heap[r].best_gain > heap[best].best_gain) best = r;
        if (best == i) break;
        _GBDTLeaf tmp = heap[i]; heap[i] = heap[best]; heap[best] = tmp;
        i = best;
    }
    return top;
}

// ── Build histogram for an explicit index list ─────────────────────────────

static void _build_hist_idx_serial(
    const int32_t* bins_flat, const float* gp, const float* hp,
    const int* indices, int count, int D, int Q,
    float* hist_g, float* hist_h)
{
    memset(hist_g, 0, (size_t)D * Q * sizeof(float));
    memset(hist_h, 0, (size_t)D * Q * sizeof(float));
    for (int k = 0; k < count; k++) {
        int si = indices[k];
        float gi = gp[si], hi = hp[si];
        const int32_t* bi = bins_flat + (size_t)si * D;
        for (int d = 0; d < D; d++) {
            size_t pos = (size_t)d * Q + bi[d];
            hist_g[pos] += gi;
            hist_h[pos] += hi;
        }
    }
}

static void _build_hist_idx_par(
    const int32_t* bins_flat, const float* gp, const float* hp,
    const int* indices, int count, int D, int Q,
    float* thr_bufs, int max_threads,
    float* hist_g, float* hist_h)
{
    size_t DQ = (size_t)D * Q;
    if (count <= 8 * D) {
        _build_hist_idx_serial(bins_flat, gp, hp, indices, count, D, Q, hist_g, hist_h);
        return;
    }
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid >= max_threads) tid = max_threads - 1;
        float* lg = thr_bufs + (size_t)tid * 2 * DQ;
        float* lh = lg + DQ;
        memset(lg, 0, 2 * DQ * sizeof(float));
        #pragma omp for schedule(static)
        for (int k = 0; k < count; k++) {
            int si = indices[k];
            float gi = gp[si], hi2 = hp[si];
            const int32_t* bi = bins_flat + (size_t)si * D;
            for (int d = 0; d < D; d++) {
                size_t pos = (size_t)d * Q + bi[d];
                lg[pos] += gi;
                lh[pos] += hi2;
            }
        }
    }
    memset(hist_g, 0, DQ * sizeof(float));
    memset(hist_h, 0, DQ * sizeof(float));
    for (int t = 0; t < max_threads; t++) {
        float* lg = thr_bufs + (size_t)t * 2 * DQ;
        float* lh = lg + DQ;
        for (size_t j = 0; j < DQ; j++) { hist_g[j] += lg[j]; hist_h[j] += lh[j]; }
    }
}

// ── Find best split on raw histogram arrays (L1+L2) ─────────────────────────

static void _best_split_raw(
    const float* hist_g, const float* hist_h,
    int D, int Q, float sum_g, float sum_h,
    float lambda, float alpha, float gamma,
    int* out_feat, int* out_bin, float* out_gain)
{
    *out_feat = -1; *out_bin = -1; *out_gain = -1.0f;
    float root_score = _gbdt_score_l1(sum_g, sum_h, lambda, alpha);
    float best = gamma;

    for (int d = 0; d < D; d++) {
        float gl = 0.0f, hl = 0.0f;
        for (int b = 0; b < Q - 1; b++) {
            size_t pos = (size_t)d * Q + b;
            gl += hist_g[pos]; hl += hist_h[pos];
            float gr = sum_g - gl, hr = sum_h - hl;
            if (hl < 1e-6f || hr < 1e-6f) continue;
            float gain = 0.5f * (_gbdt_score_l1(gl, hl, lambda, alpha)
                               + _gbdt_score_l1(gr, hr, lambda, alpha)
                               - root_score);
            if (gain > best) { best = gain; *out_feat = d; *out_bin = b; *out_gain = gain; }
        }
    }
}

// ── Histogram subtraction on raw float arrays ────────────────────────────────

static void _hist_subtract_raw(const float* pg, const float* ph,
                                const float* sg, const float* sh,
                                float* og, float* oh, size_t DQ)
{
    size_t i = 0;
#ifdef __AVX2__
    for (; i + 8 <= DQ; i += 8) {
        _mm256_storeu_ps(og+i, _mm256_sub_ps(_mm256_loadu_ps(pg+i), _mm256_loadu_ps(sg+i)));
        _mm256_storeu_ps(oh+i, _mm256_sub_ps(_mm256_loadu_ps(ph+i), _mm256_loadu_ps(sh+i)));
    }
#endif
    for (; i < DQ; i++) { og[i] = pg[i] - sg[i]; oh[i] = ph[i] - sh[i]; }
}

// ── Main leaf-wise train tree ─────────────────────────────────────────────────
// bins:        [N,D] INT32   — pre-binned samples
// g, h:        [N] FLOAT32   — gradients / hessians for this round
// Q:           number of bins per feature
// max_leaves:  maximum leaf nodes in this tree (= 2^max_depth)
// lambda,alpha,gamma,min_hess,lr — regularisation & growth control
// preds:       [N] FLOAT32 — updated IN-PLACE with += lr*leaf_value
// out_feats,out_thresholds,out_lefts,out_rights: [max_nodes] FLOAT32 — filled here
// Returns number of nodes used (tree size).
int tensor_gbdt_train_tree(
    Tensor* bins, Tensor* g, Tensor* h,
    int Q, int max_leaves,
    float lambda, float alpha, float gamma, float min_hess, float lr,
    Tensor* preds,
    Tensor* out_feats, Tensor* out_thresholds,
    Tensor* out_lefts,  Tensor* out_rights)
{
    if (!bins || !g || !h || !preds || !out_feats || !out_thresholds || !out_lefts || !out_rights)
        return 0;
    if (bins->ndim != 2 || bins->dtype != DTYPE_INT32) return 0;
    if (max_leaves < 1) max_leaves = 1;

    int N = bins->shape[0], D = bins->shape[1];
    int max_nodes = (int)out_feats->total_size;
    size_t DQ = (size_t)D * Q;
    int n_slots = max_leaves + 2;
    int max_threads = omp_get_max_threads();

    // ── Allocate working memory ──────────────────────────────────────────────
    size_t hist_bytes  = (size_t)n_slots * 2 * DQ * sizeof(float);
    size_t thr_bytes   = (size_t)max_threads * 2 * DQ * sizeof(float);
    size_t idx_bytes   = (size_t)N * sizeof(int);
    size_t slot_bytes  = (size_t)n_slots * sizeof(int);
    size_t pq_bytes    = (size_t)(max_leaves + 1) * sizeof(_GBDTLeaf);

    // hist_pool and thr_bufs use safe_memalign (pool-aware, SIMD-aligned).
    // sample_idx, tmp_idx, slot_used, pq use plain calloc/free to avoid
    // pool bin-size contamination (safe_malloc allocates exact bytes but
    // safe_free_size rounds to bin — retrieval with a larger request overflows).
    float*    hist_pool  = (float*)safe_memalign(64, hist_bytes);
    float*    thr_bufs   = (float*)safe_memalign(64, thr_bytes);
    int*      sample_idx = (int*)calloc(N, sizeof(int));
    int*      tmp_idx    = (int*)calloc(N, sizeof(int));
    int*      slot_used  = (int*)calloc((size_t)n_slots, sizeof(int));
    _GBDTLeaf* pq        = (_GBDTLeaf*)calloc((size_t)(max_leaves + 1), sizeof(_GBDTLeaf));

    int ret = 0;
    if (!hist_pool || !thr_bufs || !sample_idx || !tmp_idx || !slot_used || !pq) {
        if (hist_pool)  safe_free_size(hist_pool, hist_bytes);
        if (thr_bufs)   safe_free_size(thr_bufs,  thr_bytes);
        free(sample_idx); free(tmp_idx); free(slot_used); free(pq);
        return 0;
    }

    for (int i = 0; i < N; i++) sample_idx[i] = i;

    // ── Slot pool helpers ─────────────────────────────────────────────────────
#define _HS_G(s)    (hist_pool + (size_t)(s) * 2 * DQ)
#define _HS_H(s)    (hist_pool + (size_t)(s) * 2 * DQ + DQ)
#define _ALLOC_SLOT(out) do {                                           \
    (out) = -1;                                                          \
    for (int _s = 0; _s < n_slots; _s++) {                              \
        if (!slot_used[_s]) { slot_used[_s] = 1; (out) = _s; break; }  \
    }                                                                    \
} while(0)
#define _FREE_SLOT(s)  (slot_used[(s)] = 0)

    // ── Initialise output arrays ──────────────────────────────────────────────
    float* fp = F32(out_feats); float* tp = F32(out_thresholds);
    float* lp = F32(out_lefts); float* rp = F32(out_rights);
    for (int i = 0; i < max_nodes; i++) { fp[i]=-1.0f; tp[i]=0.0f; lp[i]=-1.0f; rp[i]=-1.0f; }

    const int32_t* bp = I32(bins);
    float* gp = F32(g); float* hp2 = F32(h); float* pp = F32(preds);

    // ── Root: build histogram and compute sums ────────────────────────────────
    int root_slot; _ALLOC_SLOT(root_slot);
    _build_hist_idx_par(bp, gp, hp2, sample_idx, N, D, Q, thr_bufs, max_threads,
                        _HS_G(root_slot), _HS_H(root_slot));

    float root_sg = 0.0f, root_sh = 0.0f;
    for (int i = 0; i < N; i++) { root_sg += gp[i]; root_sh += hp2[i]; }

    int next_node = 1;
    int pq_sz = 0;
    int splits_done = 0;
    int max_splits = max_leaves - 1;

    // Find root best split
    int rf = -1, rb = -1; float rg_gain = -1.0f;
    if (root_sh >= min_hess)
        _best_split_raw(_HS_G(root_slot), _HS_H(root_slot), D, Q,
                        root_sg, root_sh, lambda, alpha, gamma, &rf, &rb, &rg_gain);

    if (rf < 0 || rg_gain <= 0.0f || max_splits == 0) {
        // Root is a leaf — single-node tree
        float lv = _gbdt_leaf_val_l1(root_sg, root_sh, lambda, alpha);
        tp[0] = lr * lv; fp[0] = -1.0f;
        float delta = lr * lv;
        for (int i = 0; i < N; i++) pp[i] += delta;
        _FREE_SLOT(root_slot);
        ret = 1;
    } else {
        _GBDTLeaf root_entry = {0, root_slot, 0, N, root_sg, root_sh, rg_gain, rf, rb};
        _pq_push(pq, &pq_sz, root_entry);

        while (pq_sz > 0 && splits_done < max_splits) {
            _GBDTLeaf leaf = _pq_pop(pq, &pq_sz);
            splits_done++;

            int feat = leaf.best_feat, split_bin = leaf.best_bin;
            int start = leaf.idx_start, count = leaf.idx_count;
            const int* cur = sample_idx + start;

            // Partition indices: left = bin <= split_bin, right = > split_bin
            int l_count = 0, r_tail = count - 1;
            for (int k = 0; k < count; k++) {
                int si = cur[k];
                if (bp[(size_t)si * D + feat] <= split_bin)
                    tmp_idx[l_count++] = si;
                else
                    tmp_idx[r_tail--] = si;
            }
            int r_count = count - l_count;
            memcpy(sample_idx + start, tmp_idx, (size_t)count * sizeof(int));

            int l_start = start, r_start = start + l_count;
            int left_nid = next_node++, right_nid = next_node++;

            // Write internal node
            fp[leaf.node_id] = (float)feat;
            tp[leaf.node_id] = (float)split_bin;
            lp[leaf.node_id] = (float)left_nid;
            rp[leaf.node_id] = (float)right_nid;

            // Compute child sums from histogram prefix and partition
            float l_sg = 0.0f, l_sh = 0.0f;
            for (int k = 0; k < l_count; k++) {
                l_sg += gp[sample_idx[l_start + k]];
                l_sh += hp2[sample_idx[l_start + k]];
            }
            float r_sg = leaf.sum_g - l_sg, r_sh = leaf.sum_h - l_sh;

            // Allocate slots for children
            int s_slot, o_slot;
            _ALLOC_SLOT(s_slot); _ALLOC_SLOT(o_slot);

            // Build histogram for smaller child; subtract for larger
            int small_nid, large_nid, small_sl, large_sl;
            int small_start, small_count, large_start, large_count;
            float small_sg, small_sh, large_sg, large_sh;

            if (l_count <= r_count) {
                _build_hist_idx_par(bp, gp, hp2, sample_idx + l_start, l_count,
                                    D, Q, thr_bufs, max_threads, _HS_G(s_slot), _HS_H(s_slot));
                _hist_subtract_raw(_HS_G(leaf.hist_slot), _HS_H(leaf.hist_slot),
                                   _HS_G(s_slot), _HS_H(s_slot),
                                   _HS_G(o_slot), _HS_H(o_slot), DQ);
                small_nid = left_nid;  small_sl = s_slot; small_start = l_start; small_count = l_count; small_sg = l_sg; small_sh = l_sh;
                large_nid = right_nid; large_sl = o_slot; large_start = r_start; large_count = r_count; large_sg = r_sg; large_sh = r_sh;
            } else {
                _build_hist_idx_par(bp, gp, hp2, sample_idx + r_start, r_count,
                                    D, Q, thr_bufs, max_threads, _HS_G(s_slot), _HS_H(s_slot));
                _hist_subtract_raw(_HS_G(leaf.hist_slot), _HS_H(leaf.hist_slot),
                                   _HS_G(s_slot), _HS_H(s_slot),
                                   _HS_G(o_slot), _HS_H(o_slot), DQ);
                small_nid = right_nid; small_sl = s_slot; small_start = r_start; small_count = r_count; small_sg = r_sg; small_sh = r_sh;
                large_nid = left_nid;  large_sl = o_slot; large_start = l_start; large_count = l_count; large_sg = l_sg; large_sh = l_sh;
            }
            _FREE_SLOT(leaf.hist_slot);

            // Process both children — pack into fixed array to avoid goto
            int  child_nid[2]   = {small_nid,   large_nid};
            int  child_sl[2]    = {small_sl,     large_sl};
            int  child_start[2] = {small_start,  large_start};
            int  child_count[2] = {small_count,  large_count};
            float child_sg[2]   = {small_sg,     large_sg};
            float child_sh[2]   = {small_sh,     large_sh};

            for (int ci = 0; ci < 2; ci++) {
                int   cnid  = child_nid[ci];
                int   csl   = child_sl[ci];
                int   cs    = child_start[ci];
                int   cc    = child_count[ci];
                float csg   = child_sg[ci], csh = child_sh[ci];
                float* chg  = _HS_G(csl);   float* chh = _HS_H(csl);

                if (csh < min_hess || cc <= 0 || splits_done >= max_splits) {
                    float lv = _gbdt_leaf_val_l1(csg, csh, lambda, alpha);
                    fp[cnid] = -1.0f; tp[cnid] = lr * lv;
                    float delta = lr * lv;
                    for (int k = 0; k < cc; k++) pp[sample_idx[cs + k]] += delta;
                    _FREE_SLOT(csl);
                    continue;
                }
                int cf = -1, cb = -1; float cg_gain = -1.0f;
                _best_split_raw(chg, chh, D, Q, csg, csh, lambda, alpha, gamma,
                                &cf, &cb, &cg_gain);
                if (cf < 0 || cg_gain <= 0.0f) {
                    float lv = _gbdt_leaf_val_l1(csg, csh, lambda, alpha);
                    fp[cnid] = -1.0f; tp[cnid] = lr * lv;
                    float delta = lr * lv;
                    for (int k = 0; k < cc; k++) pp[sample_idx[cs + k]] += delta;
                    _FREE_SLOT(csl);
                } else {
                    _GBDTLeaf cl = {cnid, csl, cs, cc, csg, csh, cg_gain, cf, cb};
                    _pq_push(pq, &pq_sz, cl);
                }
            }
        }

        // Drain remaining PQ entries as leaves
        while (pq_sz > 0) {
            _GBDTLeaf leaf = _pq_pop(pq, &pq_sz);
            float lv = _gbdt_leaf_val_l1(leaf.sum_g, leaf.sum_h, lambda, alpha);
            tp[leaf.node_id] = lr * lv; fp[leaf.node_id] = -1.0f;
            float delta = lr * lv;
            const int* idx = sample_idx + leaf.idx_start;
            for (int k = 0; k < leaf.idx_count; k++) pp[idx[k]] += delta;
            _FREE_SLOT(leaf.hist_slot);
        }

        ret = next_node;
    }

#undef _HS_G
#undef _HS_H
#undef _ALLOC_SLOT
#undef _FREE_SLOT

    safe_free_size(hist_pool, hist_bytes);
    safe_free_size(thr_bufs,  thr_bytes);
    free(sample_idx);
    free(tmp_idx);
    free(slot_used);
    free(pq);
    return ret;
}

// ─── 22.6  QUANTILE TRANSFORM  ────────────────────────────────────────────────
/* Fit quantile landmarks for each feature.
 * X: [N,D] → returns [D, n_quantiles] FLOAT32 */
Tensor* tensor_quantile_fit(Tensor* X, int n_quantiles) {
    if (!X || X->ndim != 2 || X->dtype != DTYPE_FLOAT32 || n_quantiles < 2)
        TENSOR_ERROR("FATAL [quantile_fit]: Invalid input.");
    int N = X->shape[0], D = X->shape[1];
    Tensor* xc = tensor_is_contiguous(X) ? X : tensor_copy(X);
    Tensor* out = tensor_create_uninitialized(2, (int[]){D, n_quantiles}, DTYPE_FLOAT32);
    float* col = (float*)malloc((size_t)N * sizeof(float));
    if (!col||!out) { if (xc!=X) tensor_free(xc); if (out) tensor_free(out); free(col); TENSOR_ERROR("OOM"); }

    for (int d = 0; d < D; d++) {
        for (int i = 0; i < N; i++) col[i] = F32(xc)[(size_t)i*D+d];
        qsort(col, (size_t)N, sizeof(float), _s22_cmp_f32);
        for (int q = 0; q < n_quantiles; q++) {
            float pos = (float)q / (float)(n_quantiles - 1) * (float)(N - 1);
            int lo = (int)pos; int hi = MIN(lo+1, N-1);
            float frac = pos - (float)lo;
            F32(out)[(size_t)d*n_quantiles + q] = col[lo]*(1.0f-frac) + col[hi]*frac;
        }
    }
    free(col);
    if (xc != X) tensor_free(xc);
    return out;
}

/* Transform X using fitted quantile landmarks → uniform [0,1] output.
 * landmarks: [D, n_quantiles] from tensor_quantile_fit
 * Returns [N, D] FLOAT32 in [0, 1]. */
Tensor* tensor_quantile_transform(Tensor* X, Tensor* landmarks, int n_quantiles) {
    if (!X||!landmarks||X->ndim!=2||landmarks->ndim!=2) TENSOR_ERROR("FATAL [qt_transform]: Invalid.");
    int N = X->shape[0], D = X->shape[1];
    if (landmarks->shape[0] != D || landmarks->shape[1] != n_quantiles)
        TENSOR_ERROR("FATAL [qt_transform]: Landmark shape mismatch.");
    Tensor* xc = tensor_is_contiguous(X) ? X : tensor_copy(X);
    Tensor* lc = tensor_is_contiguous(landmarks) ? landmarks : tensor_copy(landmarks);
    Tensor* out = tensor_create_uninitialized(2, (int[]){N, D}, DTYPE_FLOAT32);
    if (!out) { if (xc!=X) tensor_free(xc); if (lc!=landmarks) tensor_free(lc); TENSOR_ERROR("OOM"); }

    #pragma omp parallel for schedule(static) if(N*D > 50000)
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < D; d++) {
            float val = F32(xc)[(size_t)i*D+d];
            const float* lm = F32(lc) + (size_t)d * n_quantiles;
            if (val <= lm[0]) { F32(out)[(size_t)i*D+d] = 0.0f; continue; }
            if (val >= lm[n_quantiles-1]) { F32(out)[(size_t)i*D+d] = 1.0f; continue; }
            int lo = 0, hi = n_quantiles - 2;
            while (lo < hi) { int mid=(lo+hi)>>1; if (lm[mid+1]<=val) lo=mid+1; else hi=mid; }
            float span = lm[lo+1] - lm[lo];
            float frac = (span > 1e-12f) ? (val - lm[lo]) / span : 0.5f;
            F32(out)[(size_t)i*D+d] = ((float)lo + frac) / (float)(n_quantiles - 1);
        }
    }
    if (xc != X) tensor_free(xc);
    if (lc != landmarks) tensor_free(lc);
    return out;
}

// ─── 22.7  YEO-JOHNSON POWER TRANSFORM  ──────────────────────────────────────
static float _yj_apply(float x, float lam) {
    if (x >= 0.0f) {
        if (fabsf(lam) < 1e-6f) return log1pf(x);
        return (powf(x + 1.0f, lam) - 1.0f) / lam;
    } else {
        float lam2 = 2.0f - lam;
        if (fabsf(lam2) < 1e-6f) return -log1pf(-x);
        return -(powf(-x + 1.0f, lam2) - 1.0f) / lam2;
    }
}

/* Fit optimal Yeo-Johnson lambda per feature via neg-log-likelihood.
 * Returns [D] FLOAT32 lambdas. */
Tensor* tensor_yj_fit(Tensor* X) {
    if (!X||X->ndim!=2||X->dtype!=DTYPE_FLOAT32) TENSOR_ERROR("FATAL [yj_fit]: Invalid.");
    int N = X->shape[0], D = X->shape[1];
    Tensor* xc = tensor_is_contiguous(X) ? X : tensor_copy(X);
    Tensor* lams = tensor_create_uninitialized(1, (int[]){D}, DTYPE_FLOAT32);
    float* col = (float*)malloc((size_t)N * sizeof(float));
    if (!col||!lams) { if (xc!=X) tensor_free(xc); if (lams) tensor_free(lams); free(col); TENSOR_ERROR("OOM"); }

    for (int d = 0; d < D; d++) {
        for (int i = 0; i < N; i++) col[i] = F32(xc)[(size_t)i*D+d];
        /* Brent's method: minimize negative log-likelihood over lambda in [-5,5] */
        float best_lam = 0.0f, best_nll = FLT_MAX;
        for (int step = 0; step <= 100; step++) {
            float lam = -5.0f + 10.0f * (float)step / 100.0f;
            /* transform column, compute variance */
            float sum = 0.0f, sum2 = 0.0f;
            float log_abs_sum = 0.0f;
            for (int i = 0; i < N; i++) {
                float t = _yj_apply(col[i], lam);
                sum += t; sum2 += t*t;
                log_abs_sum += (col[i] >= 0.0f)
                    ? logf(col[i] + 1.0f) : logf(-col[i] + 1.0f);
            }
            float mean = sum / (float)N;
            float var  = sum2/(float)N - mean*mean;
            if (var < 1e-12f) var = 1e-12f;
            float nll = 0.5f * (float)N * logf(var) - (lam - 1.0f) * log_abs_sum;
            if (nll < best_nll) { best_nll = nll; best_lam = lam; }
        }
        F32(lams)[d] = best_lam;
    }
    free(col);
    if (xc != X) tensor_free(xc);
    return lams;
}

/* Apply Yeo-Johnson transform per column using fitted lambdas.
 * lambdas: [D]; X: [N,D] → [N,D] FLOAT32 */
Tensor* tensor_yj_transform(Tensor* X, Tensor* lambdas) {
    if (!X||!lambdas||X->ndim!=2) TENSOR_ERROR("FATAL [yj_transform]: Invalid.");
    int N = X->shape[0], D = X->shape[1];
    if ((int)lambdas->total_size != D) TENSOR_ERROR("FATAL [yj_transform]: Lambda size mismatch.");
    Tensor* xc = tensor_is_contiguous(X) ? X : tensor_copy(X);
    Tensor* out = tensor_create_uninitialized(2, (int[]){N, D}, DTYPE_FLOAT32);
    float* lp = F32(lambdas);
    if (!out) { if (xc!=X) tensor_free(xc); TENSOR_ERROR("OOM"); }
    #pragma omp parallel for schedule(static) if(N*D > 50000)
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < D; d++)
            F32(out)[(size_t)i*D+d] = _yj_apply(F32(xc)[(size_t)i*D+d], lp[d]);
    }
    if (xc != X) tensor_free(xc);
    return out;
}

/* Inverse Yeo-Johnson — needed for QuantileTransformer normal output. */
static float _yj_inv(float y, float lam) {
    if (y >= 0.0f) {
        if (fabsf(lam) < 1e-6f) return expm1f(y);
        return powf(y*lam + 1.0f, 1.0f/lam) - 1.0f;
    } else {
        float lam2 = 2.0f - lam;
        if (fabsf(lam2) < 1e-6f) return -expm1f(-y);
        return 1.0f - powf(-y*lam2 + 1.0f, 1.0f/lam2);
    }
}

/* ============================================================================
 * 22.8  NaN fill — replace NaN/Inf with fill_val in-place
 * ========================================================================== */
void tensor_fill_nan(Tensor* t, float fill_val) {
    if (!t || t->dtype != DTYPE_FLOAT32) return;
    float* d = F32(t);
    int n = (int)t->total_size;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        if (_f32_is_nan(d[i])) d[i] = fill_val;
}

/* ============================================================================
 * 22.9  Pearson correlation — each column of X [N,D] vs vector y [N]
 *       Returns [D] float tensor.  NaN pairs skipped (pairwise-complete).
 * ========================================================================== */
Tensor* tensor_pearson_cols(Tensor* X, Tensor* y) {
    if (!X || !y || X->ndim != 2)
        TENSOR_ERROR("tensor_pearson_cols: X must be [N,D]");
    int N = X->shape[0], D = X->shape[1];
    /* Accept y as [N] or [N,1] — both are contiguous with N elements */
    int y_ok = (y->ndim == 1 && (int)y->total_size == N) ||
               (y->ndim == 2 && y->shape[0] == N && y->shape[1] == 1);
    if (!y_ok)
        TENSOR_ERROR("tensor_pearson_cols: y must be [N] or [N,1]");

    Tensor* Xc = tensor_is_contiguous(X) ? X : tensor_copy(X);
    Tensor* yc = tensor_is_contiguous(y) ? y : tensor_copy(y);
    Tensor* out = tensor_zeros(1, (int[]){D});
    if (!out) { if (Xc!=X) tensor_free(Xc); if (yc!=y) tensor_free(yc); TENSOR_ERROR("OOM"); }

    float* xp = F32(Xc);
    float* yp = F32(yc);
    float* rp = F32(out);

#pragma omp parallel for schedule(static)
    for (int j = 0; j < D; j++) {
        double sx=0, sy=0, sxy=0, sx2=0, sy2=0;
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            float xv = xp[i*D + j], yv = yp[i];
            if (_f32_is_nan(xv) || _f32_is_nan(yv)) continue;
            sx  += xv;  sy  += yv;
            sxy += (double)xv * yv;
            sx2 += (double)xv * xv;
            sy2 += (double)yv * yv;
            cnt++;
        }
        if (cnt < 2) { rp[j] = 0.0f; continue; }
        double num = cnt * sxy - sx * sy;
        double den = sqrt((cnt*sx2 - sx*sx) * (cnt*sy2 - sy*sy));
        rp[j] = (den < 1e-12) ? 0.0f : (float)(num / den);
    }

    if (Xc != X) tensor_free(Xc);
    if (yc != y) tensor_free(yc);
    return out;
}

/* ============================================================================
 * 23. ADVANCED EDA STATISTICAL FUNCTIONS
 * ========================================================================== */

/* 23.1  Percentile of a flat tensor (p in [0,100]) */
float tensor_percentile(Tensor* A, float p) {
    if (!A || A->dtype != DTYPE_FLOAT32 || A->total_size == 0) return 0.0f;
    size_t n = A->total_size;
    float* buf = (float*)malloc(n * sizeof(float));
    if (!buf) { TENSOR_ERROR_VAL(0.0f, "OOM"); }
    memcpy(buf, A->data, n * sizeof(float));
    qsort(buf, n, sizeof(float), cmp_float);
    float idx = (p / 100.0f) * (float)(n - 1);
    int lo = (int)idx;
    int hi = lo + 1;
    float frac = idx - (float)lo;
    float result = (hi < (int)n) ? buf[lo] * (1.0f - frac) + buf[hi] * frac : buf[lo];
    free(buf);
    return result;
}

/* 23.2  IQR (interquartile range) of a flat tensor */
float tensor_iqr(Tensor* A) {
    return tensor_percentile(A, 75.0f) - tensor_percentile(A, 25.0f);
}

/* 23.3  MAD (median absolute deviation) */
float tensor_mad(Tensor* A) {
    if (!A || A->dtype != DTYPE_FLOAT32 || A->total_size == 0) return 0.0f;
    float med = tensor_median(A);
    size_t n = A->total_size;
    float* buf = (float*)malloc(n * sizeof(float));
    if (!buf) { TENSOR_ERROR_VAL(0.0f, "OOM"); }
    float* src = F32(A);
    for (size_t i = 0; i < n; i++) buf[i] = fabsf(src[i] - med);
    qsort(buf, n, sizeof(float), cmp_float);
    float result = (n % 2 == 0) ? (buf[n/2-1] + buf[n/2]) / 2.0f : buf[n/2];
    free(buf);
    return result;
}

/* 23.4  Fisher-Pearson skewness (standardized 3rd moment) */
float tensor_skewness(Tensor* A) {
    if (!A || A->dtype != DTYPE_FLOAT32 || A->total_size < 3) return 0.0f;
    double n = (double)A->total_size;
    double mean = 0.0, m2 = 0.0, m3 = 0.0;
    float* d = F32(A);
    for (size_t i = 0; i < A->total_size; i++) mean += d[i];
    mean /= n;
    for (size_t i = 0; i < A->total_size; i++) {
        double dev = d[i] - mean;
        m2 += dev * dev;
        m3 += dev * dev * dev;
    }
    m2 /= n; m3 /= n;
    if (m2 < 1e-12) return 0.0f;
    return (float)(m3 / pow(m2, 1.5));
}

/* 23.5  Excess kurtosis (Fisher's definition: normal = 0) */
float tensor_kurtosis(Tensor* A) {
    if (!A || A->dtype != DTYPE_FLOAT32 || A->total_size < 4) return 0.0f;
    double n = (double)A->total_size;
    double mean = 0.0, m2 = 0.0, m4 = 0.0;
    float* d = F32(A);
    for (size_t i = 0; i < A->total_size; i++) mean += d[i];
    mean /= n;
    for (size_t i = 0; i < A->total_size; i++) {
        double dev = d[i] - mean;
        double dev2 = dev * dev;
        m2 += dev2;
        m4 += dev2 * dev2;
    }
    m2 /= n; m4 /= n;
    if (m2 < 1e-12) return 0.0f;
    return (float)(m4 / (m2 * m2) - 3.0);
}

/* 23.6  Shannon entropy of a flat float tensor via equal-width histogram.
 *       n_bins: number of histogram bins (recommend 32-64 for continuous data). */
float tensor_entropy_binned(Tensor* A, int n_bins) {
    if (!A || A->dtype != DTYPE_FLOAT32 || A->total_size == 0 || n_bins < 2) return 0.0f;
    float mn = tensor_min(A), mx = tensor_max(A);
    if (mx - mn < 1e-12f) return 0.0f;
    float rng = mx - mn;
    int* hist = (int*)calloc((size_t)n_bins, sizeof(int));
    if (!hist) { TENSOR_ERROR_VAL(0.0f, "OOM"); }
    float* d = F32(A);
    size_t n = A->total_size;
    size_t valid_n = 0;
    if (tensor_is_contiguous(A)) {
        for (size_t i = 0; i < n; i++) {
            if (isnanf(d[i])) continue;
            int bin = (int)((d[i] - mn) / rng * (float)n_bins);
            if (bin < 0) bin = 0;
            if (bin >= n_bins) bin = n_bins - 1;
            hist[bin]++;
            valid_n++;
        }
    } else {
        int idx[8] = {0};
        for (size_t i = 0; i < n; i++) {
            size_t offset = 0;
            for (int dim = 0; dim < A->ndim; dim++) offset += (size_t)idx[dim] * A->stride[dim];
            float val = d[offset];
            if (!isnanf(val)) {
                int bin = (int)((val - mn) / rng * (float)n_bins);
                if (bin < 0) bin = 0;
                if (bin >= n_bins) bin = n_bins - 1;
                hist[bin]++;
                valid_n++;
            }
            for (int dim = A->ndim - 1; dim >= 0; dim--) {
                idx[dim]++; if (idx[dim] < A->shape[dim]) break; idx[dim] = 0;
            }
        }
    }
    if (valid_n == 0) { free(hist); return 0.0f; }
    n = valid_n;
    double entropy = 0.0;
    for (int b = 0; b < n_bins; b++) {
        if (hist[b] > 0) {
            double p = (double)hist[b] / (double)n;
            entropy -= p * log(p);
        }
    }
    free(hist);
    return (float)entropy;
}

/* 23.7  Full-column batch statistics for a single column of a [N,D] matrix.
 *       col: column index; out must be float[10]:
 *       [mean, std, skew, kurt, median, p25, p75, iqr, mad, nan_ratio] */
void tensor_col_stats(Tensor* X, int col, float* out) {
    if (!X || !out || X->ndim != 2 || col < 0 || col >= X->shape[1]) return;
    int N = X->shape[0], D = X->shape[1];
    float* xp = F32(X);
    /* collect column, skip NaN */
    float* buf = (float*)malloc((size_t)N * sizeof(float));
    if (!buf) { TENSOR_ERROR_VOID("OOM"); }
    int valid = 0;
    int nan_count = 0;
    for (int i = 0; i < N; i++) {
        float v = xp[i * D + col];
        if (_f32_is_nan(v)) { nan_count++; continue; }
        buf[valid++] = v;
    }
    if (valid == 0) {
        for (int i = 0; i < 10; i++) out[i] = 0.0f;
        out[9] = 1.0f; /* all NaN */
        free(buf); return;
    }
    /* mean */
    double sum = 0.0;
    for (int i = 0; i < valid; i++) sum += buf[i];
    double mean = sum / valid;
    /* std, skew, kurt via single pass with Welford + moments */
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (int i = 0; i < valid; i++) {
        double d = buf[i] - mean;
        double d2 = d*d;
        m2 += d2; m3 += d2*d; m4 += d2*d2;
    }
    m2 /= valid; m3 /= valid; m4 /= valid;
    float std_v = (m2 > 1e-12) ? (float)sqrt(m2) : 0.0f;
    float skew  = (m2 > 1e-12) ? (float)(m3 / pow(m2, 1.5)) : 0.0f;
    float kurt  = (m2 > 1e-12) ? (float)(m4 / (m2*m2) - 3.0) : 0.0f;
    /* median, p25, p75 via sort */
    qsort(buf, (size_t)valid, sizeof(float), cmp_float);
    float med  = (valid % 2 == 0) ? (buf[valid/2-1]+buf[valid/2])/2.0f : buf[valid/2];
    float p25_idx = 0.25f * (valid - 1);
    float p75_idx = 0.75f * (valid - 1);
    int p25_lo = (int)p25_idx, p75_lo = (int)p75_idx;
    float p25 = buf[p25_lo] * (1.0f - (p25_idx - p25_lo)) + (p25_lo+1 < valid ? buf[p25_lo+1] : buf[p25_lo]) * (p25_idx - p25_lo);
    float p75 = buf[p75_lo] * (1.0f - (p75_idx - p75_lo)) + (p75_lo+1 < valid ? buf[p75_lo+1] : buf[p75_lo]) * (p75_idx - p75_lo);
    float iqr_v = p75 - p25;
    /* MAD */
    float med_local = med;
    for (int i = 0; i < valid; i++) buf[i] = fabsf(buf[i] - med_local);
    qsort(buf, (size_t)valid, sizeof(float), cmp_float);
    float mad_v = (valid % 2 == 0) ? (buf[valid/2-1]+buf[valid/2])/2.0f : buf[valid/2];
    out[0] = (float)mean;
    out[1] = std_v;
    out[2] = skew;
    out[3] = kurt;
    out[4] = med;
    out[5] = p25;
    out[6] = p75;
    out[7] = iqr_v;
    out[8] = mad_v;
    out[9] = (float)nan_count / (float)N;
    free(buf);
}

/* 23.8  Pairwise Pearson correlation matrix for all columns of X [N,D].
 *       Returns [D,D] float tensor. */
Tensor* tensor_correlation_matrix(Tensor* X) {
    if (!X || X->ndim != 2) TENSOR_ERROR("tensor_correlation_matrix: X must be [N,D]");
    int N = X->shape[0], D = X->shape[1];
    Tensor* Xc = tensor_is_contiguous(X) ? X : tensor_copy(X);
    Tensor* out = tensor_zeros(2, (int[]){D, D});
    if (!out) { if (Xc!=X) tensor_free(Xc); TENSOR_ERROR("OOM"); }
    float* xp = F32(Xc);
    float* op = F32(out);
    /* precompute per-column mean and std */
    double* col_mean = (double*)malloc((size_t)D * sizeof(double));
    double* col_std  = (double*)malloc((size_t)D * sizeof(double));
    if (!col_mean || !col_std) {
        free(col_mean); free(col_std);
        if (Xc!=X) tensor_free(Xc); tensor_free(out);
        TENSOR_ERROR("OOM");
    }
    for (int j = 0; j < D; j++) {
        double s = 0.0, s2 = 0.0;
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            float v = xp[i*D+j];
            if (!_f32_is_nan(v)) { s += v; s2 += (double)v*v; cnt++; }
        }
        col_mean[j] = cnt > 0 ? s/cnt : 0.0;
        double var = cnt > 1 ? s2/cnt - col_mean[j]*col_mean[j] : 0.0;
        col_std[j] = var > 1e-12 ? sqrt(var) : 0.0;
    }
#pragma omp parallel for schedule(dynamic) if(D > 16)
    for (int j = 0; j < D; j++) {
        op[j*D+j] = 1.0f;
        for (int k = j+1; k < D; k++) {
            if (col_std[j] < 1e-12 || col_std[k] < 1e-12) { op[j*D+k] = op[k*D+j] = 0.0f; continue; }
            double cov = 0.0; int cnt = 0;
            for (int i = 0; i < N; i++) {
                float vj = xp[i*D+j], vk = xp[i*D+k];
                if (!_f32_is_nan(vj) && !_f32_is_nan(vk)) {
                    cov += (vj - col_mean[j]) * (vk - col_mean[k]);
                    cnt++;
                }
            }
            float r = cnt > 1 ? (float)(cov / (cnt * col_std[j] * col_std[k])) : 0.0f;
            op[j*D+k] = op[k*D+j] = r;
        }
    }
    free(col_mean); free(col_std);
    if (Xc != X) tensor_free(Xc);
    return out;
}

/* 23.9  Mutual information between each column of X [N,D] and target y [N].
 *       Uses histogram-based estimation with n_bins equal-width bins.
 *       Returns [D] float tensor of MI values (nats). */
Tensor* tensor_mutual_info_cols(Tensor* X, Tensor* y, int n_bins) {
    if (!X || !y || X->ndim != 2) TENSOR_ERROR("tensor_mutual_info_cols: X must be [N,D]");
    int N = X->shape[0], D = X->shape[1];
    if ((int)y->total_size != N) TENSOR_ERROR("tensor_mutual_info_cols: y length mismatch");
    if (n_bins < 2) n_bins = 32;
    Tensor* Xc = tensor_is_contiguous(X) ? X : tensor_copy(X);
    Tensor* yc = tensor_is_contiguous(y) ? y : tensor_copy(y);
    Tensor* out = tensor_zeros(1, (int[]){D});
    if (!out) { if (Xc!=X) tensor_free(Xc); if (yc!=y) tensor_free(yc); TENSOR_ERROR("OOM"); }
    float* xp = F32(Xc);
    float* yp = F32(yc);
    float* op = F32(out);
    float y_min = yp[0], y_max = yp[0];
    for (int i = 1; i < N; i++) { if (yp[i] < y_min) y_min = yp[i]; if (yp[i] > y_max) y_max = yp[i]; }
    float y_rng = y_max - y_min;
    int* joint = (int*)malloc((size_t)(n_bins * n_bins) * sizeof(int));
    int* x_hist = (int*)malloc((size_t)n_bins * sizeof(int));
    int* y_hist = (int*)malloc((size_t)n_bins * sizeof(int));
    if (!joint || !x_hist || !y_hist) {
        free(joint); free(x_hist); free(y_hist);
        if (Xc!=X) tensor_free(Xc); if (yc!=y) tensor_free(yc); tensor_free(out);
        TENSOR_ERROR("OOM");
    }
    for (int j = 0; j < D; j++) {
        float x_min = xp[0*D+j], x_max = x_min;
        for (int i = 1; i < N; i++) { float v = xp[i*D+j]; if (v<x_min) x_min=v; if (v>x_max) x_max=v; }
        float x_rng = x_max - x_min;
        memset(joint, 0, (size_t)(n_bins*n_bins)*sizeof(int));
        memset(x_hist, 0, (size_t)n_bins*sizeof(int));
        memset(y_hist, 0, (size_t)n_bins*sizeof(int));
        for (int i = 0; i < N; i++) {
            float xv = xp[i*D+j], yv = yp[i];
            if (_f32_is_nan(xv) || _f32_is_nan(yv)) continue;
            int bx = (x_rng < 1e-12f) ? 0 : (int)((xv - x_min) / x_rng * n_bins);
            int by = (y_rng < 1e-12f) ? 0 : (int)((yv - y_min) / y_rng * n_bins);
            if (bx >= n_bins) bx = n_bins - 1;
            if (by >= n_bins) by = n_bins - 1;
            joint[bx * n_bins + by]++;
            x_hist[bx]++;
            y_hist[by]++;
        }
        double mi = 0.0;
        for (int bx = 0; bx < n_bins; bx++) {
            for (int by = 0; by < n_bins; by++) {
                int pxy = joint[bx*n_bins+by];
                if (pxy == 0) continue;
                int px = x_hist[bx], py = y_hist[by];
                if (px > 0 && py > 0)
                    mi += (double)pxy / N * log((double)pxy * N / ((double)px * py));
            }
        }
        op[j] = (float)mi;
    }
    free(joint); free(x_hist); free(y_hist);
    if (Xc != X) tensor_free(Xc);
    if (yc != y) tensor_free(yc);
    return out;
}

/* 23.10  Spearman rank correlation: each column of X [N,D] vs y [N].
 *        Returns [D] float tensor. */
static float* _rank_vals_ptr = NULL; /* thread-unsafe helper for qsort comparator */
static int _rank_idx_cmp(const void* a, const void* b) {
    int ia = *(int*)a, ib = *(int*)b;
    return (_rank_vals_ptr[ia] > _rank_vals_ptr[ib]) - (_rank_vals_ptr[ia] < _rank_vals_ptr[ib]);
}
static void _rank_array(float* vals, float* ranks, int n) {
    int* idx = (int*)malloc((size_t)n * sizeof(int));
    if (!idx) return;
    for (int i = 0; i < n; i++) idx[i] = i;
    _rank_vals_ptr = vals;
    qsort(idx, (size_t)n, sizeof(int), _rank_idx_cmp);
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n - 1 && vals[idx[j]] == vals[idx[j+1]]) j++;
        float avg_rank = ((float)i + (float)j) / 2.0f + 1.0f;
        for (int k = i; k <= j; k++) ranks[idx[k]] = avg_rank;
        i = j + 1;
    }
    free(idx);
}

Tensor* tensor_spearman_cols(Tensor* X, Tensor* y) {
    if (!X || !y || X->ndim != 2) TENSOR_ERROR("tensor_spearman_cols: X must be [N,D]");
    int N = X->shape[0], D = X->shape[1];
    if ((int)y->total_size != N) TENSOR_ERROR("tensor_spearman_cols: y length mismatch");
    Tensor* Xc = tensor_is_contiguous(X) ? X : tensor_copy(X);
    Tensor* yc = tensor_is_contiguous(y) ? y : tensor_copy(y);
    Tensor* out = tensor_zeros(1, (int[]){D});
    if (!out) { if (Xc!=X) tensor_free(Xc); if (yc!=y) tensor_free(yc); TENSOR_ERROR("OOM"); }
    float* xp = F32(Xc), *yp = F32(yc), *op = F32(out);
    float* x_buf = (float*)malloc((size_t)N * sizeof(float));
    float* x_rank = (float*)malloc((size_t)N * sizeof(float));
    float* y_rank = (float*)malloc((size_t)N * sizeof(float));
    if (!x_buf || !x_rank || !y_rank) {
        free(x_buf); free(x_rank); free(y_rank);
        if (Xc!=X) tensor_free(Xc); if (yc!=y) tensor_free(yc); tensor_free(out);
        TENSOR_ERROR("OOM");
    }
    _rank_array(yp, y_rank, N);
    for (int j = 0; j < D; j++) {
        for (int i = 0; i < N; i++) x_buf[i] = xp[i*D+j];
        _rank_array(x_buf, x_rank, N);
        double sx=0, sy=0, sxy=0, sx2=0, sy2=0;
        for (int i = 0; i < N; i++) {
            double rx = x_rank[i], ry = y_rank[i];
            sx += rx; sy += ry; sxy += rx*ry; sx2 += rx*rx; sy2 += ry*ry;
        }
        double num = N*sxy - sx*sy;
        double den = sqrt((N*sx2 - sx*sx) * (N*sy2 - sy*sy));
        op[j] = (den < 1e-12) ? 0.0f : (float)(num/den);
    }
    free(x_buf); free(x_rank); free(y_rank);
    if (Xc != X) tensor_free(Xc);
    if (yc != y) tensor_free(yc);
    return out;
}

/* 23.11  Class imbalance ratio: max_class_count / min_class_count.
 *        y must be integer-valued float (class labels). */
float tensor_class_imbalance_ratio(Tensor* y) {
    if (!y || y->total_size == 0) return 1.0f;
    Tensor* binned = tensor_bincount(y);
    if (!binned) return 1.0f;
    float mn = tensor_min(binned), mx = tensor_max(binned);
    tensor_free(binned);
    return (mn < 1.0f) ? mx : mx / mn;
}

/* 23.12  Low-variance feature mask: returns [D] bool tensor (1 = keep, 0 = drop).
 *        threshold: minimum variance to keep. */
Tensor* tensor_variance_threshold_mask(Tensor* X, float threshold) {
    if (!X || X->ndim != 2) TENSOR_ERROR("tensor_variance_threshold_mask: X must be [N,D]");
    int N = X->shape[0], D = X->shape[1];
    Tensor* Xc = tensor_is_contiguous(X) ? X : tensor_copy(X);
    Tensor* out = tensor_zeros(1, (int[]){D});
    if (!out) { if (Xc!=X) tensor_free(Xc); TENSOR_ERROR("OOM"); }
    float* xp = F32(Xc), *op = F32(out);
    for (int j = 0; j < D; j++) {
        double s = 0.0, s2 = 0.0; int cnt = 0;
        for (int i = 0; i < N; i++) {
            float v = xp[i*D+j];
            if (!_f32_is_nan(v)) { s += v; s2 += (double)v*v; cnt++; }
        }
        double var = (cnt > 1) ? s2/cnt - (s/cnt)*(s/cnt) : 0.0;
        op[j] = (var >= threshold) ? 1.0f : 0.0f;
    }
    if (Xc != X) tensor_free(Xc);
    return out;
}

/* 23.13  Redundancy clustering: returns [D] int tensor of cluster ids based on
 *        absolute correlation threshold.  Features with |r| > threshold are
 *        grouped into the same cluster (greedy single-linkage). */
Tensor* tensor_redundancy_clusters(Tensor* X, float threshold) {
    if (!X || X->ndim != 2) TENSOR_ERROR("tensor_redundancy_clusters: X must be [N,D]");
    int D = X->shape[1];
    Tensor* corr = tensor_correlation_matrix(X);
    if (!corr) TENSOR_ERROR("tensor_redundancy_clusters: corr failed");
    Tensor* out = tensor_create_dtype(1, (int[]){D}, DTYPE_INT32);
    if (!out) { tensor_free(corr); TENSOR_ERROR("OOM"); }
    int* op = (int*)out->data;
    float* cp = F32(corr);
    for (int j = 0; j < D; j++) op[j] = j; /* initial: each feature is its own cluster */
    for (int j = 0; j < D; j++) {
        for (int k = j+1; k < D; k++) {
            if (fabsf(cp[j*D+k]) >= threshold) {
                /* union: assign k's cluster to j's */
                int old_id = op[k], new_id = op[j];
                if (old_id != new_id)
                    for (int m = 0; m < D; m++) if (op[m] == old_id) op[m] = new_id;
            }
        }
    }
    /* normalize cluster ids to 0..n_clusters-1 */
    int remap[4096]; memset(remap, -1, sizeof(remap));
    int next_id = 0;
    for (int j = 0; j < D; j++) {
        int cid = op[j];
        if (cid < 4096) {
            if (remap[cid] < 0) remap[cid] = next_id++;
            op[j] = remap[cid];
        }
    }
    tensor_free(corr);
    return out;
}

/* 23.14  Nonlinearity score: correlation ratio eta^2 between numeric X column and target.
 *        Measures how well target variance is explained by feature-bin means.
 *        Returns [D] float tensor with eta^2 per feature. */
Tensor* tensor_nonlinearity_score(Tensor* X, Tensor* y, int n_bins) {
    if (!X || !y || X->ndim != 2) TENSOR_ERROR("tensor_nonlinearity_score: X must be [N,D]");
    int N = X->shape[0], D = X->shape[1];
    if ((int)y->total_size != N) TENSOR_ERROR("tensor_nonlinearity_score: y length mismatch");
    if (n_bins < 2) n_bins = 10;
    Tensor* Xc = tensor_is_contiguous(X) ? X : tensor_copy(X);
    Tensor* yc = tensor_is_contiguous(y) ? y : tensor_copy(y);
    Tensor* out = tensor_zeros(1, (int[]){D});
    if (!out) { if (Xc!=X) tensor_free(Xc); if (yc!=y) tensor_free(yc); TENSOR_ERROR("OOM"); }
    float* xp = F32(Xc), *yp = F32(yc), *op = F32(out);
    double y_mean = 0.0, y_ss = 0.0;
    for (int i = 0; i < N; i++) y_mean += yp[i];
    y_mean /= N;
    for (int i = 0; i < N; i++) y_ss += (yp[i]-y_mean)*(yp[i]-y_mean);
    if (y_ss < 1e-12) { if (Xc!=X) tensor_free(Xc); if (yc!=y) tensor_free(yc); return out; }
    double* bin_sum = (double*)malloc((size_t)n_bins * sizeof(double));
    int*    bin_cnt = (int*)   malloc((size_t)n_bins * sizeof(int));
    if (!bin_sum || !bin_cnt) {
        free(bin_sum); free(bin_cnt);
        if (Xc!=X) tensor_free(Xc); if (yc!=y) tensor_free(yc); tensor_free(out);
        TENSOR_ERROR("OOM");
    }
    for (int j = 0; j < D; j++) {
        float x_mn = xp[j], x_mx = x_mn;
        for (int i = 0; i < N; i++) { float v=xp[i*D+j]; if(v<x_mn)x_mn=v; if(v>x_mx)x_mx=v; }
        float x_rng = x_mx - x_mn;
        memset(bin_sum, 0, (size_t)n_bins * sizeof(double));
        memset(bin_cnt, 0, (size_t)n_bins * sizeof(int));
        for (int i = 0; i < N; i++) {
            int b = (x_rng < 1e-12f) ? 0 : (int)((xp[i*D+j]-x_mn)/x_rng*n_bins);
            if (b >= n_bins) b = n_bins-1;
            bin_sum[b] += yp[i]; bin_cnt[b]++;
        }
        double between_ss = 0.0;
        for (int b = 0; b < n_bins; b++) {
            if (bin_cnt[b] == 0) continue;
            double bm = bin_sum[b] / bin_cnt[b];
            between_ss += bin_cnt[b] * (bm - y_mean) * (bm - y_mean);
        }
        op[j] = (float)(between_ss / y_ss);
    }
    free(bin_sum); free(bin_cnt);
    if (Xc != X) tensor_free(Xc);
    if (yc != y) tensor_free(yc);
    return out;
}

/* ============================================================================
 * 24. HPC ESTIMATOR KERNELS
 * ========================================================================== */

/* 24.1  One-hot encoding: [N] float32 indices → [N,K] float32.
 * OpenMP parallel over N; zero intermediate allocation.                     */
Tensor* tensor_onehot(Tensor* indices, int K) {
    if (!indices || K < 1) { tensor_set_error("tensor_onehot: invalid args"); return NULL; }
    int N = (int)indices->total_size;
    int shape[2] = {N, K};
    Tensor* out = tensor_create(2, shape);   /* zero-initialized */
    if (!out) return NULL;
    float* ip = F32(indices);
    float* op = F32(out);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        int cls = (int)ip[i];
        if (cls >= 0 && cls < K) op[(size_t)i * K + cls] = 1.0f;
    }
    return out;
}

/* 24.2  KNN majority vote: [N,k] float32 label indices → [N] float32 class.
 * Per-row bincount + argmax. alloca scratch; safe when num_classes ≤ 4096. */
Tensor* tensor_knn_vote(Tensor* kLabels, int num_classes) {
    if (!kLabels || kLabels->ndim != 2 || num_classes < 1) {
        tensor_set_error("tensor_knn_vote: invalid args"); return NULL;
    }
    int N = kLabels->shape[0], k = kLabels->shape[1];
    Tensor* out = tensor_create(1, &N);
    if (!out) return NULL;
    float* lp = F32(kLabels);
    float* op = F32(out);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        int* counts = (int*)alloca((size_t)num_classes * sizeof(int));
        memset(counts, 0, (size_t)num_classes * sizeof(int));
        const float* row = lp + (size_t)i * k;
        for (int j = 0; j < k; j++) {
            int cls = (int)row[j];
            if (cls >= 0 && cls < num_classes) counts[cls]++;
        }
        int best = 0, best_cnt = counts[0];
        for (int c = 1; c < num_classes; c++)
            if (counts[c] > best_cnt) { best_cnt = counts[c]; best = c; }
        op[i] = (float)best;
    }
    return out;
}

/* 24.3  KMeans assignment: X[N,D] × centroids[K,D] → [N] cluster indices.
 * AVX2 fused distance; OpenMP parallel over N.                              */
Tensor* tensor_kmeans_assign(Tensor* X, Tensor* centroids) {
    if (!X || !centroids || X->ndim != 2 || centroids->ndim != 2 ||
        X->shape[1] != centroids->shape[1]) {
        tensor_set_error("tensor_kmeans_assign: shape mismatch"); return NULL;
    }
    int N = X->shape[0], D = X->shape[1], K = centroids->shape[0];
    Tensor* out = tensor_create(1, &N);
    if (!out) return NULL;
    const float* xp = F32(X);
    const float* cp = F32(centroids);
    float*       op = F32(out);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        const float* xi = xp + (size_t)i * D;
        float best_dist = FLT_MAX;
        int   best_k    = 0;
        for (int c = 0; c < K; c++) {
            const float* ck = cp + (size_t)c * D;
            float dist = 0.0f;
            int d = 0;
#ifdef __AVX2__
            __m256 vsum = _mm256_setzero_ps();
            for (; d <= D - 8; d += 8) {
                __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(xi + d),
                                            _mm256_loadu_ps(ck + d));
                vsum = _mm256_fmadd_ps(diff, diff, vsum);
            }
            __m128 lo = _mm256_castps256_ps128(vsum);
            __m128 hi = _mm256_extractf128_ps(vsum, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            dist = _mm_cvtss_f32(lo);
#endif
            for (; d < D; d++) { float df = xi[d] - ck[d]; dist += df * df; }
            if (dist < best_dist) { best_dist = dist; best_k = c; }
        }
        op[i] = (float)best_k;
    }
    return out;
}

/* 24.4  KMeans centroid update: X[N,D] × assignments[N] → [K,D].
 * Serial accumulate (race-free); AVX2 horizontal add per sample.
 * Empty clusters retain the corresponding row from old_centroids.           */
Tensor* tensor_kmeans_centroids(Tensor* X, Tensor* assignments, int K,
                                 Tensor* old_centroids) {
    if (!X || !assignments || K < 1 || X->ndim != 2) {
        tensor_set_error("tensor_kmeans_centroids: invalid args"); return NULL;
    }
    int N = X->shape[0], D = X->shape[1];
    int shape[2] = {K, D};
    Tensor* out = tensor_create(2, shape);   /* zero-initialized */
    if (!out) return NULL;
    const float* xp = F32(X);
    const float* ap = F32(assignments);
    float*       op = F32(out);
    int* cnt = (int*)calloc((size_t)K, sizeof(int));
    if (!cnt) { tensor_free(out); tensor_set_error("tensor_kmeans_centroids: OOM"); return NULL; }

    /* Accumulate — serial to avoid atomic overhead on small K */
    for (int i = 0; i < N; i++) {
        int c = (int)ap[i];
        if (c < 0 || c >= K) continue;
        cnt[c]++;
        float*       ck = op + (size_t)c * D;
        const float* xi = xp + (size_t)i * D;
        int d = 0;
#ifdef __AVX2__
        for (; d <= D - 8; d += 8)
            _mm256_storeu_ps(ck + d, _mm256_add_ps(
                _mm256_loadu_ps(ck + d), _mm256_loadu_ps(xi + d)));
#endif
        for (; d < D; d++) ck[d] += xi[d];
    }

    const float* op_old = old_centroids ? F32(old_centroids) : NULL;
    for (int c = 0; c < K; c++) {
        float* ck = op + (size_t)c * D;
        if (cnt[c] == 0) {
            if (op_old) memcpy(ck, op_old + (size_t)c * D, (size_t)D * sizeof(float));
        } else {
            float inv = 1.0f / cnt[c];
            int d = 0;
#ifdef __AVX2__
            __m256 vinv = _mm256_set1_ps(inv);
            for (; d <= D - 8; d += 8)
                _mm256_storeu_ps(ck + d, _mm256_mul_ps(_mm256_loadu_ps(ck + d), vinv));
#endif
            for (; d < D; d++) ck[d] *= inv;
        }
    }
    free(cnt);
    return out;
}

/* 24.5  Closed-form Ridge Regression: W = (X^T X + λI)^{-1} X^T y.
 * Uses symmetric positive-definite LAPACKE solver. Returns [D,1].          */
Tensor* tensor_ridge_solve(Tensor* X, Tensor* y, float lambda) {
    if (!X || !y || X->ndim != 2) {
        tensor_set_error("tensor_ridge_solve: X must be [N,D]"); return NULL;
    }
    int D = X->shape[1];

    /* XtX = X^T @ X  [D,D] */
    Tensor* XtX = tensor_matmul_ex(X, X, true, false);
    if (!XtX) return NULL;
    float* p = F32(XtX);
    for (int i = 0; i < D; i++) p[(size_t)i * D + i] += lambda;

    /* Xty = X^T @ y  [D,1] */
    int y_owned = (y->ndim == 1);
    Tensor* y2  = y_owned ? tensor_expand_dims(y, 1) : y;
    Tensor* Xty = tensor_matmul_ex(X, y2, true, false);
    if (y_owned) tensor_free(y2);
    if (!Xty)  { tensor_free(XtX); return NULL; }

    Tensor* w = tensor_solve(XtX, Xty);
    tensor_free(XtX);
    tensor_free(Xty);
    return w;   /* [D,1] */
}

/* 24.6  Copy one tree's flat node arrays into a pre-allocated ensemble buffer.
 * dest: [T * max_nodes] tensor (feats/thresh/lefts/rights).
 * tree_idx: 0-based tree index. max_nodes: nodes per tree.
 * src: [max_nodes] per-tree scratch tensor.
 * Replaces a PHP for-loop of max_nodes FFI float reads with a single memcpy. */
void tensor_gbdt_collect_tree(Tensor* dest, int tree_idx, int max_nodes, Tensor* src) {
    if (!dest || !src || max_nodes < 1) return;
    float* dp = F32(dest) + (size_t)tree_idx * max_nodes;
    memcpy(dp, F32(src), (size_t)max_nodes * sizeof(float));
}

/* 24.7  Isolation Forest batch scoring (all-C, OpenMP parallel over N).
 * X:          [N,D]            float32 test samples
 * feats_flat: [T * max_nodes]  float32 (feature index; -1 = leaf/sentinel)
 * thresh_flat:[T * max_nodes]  float32 (split threshold)
 * lefts_flat: [T * max_nodes]  float32 (left child index; -1 = leaf)
 * rights_flat:[T * max_nodes]  float32 (right child index)
 * lsize_flat: [T * max_nodes]  float32 (leaf sample size for c() correction)
 * tree_sizes: [T]              float32 (nodes used per tree)
 * c_norm:     c(sample_size)   normalising constant
 * Returns: [N] float32 anomaly scores in [0,1].                             */
Tensor* tensor_iforest_score(Tensor* X,
                              Tensor* feats_flat,  Tensor* thresh_flat,
                              Tensor* lefts_flat,  Tensor* rights_flat,
                              Tensor* lsize_flat,  Tensor* tree_sizes,
                              float c_norm) {
    if (!X || X->ndim != 2 || !feats_flat) {
        tensor_set_error("tensor_iforest_score: invalid args"); return NULL;
    }
    int N = X->shape[0], D = X->shape[1];
    int T = (int)tree_sizes->total_size;
    int max_nodes = (T > 0) ? (int)(feats_flat->total_size / T) : 1;
    if (c_norm <= 0.0f) c_norm = 1.0f;

    Tensor* out = tensor_create(1, &N);
    if (!out) return NULL;
    const float* xp  = F32(X);
    const float* fp  = F32(feats_flat);
    const float* tp  = F32(thresh_flat);
    const float* lp  = F32(lefts_flat);
    const float* rp  = F32(rights_flat);
    const float* szp = F32(lsize_flat);
    const float* tsp = F32(tree_sizes);
    float*       op  = F32(out);

#define _IF_C(n) ((n) <= 1 ? 0.0f : (n) == 2 ? 1.0f : \
    (2.0f * ((float)log((float)((n)-1)) + 0.5772156649f) - 2.0f * (float)((n)-1) / (float)(n)))

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        const float* xi = xp + (size_t)i * D;
        float path_sum = 0.0f;
        for (int t = 0; t < T; t++) {
            int used = (int)tsp[t];
            if (used <= 0) continue;
            const float* tf  = fp  + (size_t)t * max_nodes;
            const float* tt  = tp  + (size_t)t * max_nodes;
            const float* tl  = lp  + (size_t)t * max_nodes;
            const float* tr  = rp  + (size_t)t * max_nodes;
            const float* tsz = szp + (size_t)t * max_nodes;
            int node = 0;
            float length = 0.0f;
            while (node >= 0 && node < used) {
                int feat = (int)tf[node];
                if (feat < 0) break;                          /* leaf sentinel */
                float val = (feat < D) ? xi[feat] : 0.0f;
                node = (val < tt[node]) ? (int)tl[node] : (int)tr[node];
                length += 1.0f;
            }
            if (node >= 0 && node < max_nodes)
                length += _IF_C((int)szp[(size_t)t * max_nodes + (node >= 0 ? node : 0)]);
            path_sum += length;
        }
        float avg = (T > 0) ? path_sum / T : 0.0f;
        op[i] = powf(2.0f, -avg / c_norm);
    }
#undef _IF_C
    return out;
}

// ============================================================================
// 25. HPC ESTIMATOR KERNELS — BATCH 2
// ============================================================================

/* 25.1  Bootstrap sampling with replacement: returns [N] float32 indices in [0, N-1].
 *       Replaces a PHP for-loop of N mt_rand() calls.                        */
Tensor* tensor_bootstrap_indices(int N) {
    if (N < 1) { tensor_set_error("tensor_bootstrap_indices: N < 1"); return NULL; }
    Tensor* out = tensor_create(1, &N);
    if (!out) return NULL;
    float* op = F32(out);
#pragma omp parallel
    {
        unsigned seed = (unsigned)(omp_get_thread_num() * 1234567u + 98765u + (unsigned)N);
#pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            op[i] = (float)(rand_r(&seed) % N);
        }
    }
    return out;
}

/* 25.2  Majority-vote over T tree predictions.
 *       votes: [N, T] float32 integer class labels.
 *       Returns [N] float32 majority-class label.                            */
Tensor* tensor_matrix_vote(Tensor* votes, int num_classes) {
    if (!votes || votes->ndim != 2 || num_classes < 1) {
        tensor_set_error("tensor_matrix_vote: invalid args"); return NULL;
    }
    int N = votes->shape[0], T = votes->shape[1];
    Tensor* out = tensor_create(1, &N);
    if (!out) return NULL;
    const float* vp = F32(votes);
    float*       op = F32(out);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        /* stack-allocate count buffer: safe for num_classes ≤ 512 */
        int  stack_cnt[512];
        int* cnt = (num_classes <= 512) ? stack_cnt : (int*)malloc(num_classes * sizeof(int));
        if (!cnt) { op[i] = 0.0f; continue; }
        memset(cnt, 0, num_classes * sizeof(int));
        const float* row = vp + (size_t)i * T;
        for (int t = 0; t < T; t++) {
            int c = (int)row[t];
            if (c >= 0 && c < num_classes) cnt[c]++;
        }
        int best = 0, bestC = -1;
        for (int c = 0; c < num_classes; c++) {
            if (cnt[c] > bestC) { bestC = cnt[c]; best = c; }
        }
        op[i] = (float)best;
        if (num_classes > 512) free(cnt);
    }
    return out;
}

/* 25.3  All-C CART split search.
 *       X: [N, D], y: [N] float32 integer labels, feature_indices: [F] float32.
 *       Evaluates num_thresholds uniformly-spaced interior points per feature.
 *       Returns [N + 2] float32:
 *         [0]    = best feature index (-1 if no valid split found)
 *         [1]    = best threshold
 *         [2..N+1] = left mask (1.0 = left child, 0.0 = right child)         */
Tensor* tensor_cart_find_split(Tensor* X, Tensor* y,
                                Tensor* feature_indices, int num_thresholds) {
    if (!X || X->ndim != 2 || !y || !feature_indices || num_thresholds < 1) {
        tensor_set_error("tensor_cart_find_split: invalid args"); return NULL;
    }
    int N = X->shape[0], D = X->shape[1];
    int F = (int)feature_indices->total_size;
    const float* xp = F32(X);
    const float* yp = F32(y);
    const float* fp = F32(feature_indices);

    /* Infer num_classes from max label */
    int num_classes = 0;
    for (int i = 0; i < N; i++) {
        int c = (int)yp[i];
        if (c + 1 > num_classes) num_classes = c + 1;
    }

    int sz = N + 2;
    Tensor* out = tensor_create(1, &sz);
    if (!out) return NULL;
    float* op = F32(out);

    if (num_classes < 2) {
        op[0] = -1.0f; op[1] = 0.0f;
        memset(op + 2, 0, (size_t)N * sizeof(float));
        return out;
    }

    float* best_mask = (float*)malloc((size_t)N * sizeof(float));
    float* mask_buf  = (float*)malloc((size_t)N * sizeof(float));
    int*   cnt       = (int*)malloc((size_t)num_classes * 2 * sizeof(int));
    if (!best_mask || !mask_buf || !cnt) {
        free(best_mask); free(mask_buf); free(cnt);
        tensor_set_error("tensor_cart_find_split: OOM"); tensor_free(out); return NULL;
    }

    float best_gini  = (float)INFINITY;
    int   best_feat  = -1;
    float best_thresh = 0.0f;

    for (int fi = 0; fi < F; fi++) {
        int feat = (int)fp[fi];
        if (feat < 0 || feat >= D) continue;

        float fmin = (float)INFINITY, fmax = -(float)INFINITY;
        for (int i = 0; i < N; i++) {
            float v = xp[(size_t)i * D + feat];
            if (v < fmin) fmin = v;
            if (v > fmax) fmax = v;
        }
        if (fmin >= fmax) continue;

        float step = (fmax - fmin) / (float)(num_thresholds + 1);

        for (int t = 1; t <= num_thresholds; t++) {
            float thresh = fmin + step * t;
            memset(cnt, 0, (size_t)num_classes * 2 * sizeof(int));
            int nLeft = 0, nRight = 0;

            for (int i = 0; i < N; i++) {
                int c = (int)yp[i];
                if (xp[(size_t)i * D + feat] < thresh) {
                    mask_buf[i] = 1.0f; nLeft++;
                    if (c >= 0 && c < num_classes) cnt[c]++;
                } else {
                    mask_buf[i] = 0.0f; nRight++;
                    if (c >= 0 && c < num_classes) cnt[num_classes + c]++;
                }
            }
            if (nLeft == 0 || nRight == 0) continue;

            float ssL = 0.0f, ssR = 0.0f;
            float invL = 1.0f / nLeft, invR = 1.0f / nRight;
            for (int c = 0; c < num_classes; c++) {
                float pL = cnt[c] * invL, pR = cnt[num_classes + c] * invR;
                ssL += pL * pL; ssR += pR * pR;
            }
            float gini = ((float)nLeft / N) * (1.0f - ssL)
                       + ((float)nRight / N) * (1.0f - ssR);

            if (gini < best_gini) {
                best_gini   = gini;
                best_feat   = feat;
                best_thresh = thresh;
                memcpy(best_mask, mask_buf, (size_t)N * sizeof(float));
            }
        }
    }

    free(mask_buf); free(cnt);
    op[0] = (float)best_feat;
    op[1] = best_thresh;
    if (best_feat >= 0) {
        memcpy(op + 2, best_mask, (size_t)N * sizeof(float));
    } else {
        memset(op + 2, 0, (size_t)N * sizeof(float));
    }
    free(best_mask);
    return out;
}

/* 25.4  Fused (Elastic)Net mini-batch SGD step.
 *       Updates W [D,1] and bias [1] in-place.
 *       l1_ratio = 1.0 → Lasso, 0.0 → Ridge SGD, 0.5 → ElasticNet.
 *       Uses BLAS sgemv for the two dominant O(N*D) products.                */
void tensor_lasso_sgd_step(Tensor* X, Tensor* y, Tensor* W, Tensor* bias_t,
                            float alpha, float lr, float l1_ratio) {
    if (!X || X->ndim != 2 || !y || !W || !bias_t) return;
    int N = X->shape[0], D = X->shape[1];
    const float* xp = F32(X);
    const float* yp = F32(y);
    float*       wp = F32(W);
    float*       bp = F32(bias_t);
    float inv_n = 1.0f / (float)N;

    /* z[i] = X[i,:] · W + b − y[i]  (residuals)  */
    float* z = (float*)malloc((size_t)N * sizeof(float));
    if (!z) return;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, N, D, 1.0f, xp, D, wp, 1, 0.0f, z, 1);
    for (int i = 0; i < N; i++) z[i] += bp[0] - yp[i];

    /* dW = X^T · z / N  */
    float* dw = (float*)malloc((size_t)D * sizeof(float));
    if (!dw) { free(z); return; }
    cblas_sgemv(CblasRowMajor, CblasTrans, N, D, inv_n, xp, D, z, 1, 0.0f, dw, 1);

    /* Apply penalties and update W */
    for (int d = 0; d < D; d++) {
        float g = dw[d];
        if (l1_ratio > 0.0f) g += alpha * l1_ratio * (wp[d] >= 0.0f ? 1.0f : -1.0f);
        if (l1_ratio < 1.0f) g += alpha * (1.0f - l1_ratio) * wp[d];
        wp[d] -= lr * g;
    }
    /* Bias gradient = mean(z) */
    float db = 0.0f;
    for (int i = 0; i < N; i++) db += z[i];
    bp[0] -= lr * db * inv_n;

    free(z); free(dw);
}

/* 25.5  Fused Gaussian Naive Bayes log-likelihood.
 *       X: [N, D]  means_KD: [K, D]  vars_KD: [K, D]
 *       log_norms_K: [K]  where log_norms[k] = log_prior[k] − 0.5·sum(log(2π·var[k,:]))
 *       Out[i, k] = log_norms[k] − 0.5 · Σ_d (X[i,d]−means[k,d])² / vars[k,d]
 *       Returns [N, K].                                                       */
Tensor* tensor_gnb_log_likelihood(Tensor* X, Tensor* means_KD,
                                   Tensor* vars_KD,  Tensor* log_norms_K) {
    if (!X || X->ndim != 2 || !means_KD || means_KD->ndim != 2 ||
        !vars_KD || !log_norms_K) {
        tensor_set_error("tensor_gnb_log_likelihood: invalid args"); return NULL;
    }
    int N = X->shape[0], D = X->shape[1];
    int K = means_KD->shape[0];
    int out_shape[2] = {N, K};
    Tensor* out = tensor_create(2, out_shape);
    if (!out) return NULL;

    const float* xp  = F32(X);
    const float* mp  = F32(means_KD);
    const float* vp  = F32(vars_KD);
    const float* lnp = F32(log_norms_K);
    float*       op  = F32(out);

#pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            const float* xi = xp + (size_t)i * D;
            const float* mk = mp + (size_t)k * D;
            const float* vk = vp + (size_t)k * D;
            float sum = 0.0f;
            int d = 0;
#ifdef __AVX2__
            for (; d <= D - 8; d += 8) {
                __m256 xv   = _mm256_loadu_ps(xi + d);
                __m256 mv   = _mm256_loadu_ps(mk + d);
                __m256 vv   = _mm256_loadu_ps(vk + d);
                __m256 diff = _mm256_sub_ps(xv, mv);
                __m256 sq   = _mm256_mul_ps(diff, diff);
                __m256 res  = _mm256_div_ps(sq, vv);
                __m128 lo   = _mm256_extractf128_ps(res, 0);
                __m128 hi   = _mm256_extractf128_ps(res, 1);
                __m128 s    = _mm_add_ps(lo, hi);
                s = _mm_hadd_ps(s, s);
                s = _mm_hadd_ps(s, s);
                sum += _mm_cvtss_f32(s);
            }
#endif
            for (; d < D; d++) {
                float df = xi[d] - mk[d];
                sum += (df * df) / vk[d];
            }
            op[(size_t)i * K + k] = lnp[k] - 0.5f * sum;
        }
    }
    return out;
}

// ─── Section 26: Multiclass GBDT Kernels ─────────────────────────────────────
// Zero-copy multiclass extension of the binary GBDT engine.
// All per-sample math stays in C; PHP outer loop calls one C function per tree.

/* 26.1  Broadcast [K] base scores into every row of a pre-allocated [N,K] tensor.
 *       Used to initialise the prediction matrix before boosting rounds. */
void tensor_gbdt_init_preds_mc(Tensor* out_NK, Tensor* base_K) {
    if (!out_NK || !base_K || out_NK->ndim != 2) return;
    int N = out_NK->shape[0], K = out_NK->shape[1];
    if ((int)base_K->total_size != K) return;
    float* op = F32(out_NK);
    const float* bp = F32(base_K);
    #pragma omp parallel for schedule(static) if(N > 4096)
    for (int i = 0; i < N; i++) {
        float* row = op + (size_t)i * K;
        for (int k = 0; k < K; k++) row[k] = bp[k];
    }
}

/* 26.2  Softmax cross-entropy gradients and hessians for multiclass GBDT.
 *
 * raw_NK : [N, K] FLOAT32 — raw logit scores (current predictions)
 * y_N    : [N]   INT32    — integer class labels in [0, K)
 * out_g  : [N, K] FLOAT32 — output gradients  (pre-allocated by caller)
 * out_h  : [N, K] FLOAT32 — output hessians   (pre-allocated by caller)
 *
 * For sample i, class k:
 *   p_k = softmax(raw[i, :])_k
 *   g[i,k] = p_k - (y[i] == k)
 *   h[i,k] = max(p_k * (1 - p_k), 1e-16)              */
void tensor_gbdt_softmax_grad_hess(Tensor* raw_NK, Tensor* y_N,
                                   Tensor* out_g, Tensor* out_h) {
    if (!raw_NK || !y_N || !out_g || !out_h) return;
    if (raw_NK->ndim != 2) return;
    int N = raw_NK->shape[0], K = raw_NK->shape[1];
    const float*   rp = F32(raw_NK);
    const int32_t* yp = (const int32_t*)raw_NK->data; /* overridden below */
    /* y_N may be FLOAT32 (class-index float) or INT32 */
    bool y_is_f32 = (y_N->dtype == DTYPE_FLOAT32);
    const float*   yf = y_is_f32 ? F32(y_N)  : NULL;
    const int32_t* yi = y_is_f32 ? NULL       : (const int32_t*)y_N->data;
    float* gp = F32(out_g);
    float* hp = F32(out_h);

    #pragma omp parallel for schedule(static) if(N > 512)
    for (int i = 0; i < N; i++) {
        const float* row = rp + (size_t)i * K;
        float* grow = gp + (size_t)i * K;
        float* hrow = hp + (size_t)i * K;

        /* numerically stable softmax: subtract row max */
        float mx = row[0];
        for (int k = 1; k < K; k++) if (row[k] > mx) mx = row[k];

        float sum = 0.0f;
        for (int k = 0; k < K; k++) { grow[k] = expf(row[k] - mx); sum += grow[k]; }
        float inv_sum = 1.0f / sum;
        for (int k = 0; k < K; k++) grow[k] *= inv_sum;  /* grow[] now = p_k */

        int yi_val = y_is_f32 ? (int)yf[i] : (int)yi[i];
        for (int k = 0; k < K; k++) {
            float pk = grow[k];
            grow[k] = pk - (k == yi_val ? 1.0f : 0.0f);
            hrow[k] = fmaxf(pk * (1.0f - pk), 1e-16f);
        }
    }
}

/* 26.3  Multiclass GBDT tree training — column k of the [N,K] gradient matrix.
 *
 * Identical algorithm to tensor_gbdt_train_tree but reads g/h/preds at stride K.
 * No copies — reads directly from the interleaved [N,K] gradient tensor.
 *
 * bins   : [N, D] INT32
 * g_NK   : [N, K] FLOAT32 gradients (all classes, interleaved)
 * h_NK   : [N, K] FLOAT32 hessians
 * K      : number of classes
 * kc     : class column index to train (0 .. K-1)
 * preds_NK: [N, K] FLOAT32 — updated in-place for column kc only
 * Returns number of nodes used.                                               */

/* Internal histogram builders that stride by K (multiclass column slice). */
static void _build_hist_mc_serial(
    const int32_t* bins_flat, const float* gp, const float* hp,
    const int* indices, int count, int D, int Q, int K, int kc,
    float* hist_g, float* hist_h)
{
    memset(hist_g, 0, (size_t)D * Q * sizeof(float));
    memset(hist_h, 0, (size_t)D * Q * sizeof(float));
    for (int idx = 0; idx < count; idx++) {
        int si = indices[idx];
        float gi = gp[(size_t)si * K + kc];
        float hi = hp[(size_t)si * K + kc];
        const int32_t* bi = bins_flat + (size_t)si * D;
        for (int d = 0; d < D; d++) {
            size_t pos = (size_t)d * Q + bi[d];
            hist_g[pos] += gi;
            hist_h[pos] += hi;
        }
    }
}

static void _build_hist_mc_par(
    const int32_t* bins_flat, const float* gp, const float* hp,
    const int* indices, int count, int D, int Q, int K, int kc,
    float* thr_bufs, int max_threads,
    float* hist_g, float* hist_h)
{
    size_t DQ = (size_t)D * Q;
    if (count <= 8 * D) {
        _build_hist_mc_serial(bins_flat, gp, hp, indices, count, D, Q, K, kc, hist_g, hist_h);
        return;
    }
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid >= max_threads) tid = max_threads - 1;
        float* lg = thr_bufs + (size_t)tid * 2 * DQ;
        float* lh = lg + DQ;
        memset(lg, 0, 2 * DQ * sizeof(float));
        #pragma omp for schedule(static)
        for (int idx = 0; idx < count; idx++) {
            int si = indices[idx];
            float gi = gp[(size_t)si * K + kc];
            float hi = hp[(size_t)si * K + kc];
            const int32_t* bi = bins_flat + (size_t)si * D;
            for (int d = 0; d < D; d++) {
                size_t pos = (size_t)d * Q + bi[d];
                lg[pos] += gi;
                lh[pos] += hi;
            }
        }
    }
    memset(hist_g, 0, DQ * sizeof(float));
    memset(hist_h, 0, DQ * sizeof(float));
    for (int t = 0; t < max_threads; t++) {
        float* lg = thr_bufs + (size_t)t * 2 * DQ;
        float* lh = lg + DQ;
        for (size_t j = 0; j < DQ; j++) { hist_g[j] += lg[j]; hist_h[j] += lh[j]; }
    }
}

int tensor_gbdt_train_tree_mc(
    Tensor* bins, Tensor* g_NK, Tensor* h_NK, int K, int kc,
    int Q, int max_leaves,
    float lambda, float alpha, float gamma, float min_hess, float lr,
    Tensor* preds_NK,
    Tensor* out_feats, Tensor* out_thresholds,
    Tensor* out_lefts,  Tensor* out_rights)
{
    if (!bins || !g_NK || !h_NK || !preds_NK ||
        !out_feats || !out_thresholds || !out_lefts || !out_rights)
        return 0;
    if (bins->ndim != 2 || bins->dtype != DTYPE_INT32) return 0;
    if (kc < 0 || kc >= K) return 0;
    if (max_leaves < 1) max_leaves = 1;

    int N = bins->shape[0], D = bins->shape[1];
    int max_nodes = (int)out_feats->total_size;
    size_t DQ = (size_t)D * Q;
    int n_slots = max_leaves + 2;
    int max_threads = omp_get_max_threads();

    size_t hist_bytes = (size_t)n_slots * 2 * DQ * sizeof(float);
    size_t thr_bytes  = (size_t)max_threads * 2 * DQ * sizeof(float);
    float*    hist_pool  = (float*)safe_memalign(64, hist_bytes);
    float*    thr_bufs   = (float*)safe_memalign(64, thr_bytes);
    int*      sample_idx = (int*)calloc((size_t)N, sizeof(int));
    int*      tmp_idx    = (int*)calloc((size_t)N, sizeof(int));
    int*      slot_used  = (int*)calloc((size_t)n_slots, sizeof(int));
    _GBDTLeaf* pq        = (_GBDTLeaf*)calloc((size_t)(max_leaves + 1), sizeof(_GBDTLeaf));

    int ret = 0;
    if (!hist_pool || !thr_bufs || !sample_idx || !tmp_idx || !slot_used || !pq) {
        if (hist_pool) safe_free_size(hist_pool, hist_bytes);
        if (thr_bufs)  safe_free_size(thr_bufs,  thr_bytes);
        free(sample_idx); free(tmp_idx); free(slot_used); free(pq);
        return 0;
    }

    for (int i = 0; i < N; i++) sample_idx[i] = i;

#define _MC_HS_G(s)   (hist_pool + (size_t)(s) * 2 * DQ)
#define _MC_HS_H(s)   (hist_pool + (size_t)(s) * 2 * DQ + DQ)
#define _MC_ALLOC_SLOT(out) do {                                              \
    (out) = -1;                                                                \
    for (int _s = 0; _s < n_slots; _s++) {                                    \
        if (!slot_used[_s]) { slot_used[_s] = 1; (out) = _s; break; }        \
    }                                                                          \
} while(0)
#define _MC_FREE_SLOT(s)  (slot_used[(s)] = 0)

    float* fp = F32(out_feats); float* tp = F32(out_thresholds);
    float* lp = F32(out_lefts); float* rp = F32(out_rights);
    for (int i = 0; i < max_nodes; i++) { fp[i]=-1.0f; tp[i]=0.0f; lp[i]=-1.0f; rp[i]=-1.0f; }

    const int32_t* bp = I32(bins);
    const float*   gp = F32(g_NK);
    const float*   hp2 = F32(h_NK);
    float*         pp = F32(preds_NK);

    int root_slot; _MC_ALLOC_SLOT(root_slot);
    _build_hist_mc_par(bp, gp, hp2, sample_idx, N, D, Q, K, kc,
                       thr_bufs, max_threads, _MC_HS_G(root_slot), _MC_HS_H(root_slot));

    float root_sg = 0.0f, root_sh = 0.0f;
    for (int i = 0; i < N; i++) {
        root_sg += gp[(size_t)i * K + kc];
        root_sh += hp2[(size_t)i * K + kc];
    }

    int next_node = 1, pq_sz = 0, splits_done = 0;
    int max_splits = max_leaves - 1;

    int rf = -1, rb = -1; float rg_gain = -1.0f;
    if (root_sh >= min_hess)
        _best_split_raw(_MC_HS_G(root_slot), _MC_HS_H(root_slot), D, Q,
                        root_sg, root_sh, lambda, alpha, gamma, &rf, &rb, &rg_gain);

    if (rf < 0 || rg_gain <= 0.0f || max_splits == 0) {
        float lv    = _gbdt_leaf_val_l1(root_sg, root_sh, lambda, alpha);
        tp[0]       = lr * lv; fp[0] = -1.0f;
        float delta = lr * lv;
        for (int i = 0; i < N; i++) pp[(size_t)i * K + kc] += delta;
        _MC_FREE_SLOT(root_slot);
        ret = 1;
    } else {
        _GBDTLeaf root_entry = {0, root_slot, 0, N, root_sg, root_sh, rg_gain, rf, rb};
        _pq_push(pq, &pq_sz, root_entry);

        while (pq_sz > 0 && splits_done < max_splits) {
            _GBDTLeaf leaf = _pq_pop(pq, &pq_sz);
            splits_done++;

            int feat = leaf.best_feat, split_bin = leaf.best_bin;
            int start = leaf.idx_start, count = leaf.idx_count;
            const int* cur = sample_idx + start;

            int l_count = 0, r_tail = count - 1;
            for (int ci = 0; ci < count; ci++) {
                int si = cur[ci];
                if (bp[(size_t)si * D + feat] <= split_bin)
                    tmp_idx[l_count++] = si;
                else
                    tmp_idx[r_tail--] = si;
            }
            int r_count = count - l_count;
            memcpy(sample_idx + start, tmp_idx, (size_t)count * sizeof(int));

            int l_start = start, r_start = start + l_count;
            int left_nid = next_node++, right_nid = next_node++;

            fp[leaf.node_id] = (float)feat;
            tp[leaf.node_id] = (float)split_bin;
            lp[leaf.node_id] = (float)left_nid;
            rp[leaf.node_id] = (float)right_nid;

            float l_sg = 0.0f, l_sh = 0.0f;
            for (int ci = 0; ci < l_count; ci++) {
                int si = sample_idx[l_start + ci];
                l_sg += gp[(size_t)si * K + kc];
                l_sh += hp2[(size_t)si * K + kc];
            }
            float r_sg = leaf.sum_g - l_sg, r_sh = leaf.sum_h - l_sh;

            int s_slot, o_slot;
            _MC_ALLOC_SLOT(s_slot); _MC_ALLOC_SLOT(o_slot);

            int  small_nid, large_nid, small_sl, large_sl;
            int  small_start, small_count, large_start, large_count;
            float small_sg, small_sh, large_sg, large_sh;

            if (l_count <= r_count) {
                _build_hist_mc_par(bp, gp, hp2, sample_idx + l_start, l_count,
                                   D, Q, K, kc, thr_bufs, max_threads,
                                   _MC_HS_G(s_slot), _MC_HS_H(s_slot));
                _hist_subtract_raw(_MC_HS_G(leaf.hist_slot), _MC_HS_H(leaf.hist_slot),
                                   _MC_HS_G(s_slot), _MC_HS_H(s_slot),
                                   _MC_HS_G(o_slot), _MC_HS_H(o_slot), DQ);
                small_nid = left_nid;  small_sl = s_slot; small_start = l_start; small_count = l_count; small_sg = l_sg; small_sh = l_sh;
                large_nid = right_nid; large_sl = o_slot; large_start = r_start; large_count = r_count; large_sg = r_sg; large_sh = r_sh;
            } else {
                _build_hist_mc_par(bp, gp, hp2, sample_idx + r_start, r_count,
                                   D, Q, K, kc, thr_bufs, max_threads,
                                   _MC_HS_G(s_slot), _MC_HS_H(s_slot));
                _hist_subtract_raw(_MC_HS_G(leaf.hist_slot), _MC_HS_H(leaf.hist_slot),
                                   _MC_HS_G(s_slot), _MC_HS_H(s_slot),
                                   _MC_HS_G(o_slot), _MC_HS_H(o_slot), DQ);
                small_nid = right_nid; small_sl = s_slot; small_start = r_start; small_count = r_count; small_sg = r_sg; small_sh = r_sh;
                large_nid = left_nid;  large_sl = o_slot; large_start = l_start; large_count = l_count; large_sg = l_sg; large_sh = l_sh;
            }
            _MC_FREE_SLOT(leaf.hist_slot);

            int   child_nid[2]   = {small_nid,   large_nid};
            int   child_sl[2]    = {small_sl,     large_sl};
            int   child_start[2] = {small_start,  large_start};
            int   child_count[2] = {small_count,  large_count};
            float child_sg[2]    = {small_sg,     large_sg};
            float child_sh[2]    = {small_sh,     large_sh};

            for (int ci = 0; ci < 2; ci++) {
                int   cnid = child_nid[ci];
                int   csl  = child_sl[ci];
                int   cs   = child_start[ci];
                int   cc   = child_count[ci];
                float csg  = child_sg[ci], csh = child_sh[ci];

                if (csh < min_hess || cc <= 0 || splits_done >= max_splits) {
                    float lv    = _gbdt_leaf_val_l1(csg, csh, lambda, alpha);
                    fp[cnid] = -1.0f; tp[cnid] = lr * lv;
                    float delta = lr * lv;
                    for (int ck = 0; ck < cc; ck++)
                        pp[(size_t)sample_idx[cs + ck] * K + kc] += delta;
                    _MC_FREE_SLOT(csl);
                    continue;
                }
                int cf = -1, cb = -1; float cg_gain = -1.0f;
                _best_split_raw(_MC_HS_G(csl), _MC_HS_H(csl), D, Q,
                                csg, csh, lambda, alpha, gamma, &cf, &cb, &cg_gain);
                if (cf < 0 || cg_gain <= 0.0f) {
                    float lv    = _gbdt_leaf_val_l1(csg, csh, lambda, alpha);
                    fp[cnid] = -1.0f; tp[cnid] = lr * lv;
                    float delta = lr * lv;
                    for (int ck = 0; ck < cc; ck++)
                        pp[(size_t)sample_idx[cs + ck] * K + kc] += delta;
                    _MC_FREE_SLOT(csl);
                } else {
                    _GBDTLeaf cl = {cnid, csl, cs, cc, csg, csh, cg_gain, cf, cb};
                    _pq_push(pq, &pq_sz, cl);
                }
            }
        }

        while (pq_sz > 0) {
            _GBDTLeaf leaf = _pq_pop(pq, &pq_sz);
            float lv    = _gbdt_leaf_val_l1(leaf.sum_g, leaf.sum_h, lambda, alpha);
            tp[leaf.node_id] = lr * lv; fp[leaf.node_id] = -1.0f;
            float delta = lr * lv;
            const int* idx = sample_idx + leaf.idx_start;
            for (int ck = 0; ck < leaf.idx_count; ck++)
                pp[(size_t)idx[ck] * K + kc] += delta;
            _MC_FREE_SLOT(leaf.hist_slot);
        }

        ret = next_node;
    }

#undef _MC_HS_G
#undef _MC_HS_H
#undef _MC_ALLOC_SLOT
#undef _MC_FREE_SLOT

    safe_free_size(hist_pool, hist_bytes);
    safe_free_size(thr_bufs,  thr_bytes);
    free(sample_idx); free(tmp_idx); free(slot_used); free(pq);
    return ret;
}

/* 26.4  Multiclass GBDT batch prediction.
 *
 * X_bins      : [N, D]         INT32  — pre-binned samples
 * feats       : [T*K, maxNodes] FLOAT32
 * thresholds  : [T*K, maxNodes] FLOAT32
 * lefts       : [T*K, maxNodes] FLOAT32
 * rights      : [T*K, maxNodes] FLOAT32
 * tree_sizes  : [T*K]          FLOAT32
 * base_scores : [K]            FLOAT32 — initial logit per class
 * K           : number of classes
 *
 * Trees are stored round-major: tree index tk belongs to class (tk % K),
 * round (tk / K).  This matches the order they are built in PHP.
 *
 * Returns [N, K] FLOAT32 raw logit scores (apply softmax for probabilities). */
Tensor* tensor_gbdt_predict_all_mc(
    Tensor* X_bins, Tensor* feats, Tensor* thresholds,
    Tensor* lefts,  Tensor* rights, Tensor* tree_sizes,
    Tensor* base_scores, int K)
{
    if (!X_bins || !feats || !thresholds || !lefts || !rights || !tree_sizes || !base_scores)
        return NULL;
    if (X_bins->ndim != 2 || K < 2) return NULL;

    int N  = X_bins->shape[0], D = X_bins->shape[1];
    int TK = feats->shape[0],  M = feats->shape[1];

    Tensor* xb_c = tensor_is_contiguous(X_bins) ? X_bins : tensor_copy(X_bins);
    Tensor* out  = tensor_create_uninitialized(2, (int[]){N, K}, DTYPE_FLOAT32);
    if (!out) { if (xb_c != X_bins) tensor_free(xb_c); return NULL; }

    /* Initialise each row with base_scores */
    float* op = F32(out);
    const float* bs = F32(base_scores);
    for (int i = 0; i < N; i++) {
        float* row = op + (size_t)i * K;
        for (int k = 0; k < K; k++) row[k] = bs[k];
    }

    const int32_t* bp = I32(xb_c);
    const float*   fp = F32(feats);
    const float*   tp = F32(thresholds);
    const float*   lp = F32(lefts);
    const float*   rp = F32(rights);
    const float*   sp = F32(tree_sizes);

    #pragma omp parallel for schedule(static) if(N > 1000)
    for (int i = 0; i < N; i++) {
        const int32_t* xi  = bp + (size_t)i * D;
        float*         oi  = op + (size_t)i * K;
        for (int tk = 0; tk < TK; tk++) {
            int kc = tk % K;
            const float* tf = fp + (size_t)tk * M;
            const float* tt = tp + (size_t)tk * M;
            const float* tl = lp + (size_t)tk * M;
            const float* tr = rp + (size_t)tk * M;
            int sz   = (int)sp[tk];
            int node = 0;
            while (node < sz) {
                int feat = (int)tf[node];
                if (feat < 0) { oi[kc] += tt[node]; break; }
                int bin_split = (int)tt[node];
                node = (xi[feat] <= bin_split) ? (int)tl[node] : (int)tr[node];
                if (node < 0) break;
            }
        }
    }

    if (xb_c != X_bins) tensor_free(xb_c);
    return out;
}

/* 25.6  Gather: out[i] = table[floor(indices[i])].
 *       Replaces PHP array_map label-remapping loops.
 *       indices: [N] float32, table: [K] float32.
 *       Returns [N].                                                          */
Tensor* tensor_gather_indices(Tensor* indices, Tensor* table) {
    if (!indices || !table) {
        tensor_set_error("tensor_gather_indices: null arg"); return NULL;
    }
    int N = (int)indices->total_size;
    int K = (int)table->total_size;
    Tensor* out = tensor_create(1, &N);
    if (!out) return NULL;
    const float* ip = F32(indices);
    const float* tp = F32(table);
    float*       op = F32(out);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        int idx = (int)ip[i];
        op[i] = (idx >= 0 && idx < K) ? tp[idx] : 0.0f;
    }
    return out;
}


/* ── Section 27: Column permutation for permutation importance ───────────── */

void tensor_permute_column(Tensor* X, int col, Tensor* backup) {
    if (!X || !backup) { tensor_set_error("tensor_permute_column: null arg"); return; }
    int N = X->shape[0], D = X->shape[1];
    if (col < 0 || col >= D) { tensor_set_error("tensor_permute_column: col out of range"); return; }
    float* xp = (float*)X->data;
    float* bp = (float*)backup->data;
    for (int i = 0; i < N; i++) bp[i] = xp[i*D + col];
    for (int i = N-1; i > 0; i--) {
        int j = (int)((double)rand() / ((double)RAND_MAX+1.0) * (i+1));
        float tmp = xp[i*D + col]; xp[i*D + col] = xp[j*D + col]; xp[j*D + col] = tmp;
    }
}

void tensor_restore_column(Tensor* X, int col, const Tensor* backup) {
    if (!X || !backup) return;
    int N = X->shape[0], D = X->shape[1];
    float* xp = (float*)X->data;
    const float* bp = (const float*)backup->data;
    for (int i = 0; i < N; i++) xp[i*D + col] = bp[i];
}

/* ============================================================================
 * 28. GPT TRAINING PRIMITIVES
 *   - GELU activation (tanh approx, forward + backward)
 *   - LayerNorm with learnable γ/β (forward + backward)
 *   - Causal masked multi-head attention (forward + backward)
 * ============================================================================ */

#define GELU_SQRT_2_PI  0.7978845608028654f
#define GELU_COEFF      0.044715f

/* ---------------------------------------------------------------------------
 * GELU forward: GELU(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
 * ------------------------------------------------------------------------- */
Tensor* tensor_gelu(Tensor* A) {
    if (!A || A->dtype != DTYPE_FLOAT32)
        TENSOR_ERROR("FATAL [GELU]: Requires FLOAT32.");
    if (!tensor_is_contiguous(A))
        TENSOR_ERROR("FATAL [GELU]: Input must be contiguous.");

    Tensor* out = tensor_create_uninitialized(A->ndim, A->shape, DTYPE_FLOAT32);
    if (!out) return NULL;

    size_t n = A->total_size;
    const float* __restrict src = (const float*)__builtin_assume_aligned(F32(A), 64);
    float*       __restrict dst = (float*)__builtin_assume_aligned(F32(out), 64);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        float x  = src[i];
        float u  = GELU_SQRT_2_PI * (x + GELU_COEFF * x * x * x);
        float th = tanhf(u);
        dst[i]   = 0.5f * x * (1.0f + th);
    }
    return out;
}

/* ---------------------------------------------------------------------------
 * GELU backward: dx = dout * GELU'(x)
 * GELU'(x) = 0.5*(1+tanh(u)) + 0.5*x*sech²(u)*√(2/π)*(1+3·c·x²)
 * ------------------------------------------------------------------------- */
Tensor* tensor_gelu_backward(Tensor* dOut, Tensor* X) {
    if (!dOut || !X || dOut->dtype != DTYPE_FLOAT32 || X->dtype != DTYPE_FLOAT32)
        TENSOR_ERROR("FATAL [GELUBwd]: Requires FLOAT32.");
    if (!tensor_is_contiguous(dOut) || !tensor_is_contiguous(X))
        TENSOR_ERROR("FATAL [GELUBwd]: Inputs must be contiguous.");
    if (dOut->total_size != X->total_size)
        TENSOR_ERROR("FATAL [GELUBwd]: Shape mismatch.");

    Tensor* dx = tensor_create_uninitialized(X->ndim, X->shape, DTYPE_FLOAT32);
    if (!dx) return NULL;

    size_t n = X->total_size;
    const float* __restrict dp = (const float*)__builtin_assume_aligned(F32(dOut), 64);
    const float* __restrict xp = (const float*)__builtin_assume_aligned(F32(X),    64);
    float*       __restrict op = (float*)__builtin_assume_aligned(F32(dx),   64);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        float x    = xp[i];
        float u    = GELU_SQRT_2_PI * (x + GELU_COEFF * x * x * x);
        float th   = tanhf(u);
        float sech2 = 1.0f - th * th;
        float du_dx = GELU_SQRT_2_PI * (1.0f + 3.0f * GELU_COEFF * x * x);
        float gelu_prime = 0.5f * (1.0f + th) + 0.5f * x * sech2 * du_dx;
        op[i] = dp[i] * gelu_prime;
    }
    return dx;
}

/* ---------------------------------------------------------------------------
 * LayerNorm forward
 * x:      [*, D]  FLOAT32   (rows = total_size / D)
 * weight: [D]     FLOAT32   learnable γ
 * bias:   [D]     FLOAT32   learnable β (NULL = no bias)
 * Returns new [*, D] FLOAT32 tensor
 * ------------------------------------------------------------------------- */
Tensor* tensor_layernorm_forward(Tensor* x, Tensor* weight, Tensor* bias, float eps) {
    if (!x || !weight) TENSOR_ERROR("FATAL [LNFwd]: x and weight must not be NULL.");
    if (x->dtype != DTYPE_FLOAT32 || weight->dtype != DTYPE_FLOAT32)
        TENSOR_ERROR("FATAL [LNFwd]: Requires FLOAT32.");
    if (!tensor_is_contiguous(x) || !tensor_is_contiguous(weight))
        TENSOR_ERROR("FATAL [LNFwd]: x and weight must be contiguous.");
    if (bias && (!tensor_is_contiguous(bias) || bias->dtype != DTYPE_FLOAT32))
        TENSOR_ERROR("FATAL [LNFwd]: bias must be contiguous FLOAT32.");

    int D    = x->shape[x->ndim - 1];
    int rows = (int)(x->total_size / (size_t)D);

    if ((int)weight->total_size != D)
        TENSOR_ERROR("FATAL [LNFwd]: weight size must match last dim of x.");
    if (bias && (int)bias->total_size != D)
        TENSOR_ERROR("FATAL [LNFwd]: bias size must match last dim of x.");

    Tensor* out = tensor_create_uninitialized(x->ndim, x->shape, DTYPE_FLOAT32);
    if (!out) return NULL;

    const float* __restrict xp  = (const float*)__builtin_assume_aligned(F32(x), 64);
    const float* __restrict wp  = (const float*)__builtin_assume_aligned(F32(weight), 64);
    const float* __restrict bp  = bias ? (const float*)F32(bias) : NULL;
    float*       __restrict op  = (float*)__builtin_assume_aligned(F32(out), 64);

#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float* __restrict row_x = xp + (size_t)r * D;
        float*       __restrict row_o = op + (size_t)r * D;

        /* mean */
        float sum = 0.0f;
        for (int j = 0; j < D; j++) sum += row_x[j];
        float mean = sum / (float)D;

        /* variance */
        float var = 0.0f;
        for (int j = 0; j < D; j++) {
            float d = row_x[j] - mean;
            var += d * d;
        }
        float rstd = 1.0f / sqrtf(var / (float)D + eps);

        /* normalize + scale + shift */
        for (int j = 0; j < D; j++) {
            float xhat = (row_x[j] - mean) * rstd;
            row_o[j]   = wp[j] * xhat + (bp ? bp[j] : 0.0f);
        }
    }
    return out;
}

/* ---------------------------------------------------------------------------
 * LayerNorm backward
 * dY:      [*, D]  upstream gradient
 * x:       [*, D]  original forward input
 * weight:  [D]     forward γ
 * eps:     float   same as forward
 * dWeight: [D]     gradient accumulator for γ (caller must zero first; +=)
 * dBias:   [D]     gradient accumulator for β (caller must zero first; +=; NULL ok)
 * Returns: [*, D]  gradient w.r.t. x
 * ------------------------------------------------------------------------- */
Tensor* tensor_layernorm_backward(Tensor* dY, Tensor* x, Tensor* weight, float eps,
                                   Tensor* dWeight, Tensor* dBias) {
    if (!dY || !x || !weight || !dWeight)
        TENSOR_ERROR("FATAL [LNBwd]: NULL pointer.");
    if (dY->dtype != DTYPE_FLOAT32 || x->dtype != DTYPE_FLOAT32 ||
        weight->dtype != DTYPE_FLOAT32 || dWeight->dtype != DTYPE_FLOAT32)
        TENSOR_ERROR("FATAL [LNBwd]: All tensors must be FLOAT32.");
    if (!tensor_is_contiguous(dY) || !tensor_is_contiguous(x) ||
        !tensor_is_contiguous(weight) || !tensor_is_contiguous(dWeight))
        TENSOR_ERROR("FATAL [LNBwd]: All tensors must be contiguous.");
    if (dY->total_size != x->total_size)
        TENSOR_ERROR("FATAL [LNBwd]: dY and x shape mismatch.");

    int D    = x->shape[x->ndim - 1];
    int rows = (int)(x->total_size / (size_t)D);

    Tensor* dx = tensor_create_uninitialized(x->ndim, x->shape, DTYPE_FLOAT32);
    if (!dx) return NULL;

    const float* __restrict dyp = (const float*)__builtin_assume_aligned(F32(dY), 64);
    const float* __restrict xp  = (const float*)__builtin_assume_aligned(F32(x), 64);
    const float* __restrict wp  = (const float*)__builtin_assume_aligned(F32(weight), 64);
    float*       __restrict dxp = (float*)__builtin_assume_aligned(F32(dx), 64);
    float*       __restrict dwp = (float*)__builtin_assume_aligned(F32(dWeight), 64);
    float*       __restrict dbp = dBias ? (float*)F32(dBias) : NULL;

    /* Serial over rows; dWeight/dBias accumulate without races.
     * dx per row is independent → can be parallelized separately if needed. */
    for (int r = 0; r < rows; r++) {
        const float* __restrict dy = dyp + (size_t)r * D;
        const float* __restrict xr = xp  + (size_t)r * D;
        float*       __restrict dxr= dxp + (size_t)r * D;

        /* recompute mean and rstd */
        float sum = 0.0f;
        for (int j = 0; j < D; j++) sum += xr[j];
        float mean = sum / (float)D;

        float var = 0.0f;
        for (int j = 0; j < D; j++) { float d = xr[j] - mean; var += d * d; }
        float rstd = 1.0f / sqrtf(var / (float)D + eps);

        /* c1 = Σ(dxhat·xhat)/D,  c2 = Σ(dxhat)/D  where dxhat = dy·w */
        float c1 = 0.0f, c2 = 0.0f;
        for (int j = 0; j < D; j++) {
            float xhat  = (xr[j] - mean) * rstd;
            float dxhat = dy[j] * wp[j];
            c1 += dxhat * xhat;
            c2 += dxhat;
            dwp[j] += dy[j] * xhat;
            if (dbp) dbp[j] += dy[j];
        }
        c1 /= (float)D;
        c2 /= (float)D;

        /* dx[j] = rstd * (dxhat[j] - c1·xhat[j] - c2) */
        for (int j = 0; j < D; j++) {
            float xhat  = (xr[j] - mean) * rstd;
            float dxhat = dy[j] * wp[j];
            dxr[j] = rstd * (dxhat - c1 * xhat - c2);
        }
    }

    return dx;
}

/* ---------------------------------------------------------------------------
 * Causal masked scaled-dot-product attention (multi-head, training)
 * q, k, v: [nH, T, hd]  FLOAT32
 * out:      [nH, T, hd]  FLOAT32 pre-allocated output
 * attn:     [nH, T, T]   FLOAT32 pre-allocated (NULL = don't save weights)
 *
 * Per head h:
 *   S[i,j] = Q[h,i,:] · K[h,j,:] / sqrt(hd),  causal mask: j > i → −∞
 *   A[h]   = softmax(S)
 *   out[h] = A[h] @ V[h]
 * ------------------------------------------------------------------------- */
void tensor_causal_attention(Tensor* out, Tensor* q, Tensor* k, Tensor* v, Tensor* attn) {
    if (!out || !q || !k || !v)  { tensor_set_error("FATAL [CausalAttn]: NULL pointer."); return; }
    if (q->ndim != 3 || k->ndim != 3 || v->ndim != 3 || out->ndim != 3)
        { tensor_set_error("FATAL [CausalAttn]: Expects 3D [nH, T, hd]."); return; }

    int nH = q->shape[0], T = q->shape[1], hd = q->shape[2];
    if (k->shape[0] != nH || k->shape[1] != T || k->shape[2] != hd ||
        v->shape[0] != nH || v->shape[1] != T || v->shape[2] != hd ||
        out->shape[0] != nH || out->shape[1] != T || out->shape[2] != hd)
        { tensor_set_error("FATAL [CausalAttn]: Shape mismatch."); return; }
    if (attn && (attn->shape[0] != nH || attn->shape[1] != T || attn->shape[2] != T))
        { tensor_set_error("FATAL [CausalAttn]: attn must be [nH, T, T]."); return; }

    float scale = 1.0f / sqrtf((float)hd);
    const float NEG_INF = -1e9f;

    const float* Qp  = F32(q);
    const float* Kp  = F32(k);
    const float* Vp  = F32(v);
    float*       Op  = F32(out);
    float*       Ap  = attn ? F32(attn) : NULL;

    /* Allocate per-thread scratch for S [T×T] */
    size_t TT = (size_t)T * T;

#pragma omp parallel
    {
        float* S = (float*)malloc(TT * sizeof(float));
        if (!S) { tensor_set_error("FATAL [CausalAttn]: OOM for S."); }

        if (S) {
#pragma omp for schedule(static)
            for (int h = 0; h < nH; h++) {
                const float* Qh = Qp + (size_t)h * T * hd;
                const float* Kh = Kp + (size_t)h * T * hd;
                const float* Vh = Vp + (size_t)h * T * hd;
                float*       Oh = Op + (size_t)h * T * hd;

                /* S = Q @ K.T * scale */
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            T, T, hd, scale,
                            Qh, hd, Kh, hd,
                            0.0f, S, T);

                /* Causal mask + softmax per row */
                for (int i = 0; i < T; i++) {
                    float* row = S + (size_t)i * T;
                    /* mask future */
                    for (int j = i + 1; j < T; j++) row[j] = NEG_INF;
                    /* stable softmax */
                    float m = row[0];
                    for (int j = 1; j <= i; j++) if (row[j] > m) m = row[j];
                    float s = 0.0f;
                    for (int j = 0; j <= i; j++) { row[j] = fast_expf(row[j] - m); s += row[j]; }
                    float inv_s = 1.0f / s;
                    for (int j = 0; j <= i; j++) row[j] *= inv_s;
                    /* Zero masked future positions so BLAS matmul is correct */
                    for (int j = i + 1; j < T; j++) row[j] = 0.0f;
                }

                /* Save attention weights if requested */
                if (Ap) {
                    float* Ah = Ap + (size_t)h * TT;
                    memcpy(Ah, S, TT * sizeof(float));
                }

                /* out = S @ V */
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            T, hd, T, 1.0f,
                            S, T, Vh, hd,
                            0.0f, Oh, hd);
            }
            free(S);
        }
    }
}

/* ---------------------------------------------------------------------------
 * Causal attention backward
 * dOut: [nH, T, hd]  upstream gradient
 * attn: [nH, T, T]   softmax weights saved from forward
 * Q, K, V: [nH, T, hd]  from forward
 * dQ, dK, dV: [nH, T, hd]  pre-allocated gradient outputs (overwritten, not +=)
 * ------------------------------------------------------------------------- */
void tensor_causal_attention_backward(Tensor* dOut, Tensor* attn,
                                       Tensor* Q, Tensor* K, Tensor* V,
                                       Tensor* dQ, Tensor* dK, Tensor* dV) {
    if (!dOut || !attn || !Q || !K || !V || !dQ || !dK || !dV)
        { tensor_set_error("FATAL [CausalAttnBwd]: NULL pointer."); return; }

    int nH = Q->shape[0], T = Q->shape[1], hd = Q->shape[2];
    float scale = 1.0f / sqrtf((float)hd);
    size_t TT = (size_t)T * T;

    const float* dOp = F32(dOut);
    const float* Ap  = F32(attn);
    const float* Qp  = F32(Q);
    const float* Kp  = F32(K);
    const float* Vp  = F32(V);
    float*       dQp = F32(dQ);
    float*       dKp = F32(dK);
    float*       dVp = F32(dV);

#pragma omp parallel
    {
        float* dA = (float*)malloc(TT * sizeof(float));
        float* dS = (float*)malloc(TT * sizeof(float));

        if (dA && dS) {
#pragma omp for schedule(static)
            for (int h = 0; h < nH; h++) {
                const float* dOh = dOp + (size_t)h * T * hd;
                const float* Ah  = Ap  + (size_t)h * TT;
                const float* Qh  = Qp  + (size_t)h * T * hd;
                const float* Kh  = Kp  + (size_t)h * T * hd;
                const float* Vh  = Vp  + (size_t)h * T * hd;
                float*       dQh = dQp + (size_t)h * T * hd;
                float*       dKh = dKp + (size_t)h * T * hd;
                float*       dVh = dVp + (size_t)h * T * hd;

                /* dV = A.T @ dOut : [T,T].T @ [T,hd] → [T,hd] */
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            T, hd, T, 1.0f,
                            Ah, T, dOh, hd,
                            0.0f, dVh, hd);

                /* dA = dOut @ V.T : [T,hd] @ [T,hd].T → [T,T] */
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            T, T, hd, 1.0f,
                            dOh, hd, Vh, hd,
                            0.0f, dA, T);

                /* dS = softmax_backward(dA, A):  dS[i,j] = A[i,j]*(dA[i,j] - rowdot[i]) */
                memset(dS, 0, TT * sizeof(float));
                for (int i = 0; i < T; i++) {
                    const float* Ai  = Ah + (size_t)i * T;
                    const float* dAi = dA + (size_t)i * T;
                    float*       dSi = dS + (size_t)i * T;
                    float rdot = 0.0f;
                    for (int j = 0; j <= i; j++) rdot += Ai[j] * dAi[j];
                    for (int j = 0; j <= i; j++) dSi[j] = Ai[j] * (dAi[j] - rdot);
                }

                /* dQ = dS @ K * scale : [T,T] @ [T,hd] → [T,hd] */
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            T, hd, T, scale,
                            dS, T, Kh, hd,
                            0.0f, dQh, hd);

                /* dK = dS.T @ Q * scale : [T,T].T @ [T,hd] → [T,hd] */
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            T, hd, T, scale,
                            dS, T, Qh, hd,
                            0.0f, dKh, hd);
            }
        }

        free(dA);
        free(dS);
    }
}
