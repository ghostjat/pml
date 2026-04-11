#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// ============================================================================
// 1. DATA TYPES & UNIFIED STRUCT
// ============================================================================
typedef enum {
    DTYPE_FLOAT32 = 0, // Standard dense network weights & math
    DTYPE_INT32 = 1,   // NLP Token IDs & Classification Labels
    DTYPE_INT64 = 2    // High-precision indexing
} TensorDType;

// Genuine N-Dimensional Tensor with Stride, Ownership, and Type Awareness
typedef struct {
    int ndim;
    int shape[8];      // Supports up to 8D tensors natively
    size_t stride[8];  // Strides for zero-copy views and broadcasting
    size_t total_size; 
    size_t byte_size;
    bool owns_data;    // Memory Safety: tracks if C should free 'data'
    bool is_arena;     // Memory Pooling: tracks if allocated in an Arena
    TensorDType dtype; // The engine now knows what is inside 'data'
    void* data;        // Replaces float*. Safely holds floats or integers.
} Tensor;


// ============================================================================
// 1b. GLOBAL ERROR HANDLING
// ============================================================================
bool tensor_check_error(void);
const char* tensor_get_last_error(void);
void tensor_clear_error(void);

// ============================================================================
// 2. SAFE MEMORY ALLOCATORS & LIFECYCLE
// ============================================================================
void* safe_malloc(size_t size);
void* safe_memalign(size_t alignment, size_t size);
void safe_free(void** ptr);

// --- NEW: ARENA ALLOCATOR ---
typedef struct {
    uint8_t* memory;
    size_t capacity;
    size_t offset;
} TensorArena;

TensorArena* arena_create(size_t capacity);
void* arena_alloc(TensorArena* arena, size_t size);
void arena_reset(TensorArena* arena);
void arena_destroy(TensorArena* arena);

Tensor* tensor_create(int ndim, int* shape); // Defaults to FLOAT32, zero-initialized
Tensor* tensor_create_dtype(int ndim, int* shape, TensorDType dtype);
Tensor* tensor_create_uninitialized(int ndim, int* shape, TensorDType dtype); // Skip zero-fill for outputs
Tensor* tensor_create_arena(int ndim, int* shape, TensorDType dtype, TensorArena* arena);
Tensor* tensor_from_external(void* data, int ndim, int* shape, TensorDType dtype);
void tensor_free(Tensor* t);
Tensor* tensor_copy(Tensor* A);
void tensor_shape_assert(Tensor* A, Tensor* B, const char* op);
bool tensor_is_contiguous(const Tensor* t);
bool tensor_broadcast_shapes(const Tensor* A, const Tensor* B, int* out_ndim, int* out_shape, size_t* out_stride_A, size_t* out_stride_B);

// ============================================================================
// 3. ZERO-COPY VIEWS & SLICING
// ============================================================================
Tensor* tensor_view(Tensor* t);
Tensor* tensor_slice(Tensor* t, int axis, int start, int length);
Tensor* tensor_slice_step(Tensor* t, int axis, int start, int end, int step);
Tensor* tensor_row_view(Tensor* t, int row);
Tensor* tensor_column_view(Tensor* t, int col);

// ============================================================================
// 4. INITIALIZERS & RANDOM SAMPLING
// ============================================================================
void tensor_fill(Tensor* t, float val);
Tensor* tensor_zeros(int ndim, int* shape);
Tensor* tensor_ones(int ndim, int* shape);
Tensor* tensor_range(float start, float end, float step);
Tensor* tensor_linspace(float start, float end, int steps);
void tensor_random_normal(Tensor* t, float mean, float stddev);
void tensor_random_uniform(Tensor* t, float min_val, float max_val);
Tensor* tensor_random_choice(Tensor* A, int n, bool replace);
Tensor* tensor_random_permutation(Tensor* A);

// ============================================================================
// 5. SHAPE MUTATIONS
// ============================================================================
Tensor* tensor_reshape(Tensor* t, int ndim, int* new_shape);
Tensor* tensor_flatten(Tensor* t);
Tensor* tensor_expand_dims(Tensor* t, int axis);
Tensor* tensor_squeeze(Tensor* t);
Tensor* tensor_transpose_2d(Tensor* t);
Tensor* tensor_transpose_nd(Tensor* t, int* axes);
Tensor* tensor_swapaxes(Tensor* A, int axis1, int axis2);

// ============================================================================
// 6. SIMD MATH & BINARY OPS (FLOAT32 Math Only)
// ============================================================================
Tensor* tensor_add(Tensor* A, Tensor* B);
Tensor* tensor_sub(Tensor* A, Tensor* B);
Tensor* tensor_mul(Tensor* A, Tensor* B);
Tensor* tensor_div(Tensor* A, Tensor* B);
Tensor* tensor_add_scalar(Tensor* A, float val);
Tensor* tensor_mul_scalar(Tensor* A, float val);
Tensor* tensor_pow(Tensor* A, Tensor* B);
Tensor* tensor_clip(Tensor* A, float min_val, float max_val);

void tensor_add_inplace(Tensor* A, Tensor* B);
void tensor_sub_inplace(Tensor* A, Tensor* B);
void tensor_mul_inplace(Tensor* A, Tensor* B);
void tensor_div_inplace(Tensor* A, Tensor* B);
void tensor_add_scalar_inplace(Tensor* A, float val);
void tensor_mul_scalar_inplace(Tensor* A, float val);

// ============================================================================
// 7. UNARY MATH & TRIGONOMETRY
// ============================================================================
Tensor* tensor_sqrt(Tensor* A);
Tensor* tensor_square(Tensor* A);
Tensor* tensor_abs(Tensor* A);
Tensor* tensor_sign(Tensor* A);
Tensor* tensor_exp(Tensor* A);
Tensor* tensor_log(Tensor* A);
Tensor* tensor_log1p(Tensor* A);
Tensor* tensor_round(Tensor* A);
Tensor* tensor_floor(Tensor* A);
Tensor* tensor_ceil(Tensor* A);
Tensor* tensor_sigmoid(Tensor* A);
Tensor* tensor_tanh(Tensor* A);
Tensor* tensor_relu(Tensor* A);
Tensor* tensor_sin(Tensor* A);
Tensor* tensor_cos(Tensor* A);
Tensor* tensor_tan(Tensor* A);
Tensor* tensor_asin(Tensor* A);
Tensor* tensor_acos(Tensor* A);
Tensor* tensor_atan(Tensor* A);

// ============================================================================
// 8. LOGICAL & DATA VALIDATION
// ============================================================================
Tensor* tensor_equal(Tensor* A, Tensor* B);
Tensor* tensor_not_equal(Tensor* A, Tensor* B);
Tensor* tensor_greater(Tensor* A, Tensor* B);
Tensor* tensor_greater_equal(Tensor* A, Tensor* B);
Tensor* tensor_less(Tensor* A, Tensor* B);
Tensor* tensor_less_equal(Tensor* A, Tensor* B);
Tensor* tensor_logical_not(Tensor* A);
Tensor* tensor_less_scalar_f32(Tensor* A, float val);
Tensor* tensor_greater_scalar_f32(Tensor* A, float val);

Tensor* tensor_isnan(Tensor* A);
Tensor* tensor_isinf(Tensor* A);
void tensor_nan_to_num_inplace(Tensor* A, float nan_val, float posinf_val, float neginf_val);
bool tensor_any(Tensor* A);
bool tensor_all(Tensor* A);

// ============================================================================
// 9. FANCY INDEXING, MASKING & SETS
// ============================================================================
Tensor* tensor_where(Tensor* condition, Tensor* x, Tensor* y);
Tensor* tensor_boolean_index(Tensor* A, Tensor* mask);
Tensor* tensor_take(Tensor* A, Tensor* indices, int axis);
Tensor* tensor_unique(Tensor* A);
Tensor* tensor_bincount(Tensor* A);

// ============================================================================
// 10. CONCATENATION & PADDING
// ============================================================================
Tensor* tensor_concat(Tensor** tensors, int num_tensors, int axis);
Tensor* tensor_pad(Tensor* A, int* pad_width, float constant_value);

// ============================================================================
// 11. SORTING & AGGREGATIONS
// ============================================================================
Tensor* tensor_argsort(Tensor* A, int axis);
Tensor* tensor_sort(Tensor* A, int axis);
Tensor* tensor_topk(Tensor* A, int k, int axis);

float tensor_sum(Tensor* A);
float tensor_product(Tensor* A);
float tensor_mean(Tensor* A);
float tensor_min(Tensor* A);
float tensor_max(Tensor* A);
int tensor_argmin(Tensor* A);
int tensor_argmax(Tensor* A);
float tensor_variance(Tensor* A);
float tensor_std(Tensor* A);
float tensor_median(Tensor* A);

Tensor* tensor_sum_axis(Tensor* A, int axis);
Tensor* tensor_mean_axis(Tensor* A, int axis);
Tensor* tensor_max_axis(Tensor* A, int axis);
Tensor* tensor_min_axis(Tensor* A, int axis);
Tensor* tensor_cumsum_axis(Tensor* A, int axis);

Tensor* tensor_reduce_multi_axis(Tensor* A, int* axes, int num_axes, Tensor* (*reduce_fn)(Tensor*, int));
Tensor* tensor_sum_multi(Tensor* A, int* axes, int num_axes);
Tensor* tensor_mean_multi(Tensor* A, int* axes, int num_axes);
Tensor* tensor_max_multi(Tensor* A, int* axes, int num_axes);

void tensor_standardize_inplace(Tensor* A);
void tensor_normalize_inplace(Tensor* A);
Tensor* tensor_normalize(Tensor* A);
Tensor* tensor_standardize(Tensor* A);

// ============================================================================
// 12. LINEAR ALGEBRA & DECOMPOSITIONS
// ============================================================================
float tensor_dot(Tensor* A, Tensor* B);
float tensor_trace(Tensor* A);
Tensor* tensor_matmul(Tensor* A, Tensor* B);
Tensor* tensor_bmm(Tensor* A, Tensor* B);

Tensor* tensor_inverse(Tensor* A);
Tensor* tensor_pinv(Tensor* A);
Tensor* tensor_solve(Tensor* A, Tensor* B);

Tensor* tensor_cholesky(Tensor* A);
void tensor_lu(Tensor* A, Tensor** P_out, Tensor** L_out, Tensor** U_out);
void tensor_svd(Tensor* A, Tensor** U_out, Tensor** S_out, Tensor** Vt_out);
void tensor_eigen_sym(Tensor* A, Tensor** EigenVals_out, Tensor** EigenVecs_out);
Tensor* tensor_ref(Tensor* A);
Tensor* tensor_rref(Tensor* A);

// ============================================================================
// 13. DEEP LEARNING (CNNs & LLMs)
// ============================================================================
Tensor* tensor_im2col(Tensor* A, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
Tensor* tensor_col2im(Tensor* cols_tensor, int batch, int channels, int height, int width, 
                      int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
Tensor* tensor_conv2d(Tensor* X, Tensor* W, Tensor* bias, int stride_h, int stride_w, int pad_h, int pad_w);
Tensor** tensor_conv2d_backward(Tensor* dY, Tensor* X, Tensor* W, int stride_h, int stride_w, int pad_h, int pad_w);

// Bridges Integer NLP Tokens to Float32 Dense Networks (Zero Casting Overhead)
Tensor* tensor_embedding_lookup(Tensor* tokens, Tensor* embedding_weights);

// --- NEW: FUSED KERNELS & HARDWARE INFERENCE ---
void tensor_fused_bce_loss_and_grad(Tensor* preds, Tensor* targets, Tensor* grads, float* out_loss);
void tensor_fused_adam_step(Tensor* param, Tensor* grad, Tensor* m, Tensor* v, float lr, float b1, float b2, float eps, int t);

typedef struct __attribute__((packed)) {
    int feature_idx;
    float threshold;
    int left_idx;
    int right_idx;
    float value;
} HardwareNode;

void tensor_hardware_tree_predict(Tensor* X, HardwareNode* nodes, Tensor* out);

// ============================================================================
// 14. I/O SERIALIZATION
// ============================================================================
void tensor_save_to_file(Tensor* t, const char* filepath);
Tensor* tensor_load_from_file(const char* filepath);
int tensor_save_safetensors(const char* filepath, const char* json_header, uint64_t json_len, Tensor** tensors, int num_tensors);

// ============================================================================
// 15. DATASET I/O (RubixML-Style Direct Ingestion)
// ============================================================================
// Parses a CSV directly into Contiguous Memory Tensors.
// Returns an array of 2 Tensor Pointers: [Samples, Labels]. 
// If label_col is -1, Labels will be NULL.
Tensor** tensor_dataset_from_csv(const char* filepath, int label_col, int has_header);

#endif // TENSOR_H