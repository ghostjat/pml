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
void tensor_set_error(const char *msg);   /* used by dataframe.c and other TUs */

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
bool _tensor_shape_assert(Tensor* A, Tensor* B, const char* op);
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

/* Extended GEMM: out = op(A) @ op(B);  op(X)=X^T when trans=true */
Tensor* tensor_matmul_ex(Tensor* A, Tensor* B, bool transA, bool transB);
/* Zero-alloc GEMM into a pre-allocated contiguous output tensor */
void    tensor_matmul_into(Tensor* out, Tensor* A, Tensor* B, bool transA, bool transB);
/* Zero-alloc axis-sum into a pre-allocated output tensor */
void    tensor_sum_axis_into(Tensor* out, Tensor* A, int axis);

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
// 16. FUSED NEURAL NETWORK KERNELS
// ============================================================================

// Fused fully-connected: out = X @ W^T + bias  (bias optional, pass NULL to skip)
// X: [m, k]  W: [n, k]  bias: [n]  → out: [m, n]
Tensor* tensor_linear(Tensor* X, Tensor* W, Tensor* bias);

// Fused add + ReLU: out = relu(A + B)
Tensor* tensor_add_relu(Tensor* A, Tensor* B);

// Fused multiply-add (FMA): out = A * B + C
Tensor* tensor_mul_add(Tensor* A, Tensor* B, Tensor* C);

// ============================================================================
// 17. THREADING CONTROL
// ============================================================================

// Set OpenMP and BLAS thread counts independently to prevent oversubscription.
// Call once at startup: omp_threads = cores, blas_threads = 1 when outer
// OpenMP loops are used; blas_threads = cores for pure-BLAS workloads.
void tensor_configure_threading(int omp_threads, int blas_threads);

// ============================================================================
// 18. TRANSFORMER INFERENCE PRIMITIVES
// ============================================================================

/* In-place row-wise Root Mean Square Normalization.
 * x: contiguous FLOAT32, any shape; last axis = feature dim.
 * eps: small constant for numerical stability (e.g. 1e-5). */
void tensor_rmsnorm(Tensor* x, float eps);

/* In-place Rotary Position Embedding applied to q and k.
 * Both must be contiguous FLOAT32; head_dim must be even.
 * pos:       absolute position index in the sequence.
 * base_freq: rotary embedding base (e.g. 10000.0 for LLaMA-2, 500000.0 for LLaMA-3).
 * scale:     position scaling factor (1.0 = standard; <1.0 for extended context). */
void tensor_apply_rope(Tensor* q, Tensor* k, int head_dim, int pos,
                       float base_freq, float scale);

/* In-place numerically-stable softmax along the last axis.
 * x must be contiguous FLOAT32. */
void tensor_softmax_inplace(Tensor* x);

/* Scaled dot-product attention (inference, no mask, no batch).
 * q, k, v, out: contiguous FLOAT32 [seq_len, head_dim].
 * out must be pre-allocated by the caller. */
void tensor_attention(Tensor* out, Tensor* q, Tensor* k, Tensor* v);

// ============================================================================
// 19. KV CACHE
// ============================================================================

/* Interleaved K/V cache.  data[i] = [k_0..k_{hd-1}, v_0..v_{hd-1}] for token i.
 * 32-byte aligned; append-only; zero-copy reads via tensor_attention_kv. */
typedef struct {
    float* data;      /* [cap][2*head_dim], 32-byte aligned               */
    int    len;       /* tokens currently stored                          */
    int    cap;       /* maximum tokens (allocated capacity)              */
    int    head_dim;  /* dimension of each K and V vector                 */
} KVCache;

KVCache* kvcache_create(int cap, int head_dim);
void     kvcache_free(KVCache* cache);
void     kvcache_reset(KVCache* cache);
int      kvcache_len(const KVCache* cache);
void     kvcache_append(KVCache* cache, const Tensor* k, const Tensor* v);

/* Streaming attention against a KV cache (Milakov online softmax).
 * q:   [seq_q, head_dim]; out: [seq_q, head_dim] (pre-allocated).
 * No O(seq²) scores buffer; O(head_dim) working memory per thread. */
void tensor_attention_kv(Tensor* out, Tensor* q, const KVCache* cache);

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

void tensor_copy_from(Tensor* dest, Tensor* src);

// ============================================================================
// 20. ADVANCED INFERENCE & TRAINING PRIMITIVES
// ============================================================================

/* Zero-copy tensor backed by a memory-mapped file region (MAP_PRIVATE).
 * byte_offset: byte position within the file where the tensor data starts.
 * owns_data is false; call tensor_mmap_free() instead of tensor_free().    */
Tensor* tensor_from_mmap(const char* filepath, size_t byte_offset,
                         int ndim, const int* shape, int dtype);

/* Unmap and free a tensor created by tensor_from_mmap(). */
void tensor_mmap_free(Tensor* t);

/* SiLU (Swish) activation: out[i] = x[i] * sigmoid(x[i]).
 * AVX2 vectorized. Returns a new FLOAT32 tensor.                           */
Tensor* tensor_silu(Tensor* A);

/* Fused SwiGLU: out[i] = silu(gate[i]) * up[i].
 * gate and up must have identical shape. Returns new FLOAT32 tensor.       */
Tensor* tensor_swiglu(Tensor* gate, Tensor* up);

/* Fused cross-entropy: numerically stable softmax + NLL loss + gradient.
 * logits:     [batch, vocab] FLOAT32   input logits
 * target_ids: [batch]        INT32     correct token ids
 * grads:      [batch, vocab] FLOAT32   pre-allocated; receives (probs - one_hot)
 * out_loss:   receives mean loss over the batch                            */
void tensor_fused_cross_entropy_loss_and_grad(Tensor* logits, Tensor* target_ids,
                                               Tensor* grads,  float* out_loss);

/* RMSNorm backward: dx_j = (1/r)*w_j*dy_j - x_j*S/(d*r³).
 * dY:      [batch, d] upstream gradient
 * X:       [batch, d] original forward input
 * weights: [d]        RMSNorm weight vector
 * Returns: [batch, d] gradient w.r.t. X                                   */
Tensor* tensor_rmsnorm_backward(Tensor* dY, Tensor* X, Tensor* weights, float eps);

/* Embedding backward: accumulates dY rows into dWeights[token_id] rows.
 * dY:        [seq_len, embed_dim] upstream gradients
 * token_ids: [seq_len]            INT32 token indices
 * dWeights:  [vocab, embed_dim]   gradient accumulator (caller must zero first) */
void tensor_embedding_backward(Tensor* dY, Tensor* token_ids, Tensor* dWeights);

// ============================================================================
// 21. MAMBA / SELECTIVE SSM ENGINE
//
// Shapes (B=batch, T=seq_len, D=d_model, N=d_state):
//   x        [B,T,D]    input sequence                        FLOAT32
//   A_log    [D,N]      log(-eigenvalues), must be ≤ 0        FLOAT32
//   B_proj   [B,T,N]    input-dependent B projection          FLOAT32
//   C_proj   [B,T,N]    input-dependent C projection          FLOAT32
//   D_skip   [D]        skip-connection weight (NULL=off)     FLOAT32
//   delta    [B,T,D]    discretization step (post-softplus)   FLOAT32
//   state    [B,D,N]    recurrent state: initial in, final out FLOAT32
//   out      [B,T,D]    pre-allocated output                  FLOAT32
//   cache    [B,D,T,N]  per-step state snapshots (NULL=infer) FLOAT32
//
// ZOH recurrence:
//   Ā[t,d,n] = exp(delta[b,t,d] * A_log[d,n])          (0 < Ā ≤ 1 when A_log ≤ 0)
//   h[t,d,n] = Ā*h[t-1,d,n] + delta[b,t,d]*B[b,t,n]*x[b,t,d]
//   y[b,t,d] = Σ_n C[b,t,n]*h[t,d,n]  +  D_skip[d]*x[b,t,d]
//
// Parallelism: OpenMP collapse(B,D); T is sequential (serial recurrence).
//              Automatic tile-scan kicks in when B*D < nthreads*4 && T≥128.
// SIMD:        AVX512 > AVX2 over the N-dimension; scalar fallback.
// ============================================================================

/* Fused forward — training fills cache, inference streams state in-place. */
void tensor_mamba_forward(Tensor* x,      Tensor* A_log,
                          Tensor* B_proj, Tensor* C_proj,
                          Tensor* D_skip, Tensor* delta,
                          Tensor* state,  Tensor* out,
                          Tensor* cache,  int training);

/* Fused backward — all gradient outputs are pre-allocated and zeroed by
 * the caller (dA and dD use += accumulation across steps).
 * h0: [B,D,N] initial state saved with tensor_copy() before forward(). */
void tensor_mamba_backward(Tensor* dout,   Tensor* x,
                           Tensor* A_log,  Tensor* B_proj, Tensor* C_proj,
                           Tensor* D_skip, Tensor* delta,
                           Tensor* h0,     Tensor* cache,
                           Tensor* dx,     Tensor* dA,
                           Tensor* dB,     Tensor* dC,
                           Tensor* dD,     Tensor* ddelta);

/* Convenience zero-allocators for state and training cache tensors. */
Tensor* tensor_mamba_alloc_state(int batch, int d_model, int d_state);
Tensor* tensor_mamba_alloc_cache(int batch, int seq_len, int d_model, int d_state);

/* ── Section 22: Classical ML Extensions ────────────────────────────────────── */

/* 22.1  Argmax along axis — returns FLOAT32 tensor of indices */
Tensor* tensor_argmax_axis(Tensor* A, int axis);

/* 22.2  Pairwise squared L2: A[m,D], B[n,D] → [m,n] */
Tensor* tensor_pairwise_sq_l2(Tensor* A, Tensor* B);

/* 22.3  In-place element-wise unary ops (return void; modify A in-place) */
void tensor_exp_inplace(Tensor* A);
void tensor_log_inplace(Tensor* A);
void tensor_sqrt_inplace(Tensor* A);
void tensor_sigmoid_inplace(Tensor* A);
void tensor_tanh_inplace(Tensor* A);
void tensor_relu_inplace(Tensor* A);

/* 22.4  Numerically stable row-wise softmax in-place */
void tensor_row_softmax_inplace(Tensor* A);

/* 22.5  GBDT engine */
/* Compute [D, Q-1] quantile bin boundaries from [N,D] X */
Tensor* tensor_gbdt_compute_boundaries(Tensor* X, int Q);
/* Bin [N,D] X into INT32 [N,D] bin indices using [D, Q-1] boundaries */
Tensor* tensor_gbdt_bin_samples(Tensor* X, Tensor* boundaries, int Q);
/* MSE gradients/hessians: preds [N], y [N]; writes into pre-allocated out_g/out_h */
void    tensor_gbdt_mse_grad_hess(Tensor* preds, Tensor* y, Tensor* out_g, Tensor* out_h);
/* Log-loss gradients/hessians for binary classification */
void    tensor_gbdt_logloss_grad_hess(Tensor* preds, Tensor* y, Tensor* out_g, Tensor* out_h);
/* Build [D,Q] gradient histograms into pre-allocated hist_g, hist_h (caller zeros them) */
void    tensor_gbdt_histogram(Tensor* bins, Tensor* g, Tensor* h, Tensor* mask,
                              int Q, Tensor* hist_g, Tensor* hist_h);
/* Find best split scanning histograms; outputs via pointer args */
void    tensor_gbdt_best_split(Tensor* hist_g, Tensor* hist_h, int Q,
                               float sum_g, float sum_h, int node_n,
                               float lambda, float gamma,
                               int* out_feat, int* out_bin, float* out_gain);
/* Split node mask into left/right masks (caller pre-allocates out_left, out_right) */
void    tensor_gbdt_split_node(Tensor* bins, Tensor* mask, int feat, int bin,
                               Tensor* out_left, Tensor* out_right);
/* Update preds in-place with leaf delta; returns leaf value */
float   tensor_gbdt_leaf_update(Tensor* preds, Tensor* mask,
                                float sum_g, float sum_h, float lr, float lambda);
/* Multi-tree inference: bins [N,D] INT32 + packed tree arrays → [N] float predictions */
Tensor* tensor_gbdt_predict_all(Tensor* X_bins, Tensor* feats, Tensor* thresholds,
                                Tensor* lefts, Tensor* rights,
                                Tensor* tree_sizes, float base_score);

/* 22.6  Quantile transform */
/* Fit [D, n_quantiles] landmark matrix from [N,D] X */
Tensor* tensor_quantile_fit(Tensor* X, int n_quantiles);
/* Apply quantile transform [N,D] → [N,D] uniform [0,1]; n_quantiles must match landmarks shape[1] */
Tensor* tensor_quantile_transform(Tensor* X, Tensor* landmarks, int n_quantiles);

/* 22.7  Yeo-Johnson power transform */
/* Fit [D] optimal lambda vector from [N,D] X */
Tensor* tensor_yj_fit(Tensor* X);
/* Apply Yeo-Johnson column-wise [N,D] → [N,D] */
Tensor* tensor_yj_transform(Tensor* X, Tensor* lambdas);

#endif // TENSOR_H