<?php 
declare(strict_types=1);

namespace Pml\Lib;

/**
 * FFI Engine strictly linking OpenBLAS, LAPACKE, and AVX2 C logic.
 */
final class TensorEngine {
    private static ?\FFI $ffi = null;

    public static function get(): \FFI {
        if (self::$ffi === null) {
            $libPath = __DIR__ . '/libtensor.so';
            if (!file_exists($libPath)) {
                echo "[Compiler] Building C-Core with OpenBLAS, LAPACKE, and AVX2...\n";
                // Glob relative to __DIR__ — process CWD is unreliable under php-fpm.
                $cFiles = implode(' ', array_map('escapeshellarg', glob(__DIR__ . '/*.c')));
                $result = shell_exec(
                    "gcc -O3 -march=native -mfma -ffast-math -fopenmp -funroll-loops"
                    . " -fomit-frame-pointer -shared -fPIC -o "
                    . escapeshellarg($libPath) . " " . $cFiles
                    . " -lopenblas -llapacke -lm 2>&1"
                );
                if (!file_exists($libPath)) {
                    throw new \RuntimeException("[Compiler] Build failed:\n" . (string)$result);
                }
            }

            self::$ffi = \FFI::cdef("
                typedef struct { 
                    int ndim; 
                    int shape[8];
                    size_t stride[8];
                    size_t total_size; 
                    size_t byte_size; 
                    bool owns_data;
                    bool is_arena; // Memory Pooling: tracks if allocated in an Arena
                    int dtype; // 0 = FLOAT32, 1 = INT32, 2 = INT64
                    void* data;
                } TensorC;

                typedef struct TensorArena TensorArena;

                // --- GLOBAL ERROR HANDLING ---
                bool tensor_check_error(void);
                const char* tensor_get_last_error(void);
                void tensor_clear_error(void);
                void tensor_set_error(const char* msg);

                void free(void *ptr);

                void* safe_malloc(size_t size);
                void* safe_memalign(size_t alignment, size_t size);
                void safe_free(void** ptr);

                TensorArena* arena_create(size_t capacity);
                void* arena_alloc(TensorArena* arena, size_t size);
                void arena_reset(TensorArena* arena);
                void arena_destroy(TensorArena* arena);

                TensorC* tensor_create(int ndim, int* shape);
                TensorC* tensor_create_dtype(int ndim, int* shape, int dtype);
                TensorC* tensor_create_uninitialized(int ndim, int* shape, int dtype);
                TensorC* tensor_create_arena(int ndim, int* shape, int dtype, TensorArena* arena);
                TensorC* tensor_from_external(void* data, int ndim, int* shape, int dtype);
                void tensor_free(TensorC* t);
                TensorC* tensor_copy(TensorC* A);
                void _tensor_shape_assert(TensorC* A, TensorC* B, const char* op);
                bool tensor_is_contiguous(const TensorC* t);
                bool tensor_broadcast_shapes(const TensorC* A, const TensorC* B, int* out_ndim, int* out_shape, size_t* out_stride_A, size_t* out_stride_B);

                TensorC* tensor_view(TensorC* t);
                TensorC* tensor_slice(TensorC* t, int axis, int start, int length);
                TensorC* tensor_slice_step(TensorC* t, int axis, int start, int end, int step);
                TensorC* tensor_row_view(TensorC* t, int row);
                TensorC* tensor_column_view(TensorC* t, int col);

                void tensor_fill(TensorC* t, float val);
                TensorC* tensor_zeros(int ndim, int* shape);
                TensorC* tensor_ones(int ndim, int* shape);
                TensorC* tensor_range(float start, float end, float step);
                TensorC* tensor_linspace(float start, float end, int steps);
                void tensor_random_normal(TensorC* t, float mean, float stddev);
                void tensor_random_uniform(TensorC* t, float min_val, float max_val);
                TensorC* tensor_random_choice(TensorC* A, int n, bool replace);
                TensorC* tensor_random_permutation(TensorC* A);

                TensorC* tensor_reshape(TensorC* t, int ndim, int* new_shape);
                TensorC* tensor_flatten(TensorC* t);
                TensorC* tensor_expand_dims(TensorC* t, int axis);
                TensorC* tensor_squeeze(TensorC* t);
                TensorC* tensor_transpose_2d(TensorC* t);
                TensorC* tensor_transpose_nd(TensorC* t, int* axes);
                TensorC* tensor_swapaxes(TensorC* A, int axis1, int axis2);

                TensorC* tensor_add(TensorC* A, TensorC* B);
                TensorC* tensor_sub(TensorC* A, TensorC* B);
                TensorC* tensor_mul(TensorC* A, TensorC* B);
                TensorC* tensor_div(TensorC* A, TensorC* B);
                TensorC* tensor_add_scalar(TensorC* A, float val);
                TensorC* tensor_mul_scalar(TensorC* A, float val);
                TensorC* tensor_pow(TensorC* A, TensorC* B);
                TensorC* tensor_clip(TensorC* A, float min_val, float max_val);

                void tensor_add_inplace(TensorC* A, TensorC* B);
                void tensor_sub_inplace(TensorC* A, TensorC* B);
                void tensor_mul_inplace(TensorC* A, TensorC* B);
                void tensor_div_inplace(TensorC* A, TensorC* B);
                void tensor_add_scalar_inplace(TensorC* A, float val);
                void tensor_mul_scalar_inplace(TensorC* A, float val);

                TensorC* tensor_sqrt(TensorC* A);
                TensorC* tensor_square(TensorC* A);
                TensorC* tensor_abs(TensorC* A);
                TensorC* tensor_sign(TensorC* A);
                TensorC* tensor_exp(TensorC* A);
                TensorC* tensor_log(TensorC* A);
                TensorC* tensor_log1p(TensorC* A);
                TensorC* tensor_round(TensorC* A);
                TensorC* tensor_floor(TensorC* A);
                TensorC* tensor_ceil(TensorC* A);
                TensorC* tensor_sigmoid(TensorC* A);
                TensorC* tensor_tanh(TensorC* A);
                TensorC* tensor_relu(TensorC* A);
                TensorC* tensor_sin(TensorC* A);
                TensorC* tensor_cos(TensorC* A);
                TensorC* tensor_tan(TensorC* A);
                TensorC* tensor_asin(TensorC* A);
                TensorC* tensor_acos(TensorC* A);
                TensorC* tensor_atan(TensorC* A);

                TensorC* tensor_equal(TensorC* A, TensorC* B);
                TensorC* tensor_not_equal(TensorC* A, TensorC* B);
                TensorC* tensor_greater(TensorC* A, TensorC* B);
                TensorC* tensor_greater_equal(TensorC* A, TensorC* B);
                TensorC* tensor_less(TensorC* A, TensorC* B);
                TensorC* tensor_less_equal(TensorC* A, TensorC* B);
                TensorC* tensor_logical_not(TensorC* A);

                TensorC* tensor_less_scalar_f32(TensorC* A, float val);
                TensorC* tensor_greater_scalar_f32(TensorC* A, float val);

                TensorC* tensor_isnan(TensorC* A);
                TensorC* tensor_isinf(TensorC* A);
                void tensor_nan_to_num_inplace(TensorC* A, float nan_val, float posinf_val, float neginf_val);
                bool tensor_any(TensorC* A);
                bool tensor_all(TensorC* A);

                TensorC* tensor_where(TensorC* condition, TensorC* x, TensorC* y);
                TensorC* tensor_boolean_index(TensorC* A, TensorC* mask);
                TensorC* tensor_take(TensorC* A, TensorC* indices, int axis);
                TensorC* tensor_unique(TensorC* A);
                TensorC* tensor_bincount(TensorC* A);

                TensorC* tensor_concat(TensorC** tensors, int num_tensors, int axis);
                TensorC* tensor_pad(TensorC* A, int* pad_width, float constant_value);

                TensorC* tensor_argsort(TensorC* A, int axis);
                TensorC* tensor_sort(TensorC* A, int axis);
                TensorC* tensor_topk(TensorC* A, int k, int axis);

                float tensor_sum(TensorC* A);
                float tensor_product(TensorC* A);
                float tensor_mean(TensorC* A);
                float tensor_min(TensorC* A);
                float tensor_max(TensorC* A);
                int tensor_argmin(TensorC* A);
                int tensor_argmax(TensorC* A);
                float tensor_variance(TensorC* A);
                float tensor_std(TensorC* A);
                float tensor_median(TensorC* A);

                TensorC* tensor_sum_axis(TensorC* A, int axis);
                TensorC* tensor_mean_axis(TensorC* A, int axis);
                TensorC* tensor_max_axis(TensorC* A, int axis);
                TensorC* tensor_min_axis(TensorC* A, int axis);
                TensorC* tensor_cumsum_axis(TensorC* A, int axis);

                TensorC* tensor_sum_multi(TensorC* A, int* axes, int num_axes);
                TensorC* tensor_mean_multi(TensorC* A, int* axes, int num_axes);
                TensorC* tensor_max_multi(TensorC* A, int* axes, int num_axes);

                void tensor_standardize_inplace(TensorC* A);
                void tensor_normalize_inplace(TensorC* A);
                TensorC* tensor_normalize(TensorC* A);
                TensorC* tensor_standardize(TensorC* A);

                float tensor_dot(TensorC* A, TensorC* B);
                float tensor_trace(TensorC* A);
                TensorC* tensor_matmul(TensorC* A, TensorC* B);
                TensorC* tensor_bmm(TensorC* A, TensorC* B);
                TensorC* tensor_matmul_ex(TensorC* A, TensorC* B, bool transA, bool transB);
                void     tensor_matmul_into(TensorC* out, TensorC* A, TensorC* B, bool transA, bool transB);
                void     tensor_sum_axis_into(TensorC* out, TensorC* A, int axis);

                TensorC* tensor_inverse(TensorC* A);
                TensorC* tensor_pinv(TensorC* A);
                TensorC* tensor_solve(TensorC* A, TensorC* B);

                TensorC* tensor_cholesky(TensorC* A);
                void tensor_lu(TensorC* A, TensorC** P_out, TensorC** L_out, TensorC** U_out);
                void tensor_svd(TensorC* A, TensorC** U_out, TensorC** S_out, TensorC** Vt_out);
                void tensor_eigen_sym(TensorC* A, TensorC** EigenVals_out, TensorC** EigenVecs_out);
                TensorC* tensor_ref(TensorC* A);
                TensorC* tensor_rref(TensorC* A);

                TensorC* tensor_im2col(TensorC* A, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
                TensorC* tensor_col2im(TensorC* cols_tensor, int batch, int channels, int height, int width, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
                TensorC* tensor_conv2d(TensorC* X, TensorC* W, TensorC* bias, int stride_h, int stride_w, int pad_h, int pad_w);
                TensorC** tensor_conv2d_backward(TensorC* dY, TensorC* X, TensorC* W, int stride_h, int stride_w, int pad_h, int pad_w);

                TensorC* tensor_embedding_lookup(TensorC* tokens, TensorC* weights);
                TensorC** tensor_dataset_from_csv(const char* filepath, int label_col, int has_header);

                void tensor_fused_bce_loss_and_grad(TensorC* preds, TensorC* targets, TensorC* grads, float* out_loss);
                void tensor_fused_adam_step(TensorC* param, TensorC* grad, TensorC* m, TensorC* v, float lr, float b1, float b2, float eps, int t);

                // --- FUSED NEURAL NETWORK KERNELS ---
                // out = X @ W^T + bias  (bias may be NULL)
                TensorC* tensor_linear(TensorC* X, TensorC* W, TensorC* bias);
                // out = relu(A + B)
                TensorC* tensor_add_relu(TensorC* A, TensorC* B);
                // out = A * B + C
                TensorC* tensor_mul_add(TensorC* A, TensorC* B, TensorC* C);

                // --- THREADING CONTROL ---
                void tensor_configure_threading(int omp_threads, int blas_threads);

                // --- TRANSFORMER INFERENCE PRIMITIVES ---
                void tensor_rmsnorm(TensorC* x, float eps);
                void tensor_apply_rope(TensorC* q, TensorC* k, int head_dim, int pos, float base_freq, float scale);
                void tensor_softmax_inplace(TensorC* x);
                void tensor_attention(TensorC* out, TensorC* q, TensorC* k, TensorC* v);

                // --- KV CACHE ---
                typedef struct {
                    float*  data;
                    int     len;
                    int     cap;
                    int     head_dim;
                } KVCache;

                KVCache* kvcache_create(int cap, int head_dim);
                void     kvcache_free(KVCache* cache);
                void     kvcache_reset(KVCache* cache);
                int      kvcache_len(KVCache* cache);
                void     kvcache_append(KVCache* cache, TensorC* k, TensorC* v);
                void     tensor_attention_kv(TensorC* out, TensorC* q, KVCache* cache);

                typedef struct {
                    int feature_idx;
                    float threshold;
                    int left_idx;
                    int right_idx;
                    float value;
                } HardwareNode;
                void tensor_hardware_tree_predict(TensorC* X, HardwareNode* nodes, TensorC* out);

                void tensor_save_to_file(TensorC* t, const char* filepath);
                TensorC* tensor_load_from_file(const char* filepath);
                int tensor_save_safetensors(const char* filepath, const char* json_header, uint64_t json_len, TensorC** tensors, int num_tensors);
                void tensor_copy_from(TensorC* dest, TensorC* src);

                // --- ADVANCED INFERENCE & TRAINING PRIMITIVES ---
                TensorC* tensor_from_mmap(const char* filepath, size_t byte_offset, int ndim, const int* shape, int dtype);
                void     tensor_mmap_free(TensorC* t);
                TensorC* tensor_silu(TensorC* A);
                TensorC* tensor_swiglu(TensorC* gate, TensorC* up);
                void     tensor_fused_cross_entropy_loss_and_grad(TensorC* logits, TensorC* target_ids, TensorC* grads, float* out_loss);
                TensorC* tensor_rmsnorm_backward(TensorC* dY, TensorC* X, TensorC* weights, float eps);
                void     tensor_embedding_backward(TensorC* dY, TensorC* token_ids, TensorC* dWeights);

                // ---------------------------------------------------------------
                // Columnar DataFrame + ETL  (src/Lib/dataset_io.c)
                // Opaque struct — PHP only ever holds a DataFrame* pointer;
                // all access goes through df_* functions.
                // ---------------------------------------------------------------
                typedef struct DataFrame DataFrame;

                DataFrame*  df_read_csv(const char* filepath, bool has_header);
                void        df_free(DataFrame* df);

                DataFrame*  df_select_columns(const DataFrame* df,
                                              const int* col_indices, int n);
                DataFrame*  df_drop_nans(const DataFrame* df);
                DataFrame*  df_one_hot_encode(const DataFrame* df, int col_idx);

                TensorC*    df_to_tensor(const DataFrame* df,
                                         const int* col_indices, int n);

                size_t      df_num_rows(const DataFrame* df);
                size_t      df_num_cols(const DataFrame* df);
                const char* df_col_name(const DataFrame* df, int col_idx);
                int         df_col_dtype(const DataFrame* df, int col_idx);
                int         df_col_n_categories(const DataFrame* df, int col_idx);
                const char* df_col_category_name(const DataFrame* df,
                                                  int col_idx, int cat_idx);
            ", $libPath);
        }
        return self::$ffi;
    }
}