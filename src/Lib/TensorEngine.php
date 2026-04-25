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
                    "gcc -O3 -march=native -mtune=native -mfma -ffast-math -fno-math-errno -funsafe-math-optimizations -fopenmp -funroll-loops -flto -fomit-frame-pointer -D_GNU_SOURCE -shared -fPIC -o "
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
                // MAMBA / SELECTIVE SSM ENGINE
                // All tensors are TensorC*; shapes described in tensor.h §21.
                // ---------------------------------------------------------------

                /* Forward pass — training fills cache, inference streams state. */
                void tensor_mamba_forward(
                    TensorC* x,      TensorC* A_log,
                    TensorC* B_proj, TensorC* C_proj,
                    TensorC* D_skip, TensorC* delta,
                    TensorC* state,  TensorC* out,
                    TensorC* cache,  int training
                );

                /* Backward pass — all gradient tensors pre-allocated & zeroed by caller. */
                void tensor_mamba_backward(
                    TensorC* dout,   TensorC* x,
                    TensorC* A_log,  TensorC* B_proj, TensorC* C_proj,
                    TensorC* D_skip, TensorC* delta,
                    TensorC* h0,     TensorC* cache,
                    TensorC* dx,     TensorC* dA,
                    TensorC* dB,     TensorC* dC,
                    TensorC* dD,     TensorC* ddelta
                );

                /* Convenience zero-allocators. */
                TensorC* tensor_mamba_alloc_state(int batch, int d_model, int d_state);
                TensorC* tensor_mamba_alloc_cache(int batch, int seq_len, int d_model, int d_state);

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
                DataFrame*  df_slice_rows(const DataFrame* df, size_t offset, size_t n);
                DataFrame*  df_head_rows(const DataFrame* df, size_t n);
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
                typedef struct Vocab Vocab;
                Vocab*  df_vocab_build(void* df, int col_idx, int max_features);
                void    vocab_free(Vocab* v);
                int     vocab_size(Vocab* v);
                TensorC* df_transform_bow(void* df, int col_idx, Vocab* v);
                void vocab_save(Vocab* v, const char* filepath);
                Vocab* vocab_load(const char* filepath);

                // ── Section 9: C Transform Pipeline ──────────────────────────
                typedef struct TransformPipeline TransformPipeline;

                TensorC** df_fit_transformers(const DataFrame* df,
                                              size_t train_rows,
                                              int text_col,
                                              const Vocab* vocab);

                TransformPipeline* pipeline_create(const Vocab*   vocab,
                                                    const TensorC* idf,
                                                    const TensorC* stds,
                                                    int text_col,
                                                    int label_col,
                                                    int n_classes);

                void pipeline_free(TransformPipeline* pl);

                TensorC** pipeline_transform_batch(const DataFrame*         df,
                                                    size_t                   offset,
                                                    size_t                   n,
                                                    const TransformPipeline* pl);

                // ── Section 22: Classical ML Extensions ──────────────────────
                TensorC* tensor_argmax_axis(TensorC* A, int axis);
                TensorC* tensor_pairwise_sq_l2(TensorC* A, TensorC* B);

                void tensor_exp_inplace(TensorC* A);
                void tensor_log_inplace(TensorC* A);
                void tensor_sqrt_inplace(TensorC* A);
                void tensor_sigmoid_inplace(TensorC* A);
                void tensor_tanh_inplace(TensorC* A);
                void tensor_relu_inplace(TensorC* A);

                void tensor_row_softmax_inplace(TensorC* A);

                TensorC* tensor_gbdt_compute_boundaries(TensorC* X, int Q);
                TensorC* tensor_gbdt_bin_samples(TensorC* X, TensorC* boundaries, int Q);
                void     tensor_gbdt_mse_grad_hess(TensorC* preds, TensorC* y, TensorC* out_g, TensorC* out_h);
                void     tensor_gbdt_logloss_grad_hess(TensorC* preds, TensorC* y, TensorC* out_g, TensorC* out_h);
                void     tensor_gbdt_histogram(TensorC* bins, TensorC* g, TensorC* h, TensorC* mask,
                                               int Q, TensorC* hist_g, TensorC* hist_h);
                void     tensor_gbdt_best_split(TensorC* hist_g, TensorC* hist_h, int Q,
                                               float sum_g, float sum_h, int node_n,
                                               float lambda, float gamma,
                                               int* out_feat, int* out_bin, float* out_gain);
                void     tensor_gbdt_split_node(TensorC* bins, TensorC* mask, int feat, int bin,
                                               TensorC* out_left, TensorC* out_right);
                float    tensor_gbdt_leaf_update(TensorC* preds, TensorC* mask,
                                                float sum_g, float sum_h, float lr, float lambda);
                TensorC* tensor_gbdt_predict_all(TensorC* X_bins, TensorC* feats, TensorC* thresholds,
                                                 TensorC* lefts, TensorC* rights,
                                                 TensorC* tree_sizes, float base_score);
                void     tensor_gbdt_hist_subtract(TensorC* parent_g, TensorC* parent_h,
                                                   TensorC* sibling_g, TensorC* sibling_h,
                                                   TensorC* out_g, TensorC* out_h);
                int      tensor_gbdt_train_tree(TensorC* bins, TensorC* g, TensorC* h,
                                                int Q, int max_leaves,
                                                float lambda, float alpha, float gamma,
                                                float min_hess, float lr,
                                                TensorC* preds,
                                                TensorC* out_feats, TensorC* out_thresholds,
                                                TensorC* out_lefts, TensorC* out_rights);

                TensorC* tensor_quantile_fit(TensorC* X, int n_quantiles);
                TensorC* tensor_quantile_transform(TensorC* X, TensorC* landmarks, int n_quantiles);

                TensorC* tensor_yj_fit(TensorC* X);
                TensorC* tensor_yj_transform(TensorC* X, TensorC* lambdas);

                // ── DataFrame lifecycle (internal allocation) ─────────────────
                DataFrame*  df_create(size_t n_rows, size_t n_cols);

                // ── Section 10–15: Extended DataFrame Operations ──────────────

                // Vectorized filtering
                DataFrame*  df_apply_mask(const DataFrame* df, const int32_t* mask);
                DataFrame*  df_where_f32(const DataFrame* df, int col_idx, int cmp_op, float val);
                DataFrame*  df_where_str(const DataFrame* df, int col_idx, const char* val);

                // Sorting
                DataFrame*  df_sort_by_col(const DataFrame* df, int col_idx, bool ascending);

                // GroupBy aggregation (group_col must be STRING / categorical)
                DataFrame*  df_groupby_agg(const DataFrame* df,
                                            int group_col_idx,
                                            const int* agg_col_idxs, int n_agg,
                                            int agg_type);
                DataFrame*  df_groupby_multi_agg(const DataFrame* df,
                                                   int group_col_idx,
                                                   const int* agg_col_idxs,
                                                   const int* agg_types,
                                                   int n);

                // Join / merge
                DataFrame*  df_join(const DataFrame* left, const DataFrame* right,
                                     int left_col_idx, int right_col_idx, int join_type);

                // Schema mutations
                DataFrame*  df_add_f32_column(const DataFrame* df, const char* name,
                                               const float* data, size_t n_rows);
                DataFrame*  df_drop_column_new(const DataFrame* df, int col_idx);
                void        df_rename_column(DataFrame* df, int col_idx, const char* new_name);
                DataFrame*  df_cast_to_f32(const DataFrame* df, int col_idx);
                DataFrame*  df_fill_null_f32(const DataFrame* df, int col_idx, float fill_val);
                DataFrame*  df_concat_rows(const DataFrame* a, const DataFrame* b);

                // Describe / sample / value counts
                TensorC*    df_describe(const DataFrame* df);
                DataFrame*  df_value_counts(const DataFrame* df, int col_idx);
                DataFrame*  df_sample_rows(const DataFrame* df, size_t n,
                                            bool replace, uint64_t seed);

                // ── BPE Tokenizer ─────────────────────────────────────────────
                typedef struct Tokenizer Tokenizer;

                Tokenizer*  tok_load_json(const char* tokenizer_json_path);
                Tokenizer*  tok_load(const char* vocab_path, const char* merges_path);
                void        tok_free(Tokenizer* tok);

                int32_t*    tok_encode(const Tokenizer* tok, const char* text,
                                        bool add_bos, int* n_out);
                TensorC*    tok_encode_batch(const Tokenizer* tok,
                                              const char** texts, int n_texts,
                                              bool add_bos, int max_len);

                char*       tok_decode(const Tokenizer* tok, const int32_t* ids,
                                        int n, bool skip_special);

                const char* tok_id_to_str(const Tokenizer* tok, int id);
                int         tok_str_to_id(const Tokenizer* tok, const char* str);
                bool        tok_is_special(const Tokenizer* tok, int id);

                int         tok_vocab_size(const Tokenizer* tok);
                int         tok_bos_id(const Tokenizer* tok);
                int         tok_eos_id(const Tokenizer* tok);
                int         tok_pad_id(const Tokenizer* tok);
                int         tok_unk_id(const Tokenizer* tok);

                // ── Inference Engine ──────────────────────────────────────────
                typedef struct InferenceSession InferenceSession;

                typedef struct {
                    int   arch;
                    int   vocab_size;
                    int   n_layers;
                    int   n_heads;
                    int   n_kv_heads;
                    int   d_model;
                    int   d_ff;
                    int   max_seq_len;
                    float rms_eps;
                    float rope_base;
                    float rope_scale;
                    float attn_scale;
                    bool  tie_embeddings;
                    int   bos_id;
                    int   eos_id;
                } ModelConfig;

                InferenceSession* inf_load(const char* model_dir,
                                            const ModelConfig* cfg,
                                            Tokenizer* tok);
                InferenceSession* inf_load_file(const char* weights_path,
                                                 const ModelConfig* cfg,
                                                 Tokenizer* tok);
                void              inf_free(InferenceSession* sess);

                TensorC*  inf_step(InferenceSession* sess,
                                    int32_t token_id, int pos);
                TensorC*  inf_forward(InferenceSession* sess,
                                       const int32_t* tokens, int n_tokens);
                void      inf_reset_kv(InferenceSession* sess);

                int32_t   inf_sample_greedy(const TensorC* logits);
                int32_t   inf_sample(InferenceSession* sess,
                                      const TensorC* logits,
                                      float temperature, float top_p);
                int       inf_generate_ids(InferenceSession* sess,
                                            const int32_t* prompt_ids, int n_prompt,
                                            int max_new_tokens,
                                            float temperature, float top_p,
                                            uint64_t seed,
                                            int32_t* out_ids);

                bool      inf_parse_config(const char* config_json_path,
                                            ModelConfig* cfg);
                TensorC*  inf_get_weight(const InferenceSession* sess,
                                          const char* name);
            ", $libPath);
        }
        return self::$ffi;
    }
}
