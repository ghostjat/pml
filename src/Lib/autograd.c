/*
 * autograd.c — Forward ops + fused accumulation kernels.
 *
 * Build:
 *   gcc -O2 -march=native -D_GNU_SOURCE -DAG_RUN_TESTS \
 *       graph.c autograd.c -L. -Wl,-rpath,. -ltensor \
 *       -lm -fopenmp -lopenblas -llapacke -o ag_test && ./ag_test
 */
#include "autograd.h"
#include <string.h>
#include <stdint.h>

/*
 * Force O3 + tree-loop-vectorize + loop-unrolling for all functions in this
 * translation unit.  This is necessary because GCC 14 at -O2 fails to
 * vectorize float loops with dynamic bounds (confirmed via -fopt-info-vec-all).
 * The pragma applies only to functions defined after this point in this TU;
 * it does not affect graph.c or tensor.c.
 */
#pragma GCC optimize("O3,tree-loop-vectorize,unroll-loops")

/* ════════════════════════════════════════════════════════════════════════════
 * FUSED ACCUMULATION KERNELS
 *
 * Requirements enforced:
 *   - `restrict` on every pointer → no aliasing, mandatory for SIMD auto-vec.
 *   - All loops are stride-1 over contiguous float buffers.
 *   - #pragma GCC ivdep before inner loops → overrides conservative alias
 *     analysis left over from pointer-provenance tracking.
 *   - __builtin_assume_aligned(..., 16) → malloc on x86-64 guarantees 16-byte
 *     alignment; this unlocks 128/256-bit aligned-load code-gen.
 *   - No branches in inner loops (relu uses select/ternary, not if/else).
 *   - No heap allocation.
 * ════════════════════════════════════════════════════════════════════════════ */

/*
 * dst[i] += src[i]
 * add backward: pass upstream gradient unchanged to both inputs.
 * Vectorises to: vmovups + vaddps + vmovups  (8 floats/cycle on AVX2)
 */
static void
kern_add_accum(float* restrict       dst,
               const float* restrict src,
               size_t n)
{
    dst = (float*)       AG_ASSUME_ALIGNED(dst, 16);
    src = (const float*) AG_ASSUME_ALIGNED(src, 16);
    #pragma GCC ivdep
    for (size_t i = 0; i < n; i++)
        dst[i] += src[i];
}

/*
 * dst[i] += dz[i] * other[i]
 * mul backward: grad of one input = upstream grad ⊙ the other input's data.
 * Vectorises to: vfmadd231ps (3 inputs, 1 FMA unit)
 */
static void
kern_mul_accum(float* restrict       dst,
               const float* restrict dz,
               const float* restrict other,
               size_t n)
{
    dst   = (float*)       AG_ASSUME_ALIGNED(dst,   16);
    dz    = (const float*) AG_ASSUME_ALIGNED(dz,    16);
    other = (const float*) AG_ASSUME_ALIGNED(other, 16);
    #pragma GCC ivdep
    for (size_t i = 0; i < n; i++)
        dst[i] += dz[i] * other[i];
}

/*
 * dst[i] += (input[i] > 0) ? dz[i] : 0
 *
 * ReLU backward.  The ternary select form is used over (float)(x > 0) to
 * avoid an integer-conversion step; GCC emits: vcmpps + vblendvps + vaddps.
 * The forward input is recomputed inline — zero mask-tensor allocation.
 */
static void
kern_relu_bwd_accum(float* restrict       dst,
                    const float* restrict dz,
                    const float* restrict input,
                    size_t n)
{
    dst   = (float*)       AG_ASSUME_ALIGNED(dst,   16);
    dz    = (const float*) AG_ASSUME_ALIGNED(dz,    16);
    input = (const float*) AG_ASSUME_ALIGNED(input, 16);
    #pragma GCC ivdep
    for (size_t i = 0; i < n; i++)
        dst[i] += input[i] > 0.0f ? dz[i] : 0.0f;
}

/*
 * dA[M,K] += dZ[M,N] @ B[K,N]ᵀ
 * dA[m,k]  = Σ_n  dZ[m,n] * B[k,n]
 *
 * Loop order m→k→n:
 *   - Both dZ[m,:] (N floats) and B[k,:] (N floats) are loaded stride-1.
 *   - The N-loop is a dot product — auto-vectorised as vfmadd + vhaddps / phsum.
 *   - dZ[m,:] stays in L1 across all K inner iterations (reuse).
 *   - Mirrors BLAS dgemm(NoTrans, Trans) for easy future BLAS replacement.
 */
static void
kern_dA_matmul(float* restrict       dA,
               const float* restrict dZ,
               const float* restrict B,
               int M, int K, int N)
{
    for (int m = 0; m < M; m++) {
        const float* dz_row = dZ + (size_t)m * N;
              float* da_row = dA + (size_t)m * K;
        for (int k = 0; k < K; k++) {
            const float* b_row = B + (size_t)k * N;
            float acc = 0.0f;
            #pragma GCC ivdep
            for (int n = 0; n < N; n++)
                acc += dz_row[n] * b_row[n];   /* dot product, vectorisable */
            da_row[k] += acc;
        }
    }
}

/*
 * dB[K,N] += A[M,K]ᵀ @ dZ[M,N]
 * dB[k,n]  = Σ_m  A[m,k] * dZ[m,n]
 *
 * Loop order m→k→n:
 *   - Inner n-loop is a pure FMA row-update: dB[k,:] += a_mk * dZ[m,:].
 *   - Both dB[k,:] (N floats) and dZ[m,:] (N floats) are stride-1.
 *   - Vectorises to: vbroadcastss a_mk + vfmadd231ps  (8 floats/cycle AVX2).
 *   - Mirrors BLAS dgemm(Trans, NoTrans) for easy future BLAS replacement.
 */
static void
kern_dB_matmul(float* restrict       dB,
               const float* restrict A,
               const float* restrict dZ,
               int M, int K, int N)
{
    for (int m = 0; m < M; m++) {
        const float* a_row  = A  + (size_t)m * K;
        const float* dz_row = dZ + (size_t)m * N;
        for (int k = 0; k < K; k++) {
            float  a_mk   = a_row[k];
            float* db_row = dB + (size_t)k * N;
            #pragma GCC ivdep
            for (int n = 0; n < N; n++)
                db_row[n] += a_mk * dz_row[n];  /* SAXPY, vectorisable */
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * BACKWARD FUNCTIONS
 *
 * Each function receives only its OpNode*.  All inputs, outputs, and grad
 * buffers are reached through OpNode members — no extra stack arguments.
 * All grad buffers were pre-allocated and zeroed in tape_backward Phase 1;
 * these functions perform pure in-place accumulation (zero allocations).
 * ════════════════════════════════════════════════════════════════════════════ */

AG_HOT static void bwd_add(OpNode* op)
{
    const float* dz = (const float*)op->output->grad->data;
    size_t       n  = op->output->grad->total_size;

    if (op->inputs[0]->requires_grad && op->inputs[0]->grad)
        kern_add_accum((float*)op->inputs[0]->grad->data, dz, n);

    if (op->inputs[1]->requires_grad && op->inputs[1]->grad)
        kern_add_accum((float*)op->inputs[1]->grad->data, dz, n);
}

AG_HOT static void bwd_mul(OpNode* op)
{
    const float* dz = (const float*)op->output->grad->data;
    size_t       n  = op->output->grad->total_size;

    /* grad_A += dZ ⊙ B;   grad_B += dZ ⊙ A
     * Correct when inputs[0] == inputs[1] (self-mul): each call accumulates
     * independently into the same grad buffer → net += 2 * dZ * A.data ✓  */
    if (op->inputs[0]->requires_grad && op->inputs[0]->grad)
        kern_mul_accum((float*)op->inputs[0]->grad->data,
                       dz,
                       (const float*)op->inputs[1]->data->data,
                       n);

    if (op->inputs[1]->requires_grad && op->inputs[1]->grad)
        kern_mul_accum((float*)op->inputs[1]->grad->data,
                       dz,
                       (const float*)op->inputs[0]->data->data,
                       n);
}

AG_HOT static void bwd_relu(OpNode* op)
{
    if (!op->inputs[0]->requires_grad || !op->inputs[0]->grad) return;

    kern_relu_bwd_accum(
        (float*)      op->inputs[0]->grad->data,
        (const float*)op->output->grad->data,
        (const float*)op->inputs[0]->data->data,   /* saved forward input */
        op->inputs[0]->data->total_size);
}

AG_HOT static void bwd_matmul(OpNode* op)
{
    VarNode* va = op->inputs[0];   /* A: [M,K] */
    VarNode* vb = op->inputs[1];   /* B: [K,N] */
    VarNode* vz = op->output;      /* Z: [M,N] */

    int M = va->data->shape[0];
    int K = va->data->shape[1];
    int N = vb->data->shape[1];

    const float* dZ = (const float*)vz->grad->data;
    const float* A  = (const float*)va->data->data;
    const float* B  = (const float*)vb->data->data;

    if (va->requires_grad && va->grad)   /* dA += dZ @ Bᵀ */
        kern_dA_matmul((float*)va->grad->data, dZ, B, M, K, N);

    if (vb->requires_grad && vb->grad)   /* dB += Aᵀ @ dZ */
        kern_dB_matmul((float*)vb->grad->data, A, dZ, M, K, N);
}

/* ════════════════════════════════════════════════════════════════════════════
 * FORWARD OPS
 *
 * Atomicity guarantee: capacity for both the VarNode slot and the OpNode slot
 * is pre-checked before any state is modified. On any failure, the tape is
 * left exactly as it was — no partial allocations, no orphan nodes.
 * ════════════════════════════════════════════════════════════════════════════ */

static inline bool any_rg(VarNode* a, VarNode* b)
{
    return (a && a->requires_grad) || (b && b->requires_grad);
}

VarNode* ag_var(Tape* tape, Tensor* data, bool requires_grad)
{
    if (AG_UNLIKELY(!tape || !data)) {
        tensor_set_error("ag_var: NULL tape or data");
        return NULL;
    }
    return tape_alloc_var(tape, data, requires_grad);
}

VarNode* ag_add(Tape* tape, VarNode* a, VarNode* b)
{
    if (AG_UNLIKELY(!tape || !a || !b || !a->data || !b->data)) {
        tensor_set_error("ag_add: NULL argument");
        return NULL;
    }

    bool rg = any_rg(a, b);

    /* Pre-check capacity — ensures all-or-nothing semantics. */
    if (AG_UNLIKELY(tape->n_vars >= tape->vars_cap)) {
        tensor_set_error("ag_add: vars pool exhausted");
        return NULL;
    }
    if (rg && AG_UNLIKELY(tape->n_ops >= tape->ops_cap)) {
        tensor_set_error("ag_add: ops pool exhausted");
        return NULL;
    }

    Tensor* out_data = tensor_add(a->data, b->data);
    if (AG_UNLIKELY(!out_data)) return NULL;

    /* Capacity pre-checked: these cannot return NULL. */
    VarNode* out     = tape_alloc_var(tape, out_data, rg);
    out->data_owned  = true;

    if (rg) {
        OpNode* op      = tape_alloc_op(tape);
        op->backward_fn = bwd_add;
        op->inputs[0]   = a;
        op->inputs[1]   = b;
        op->output      = out;
        op->n_inputs    = 2;
        out->op_idx     = tape->n_ops - 1;
    }
    return out;
}

VarNode* ag_mul(Tape* tape, VarNode* a, VarNode* b)
{
    if (AG_UNLIKELY(!tape || !a || !b || !a->data || !b->data)) {
        tensor_set_error("ag_mul: NULL argument");
        return NULL;
    }

    bool rg = any_rg(a, b);

    if (AG_UNLIKELY(tape->n_vars >= tape->vars_cap)) {
        tensor_set_error("ag_mul: vars pool exhausted");
        return NULL;
    }
    if (rg && AG_UNLIKELY(tape->n_ops >= tape->ops_cap)) {
        tensor_set_error("ag_mul: ops pool exhausted");
        return NULL;
    }

    Tensor* out_data = tensor_mul(a->data, b->data);
    if (AG_UNLIKELY(!out_data)) return NULL;

    VarNode* out     = tape_alloc_var(tape, out_data, rg);
    out->data_owned  = true;

    if (rg) {
        OpNode* op      = tape_alloc_op(tape);
        op->backward_fn = bwd_mul;
        op->inputs[0]   = a;
        op->inputs[1]   = b;
        op->output      = out;
        op->n_inputs    = 2;
        out->op_idx     = tape->n_ops - 1;
    }
    return out;
}

VarNode* ag_matmul(Tape* tape, VarNode* a, VarNode* b)
{
    if (AG_UNLIKELY(!tape || !a || !b || !a->data || !b->data)) {
        tensor_set_error("ag_matmul: NULL argument");
        return NULL;
    }
    if (AG_UNLIKELY(a->data->ndim != 2 || b->data->ndim != 2)) {
        tensor_set_error("ag_matmul: inputs must be 2-D tensors");
        return NULL;
    }
    if (AG_UNLIKELY(a->data->shape[1] != b->data->shape[0])) {
        tensor_set_error("ag_matmul: inner dimensions must match (A[M,K], B[K,N])");
        return NULL;
    }

    bool rg = any_rg(a, b);

    if (AG_UNLIKELY(tape->n_vars >= tape->vars_cap)) {
        tensor_set_error("ag_matmul: vars pool exhausted");
        return NULL;
    }
    if (rg && AG_UNLIKELY(tape->n_ops >= tape->ops_cap)) {
        tensor_set_error("ag_matmul: ops pool exhausted");
        return NULL;
    }

    Tensor* out_data = tensor_matmul(a->data, b->data);
    if (AG_UNLIKELY(!out_data)) return NULL;

    VarNode* out     = tape_alloc_var(tape, out_data, rg);
    out->data_owned  = true;

    if (rg) {
        OpNode* op      = tape_alloc_op(tape);
        op->backward_fn = bwd_matmul;
        op->inputs[0]   = a;
        op->inputs[1]   = b;
        op->output      = out;
        op->n_inputs    = 2;
        out->op_idx     = tape->n_ops - 1;
    }
    return out;
}

VarNode* ag_relu(Tape* tape, VarNode* a)
{
    if (AG_UNLIKELY(!tape || !a || !a->data)) {
        tensor_set_error("ag_relu: NULL argument");
        return NULL;
    }

    if (AG_UNLIKELY(tape->n_vars >= tape->vars_cap)) {
        tensor_set_error("ag_relu: vars pool exhausted");
        return NULL;
    }
    if (a->requires_grad && AG_UNLIKELY(tape->n_ops >= tape->ops_cap)) {
        tensor_set_error("ag_relu: ops pool exhausted");
        return NULL;
    }

    Tensor* out_data = tensor_relu(a->data);
    if (AG_UNLIKELY(!out_data)) return NULL;

    VarNode* out     = tape_alloc_var(tape, out_data, a->requires_grad);
    out->data_owned  = true;

    if (a->requires_grad) {
        OpNode* op      = tape_alloc_op(tape);
        op->backward_fn = bwd_relu;
        op->inputs[0]   = a;
        op->output      = out;
        op->n_inputs    = 1;
        out->op_idx     = tape->n_ops - 1;
    }
    return out;
}

/* ════════════════════════════════════════════════════════════════════════════
 * VALIDATION TESTS  (compile with -DAG_RUN_TESTS)
 * ════════════════════════════════════════════════════════════════════════════ */
#ifdef AG_RUN_TESTS
#include <stdio.h>
#include <math.h>
#include <assert.h>

static float sval(VarNode* v)  { return ((const float*)v->data->data)[0]; }
static float sgrad(VarNode* v) { return v->grad ? ((const float*)v->grad->data)[0] : 0.0f; }

/* ── Test 1: scalar chain  z = (a * b) + c ────────────────────────────── */
static void test_basic(void)
{
    printf("Test 1: z = (a * b) + c\n");
    Tape* tape = tape_create(0, 0);

    int s[1] = {1};
    Tensor* ta = tensor_create(1, s); ((float*)ta->data)[0] = 3.0f;
    Tensor* tb = tensor_create(1, s); ((float*)tb->data)[0] = 4.0f;
    Tensor* tc = tensor_create(1, s); ((float*)tc->data)[0] = 5.0f;

    VarNode* a  = ag_var(tape, ta, true);
    VarNode* b  = ag_var(tape, tb, true);
    VarNode* c  = ag_var(tape, tc, true);
    VarNode* ab = ag_mul(tape, a, b);
    VarNode* z  = ag_add(tape, ab, c);

    printf("  forward: z = %.1f  (expected 17.0)\n", sval(z));
    assert(fabsf(sval(z) - 17.0f) < 1e-6f);

    tape_backward(tape, z);

    printf("  grad_a  = %.1f  (expected 4.0)\n", sgrad(a));
    printf("  grad_b  = %.1f  (expected 3.0)\n", sgrad(b));
    printf("  grad_c  = %.1f  (expected 1.0)\n", sgrad(c));
    assert(fabsf(sgrad(a) - 4.0f) < 1e-6f);
    assert(fabsf(sgrad(b) - 3.0f) < 1e-6f);
    assert(fabsf(sgrad(c) - 1.0f) < 1e-6f);
    printf("  PASSED\n\n");

    tape_destroy(tape);
    tensor_free(ta); tensor_free(tb); tensor_free(tc);
}

/* ── Test 2: shared node  y = z + z ────────────────────────────────────── */
static void test_shared_node(void)
{
    printf("Test 2: y = z + z  (shared node)\n");
    Tape* tape = tape_create(0, 0);

    int s[1] = {1};
    Tensor* tz = tensor_create(1, s);
    ((float*)tz->data)[0] = 2.0f;

    VarNode* z = ag_var(tape, tz, true);
    VarNode* y = ag_add(tape, z, z);

    printf("  forward: y = %.1f  (expected 4.0)\n", sval(y));
    assert(fabsf(sval(y) - 4.0f) < 1e-6f);

    tape_backward(tape, y);

    printf("  grad_z  = %.1f  (expected 2.0)\n", sgrad(z));
    assert(fabsf(sgrad(z) - 2.0f) < 1e-6f);
    printf("  PASSED\n\n");

    tape_destroy(tape);
    tensor_free(tz);
}

/* ── Test 3: ReLU (gate closed + gate open) ─────────────────────────────── */
static void test_relu(void)
{
    printf("Test 3a: relu(-2) * 3  (gate closed)\n");
    {
        Tape* tape = tape_create(0, 0);
        int s[1] = {1};
        Tensor* ta = tensor_create(1, s); ((float*)ta->data)[0] = -2.0f;
        Tensor* tb = tensor_create(1, s); ((float*)tb->data)[0] =  3.0f;
        VarNode* a = ag_var(tape, ta, true);
        VarNode* b = ag_var(tape, tb, true);
        VarNode* r = ag_relu(tape, a);
        VarNode* z = ag_mul(tape, r, b);
        tape_backward(tape, z);
        printf("  grad_a = %.1f  (expected 0.0)\n", sgrad(a));
        printf("  grad_b = %.1f  (expected 0.0)\n", sgrad(b));
        assert(fabsf(sgrad(a)) < 1e-6f);
        assert(fabsf(sgrad(b)) < 1e-6f);
        printf("  PASSED\n");
        tape_destroy(tape);
        tensor_free(ta); tensor_free(tb);
    }
    printf("Test 3b: relu(5) * 3  (gate open)\n");
    {
        Tape* tape = tape_create(0, 0);
        int s[1] = {1};
        Tensor* ta = tensor_create(1, s); ((float*)ta->data)[0] = 5.0f;
        Tensor* tb = tensor_create(1, s); ((float*)tb->data)[0] = 3.0f;
        VarNode* a = ag_var(tape, ta, true);
        VarNode* b = ag_var(tape, tb, true);
        VarNode* r = ag_relu(tape, a);
        VarNode* z = ag_mul(tape, r, b);
        tape_backward(tape, z);
        printf("  grad_a = %.1f  (expected 3.0)\n", sgrad(a));
        printf("  grad_b = %.1f  (expected 5.0)\n", sgrad(b));
        assert(fabsf(sgrad(a) - 3.0f) < 1e-6f);
        assert(fabsf(sgrad(b) - 5.0f) < 1e-6f);
        printf("  PASSED\n\n");
        tape_destroy(tape);
        tensor_free(ta); tensor_free(tb);
    }
}

/* ── Test 4: 2×2 matmul gradients ─────────────────────────────────────── */
static void test_matmul(void)
{
    printf("Test 4: 2x2 matmul gradients\n");
    Tape* tape = tape_create(0, 0);

    int s[2] = {2, 2};
    Tensor* ta = tensor_create(2, s);
    Tensor* tb = tensor_create(2, s);
    float a_vals[4] = {1,2,3,4};
    float b_vals[4] = {5,6,7,8};
    memcpy(ta->data, a_vals, 4 * sizeof(float));
    memcpy(tb->data, b_vals, 4 * sizeof(float));

    VarNode* a = ag_var(tape, ta, true);
    VarNode* b = ag_var(tape, tb, true);
    VarNode* z = ag_matmul(tape, a, b);

    const float* zd = (const float*)z->data->data;
    printf("  Z = [%.0f %.0f / %.0f %.0f]  (exp 19 22 / 43 50)\n",
           zd[0], zd[1], zd[2], zd[3]);
    assert(fabsf(zd[0]-19)<1e-4f && fabsf(zd[1]-22)<1e-4f &&
           fabsf(zd[2]-43)<1e-4f && fabsf(zd[3]-50)<1e-4f);

    tape_backward(tape, z);

    const float* da = (const float*)a->grad->data;
    const float* db = (const float*)b->grad->data;
    printf("  dA = [%.0f %.0f / %.0f %.0f]  (exp 11 15 / 11 15)\n",
           da[0],da[1],da[2],da[3]);
    printf("  dB = [%.0f %.0f / %.0f %.0f]  (exp  4  4 /  6  6)\n",
           db[0],db[1],db[2],db[3]);
    assert(fabsf(da[0]-11)<1e-4f && fabsf(da[1]-15)<1e-4f &&
           fabsf(da[2]-11)<1e-4f && fabsf(da[3]-15)<1e-4f);
    assert(fabsf(db[0]-4)<1e-4f  && fabsf(db[1]-4)<1e-4f  &&
           fabsf(db[2]-6)<1e-4f  && fabsf(db[3]-6)<1e-4f);
    printf("  PASSED\n\n");

    tape_destroy(tape);
    tensor_free(ta); tensor_free(tb);
}

/* ── Test 5: double backward without tape_reset ────────────────────────── */
static void test_double_backward(void)
{
    printf("Test 5: double backward on same tape (without tape_reset)\n");
    Tape* tape = tape_create(0, 0);

    int s[1] = {1};
    Tensor* ta = tensor_create(1, s); ((float*)ta->data)[0] = 3.0f;
    Tensor* tb = tensor_create(1, s); ((float*)tb->data)[0] = 4.0f;
    VarNode* a = ag_var(tape, ta, true);
    VarNode* b = ag_var(tape, tb, true);
    VarNode* z = ag_mul(tape, a, b);   /* z = a*b = 12 */

    tape_backward(tape, z);
    printf("  1st bwd: grad_a=%.1f grad_b=%.1f  (expected 4.0 3.0)\n",
           sgrad(a), sgrad(b));
    assert(fabsf(sgrad(a) - 4.0f) < 1e-6f);
    assert(fabsf(sgrad(b) - 3.0f) < 1e-6f);

    /* Second backward — Phase 1 resets grads, should give same result. */
    tape_backward(tape, z);
    printf("  2nd bwd: grad_a=%.1f grad_b=%.1f  (expected 4.0 3.0)\n",
           sgrad(a), sgrad(b));
    assert(fabsf(sgrad(a) - 4.0f) < 1e-6f);
    assert(fabsf(sgrad(b) - 3.0f) < 1e-6f);
    printf("  PASSED\n\n");

    tape_destroy(tape);
    tensor_free(ta); tensor_free(tb);
}

/* ── Test 6: NULL safety ────────────────────────────────────────────────── */
static void test_null_safety(void)
{
    printf("Test 6: NULL safety (tape_destroy/reset/clear_grads with NULL)\n");
    tape_destroy(NULL);      /* must not crash */
    tape_reset(NULL);        /* must not crash */
    tape_clear_grads(NULL);  /* must not crash */
    printf("  PASSED\n\n");
}

/* ── Test 7: all-or-nothing on pool exhaustion ──────────────────────────── */
static void test_pool_exhaustion(void)
{
    printf("Test 7: pool exhaustion returns NULL cleanly\n");

    /* vars_cap=2: only room for 2 leaves — ag_add will fail (needs 3rd var slot) */
    Tape* tape = tape_create(4096, 2);
    int s[1] = {1};
    Tensor* ta = tensor_create(1, s); ((float*)ta->data)[0] = 1.0f;
    Tensor* tb = tensor_create(1, s); ((float*)tb->data)[0] = 2.0f;
    VarNode* a  = ag_var(tape, ta, true);
    VarNode* b  = ag_var(tape, tb, true);
    int n_vars_before = tape->n_vars;
    int n_ops_before  = tape->n_ops;
    VarNode* r  = ag_add(tape, a, b);   /* must fail — pool full */
    printf("  ag_add returned: %s  (expected NULL)\n", r ? "non-NULL" : "NULL");
    printf("  tape state unchanged: n_vars=%d==%d  n_ops=%d==%d\n",
           tape->n_vars, n_vars_before, tape->n_ops, n_ops_before);
    assert(r == NULL);
    assert(tape->n_vars == n_vars_before);
    assert(tape->n_ops  == n_ops_before);
    printf("  PASSED\n\n");

    tape_destroy(tape);
    tensor_free(ta); tensor_free(tb);
}

int main(void)
{
    test_basic();
    test_shared_node();
    test_relu();
    test_matmul();
    test_double_backward();
    test_null_safety();
    test_pool_exhaustion();
    printf("All 7 tests passed.\n");
    return 0;
}
#endif /* AG_RUN_TESTS */
