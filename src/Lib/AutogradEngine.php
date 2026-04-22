<?php
declare(strict_types=1);

namespace Pml\Lib;

/**
 * FFI singleton for the autograd computation-graph engine.
 *
 * Declares VarNode / Tape structs and all tape_* / ag_* functions from
 * graph.h + autograd.h.  The same libtensor.so as TensorEngine is used —
 * graph.c + autograd.c are compiled into it.
 *
 * VarNode.data / VarNode.grad are TensorC* pointers so they compose
 * directly with TensorEngine::get() operations.
 */
final class AutogradEngine
{
    private static ?\FFI $ffi = null;

    public static function get(): \FFI
    {
        if (self::$ffi === null) {
            $libPath = __DIR__ . '/libtensor.so';
            if (!file_exists($libPath)) {
                throw new \RuntimeException(
                    '[AutogradEngine] libtensor.so not found — build it via TensorEngine first.'
                );
            }

            self::$ffi = \FFI::cdef('
                /* ── Tensor (mirror of TensorC in TensorEngine) ──────────────── */
                typedef struct {
                    int    ndim;
                    int    shape[8];
                    size_t stride[8];
                    size_t total_size;
                    size_t byte_size;
                    bool   owns_data;
                    bool   is_arena;
                    int    dtype;
                    void*  data;
                } TensorC;

                /* ── OpNode — opaque from PHP side ───────────────────────────── */
                typedef struct OpNode OpNode;

                /* ── VarNode ─────────────────────────────────────────────────── */
                typedef struct {
                    TensorC* data;
                    TensorC* grad;
                    int      op_idx;
                    bool     requires_grad;
                    bool     data_owned;
                } VarNode;

                /* ── Tape ────────────────────────────────────────────────────── */
                typedef struct {
                    OpNode*  ops;
                    VarNode* vars;
                    int      n_ops;
                    int      ops_cap;
                    int      n_vars;
                    int      vars_cap;
                } Tape;

                /* ── Tape lifecycle ──────────────────────────────────────────── */
                Tape*    tape_create(int ops_cap, int vars_cap);
                void     tape_reset(Tape* t);
                void     tape_destroy(Tape* t);
                void     tape_clear_grads(Tape* t);
                void     tape_backward(Tape* t, VarNode* root);

                /* ── Autograd ops ────────────────────────────────────────────── */
                VarNode* ag_var(Tape* tape, TensorC* data, bool requires_grad);
                VarNode* ag_add(Tape* tape, VarNode* a, VarNode* b);
                VarNode* ag_mul(Tape* tape, VarNode* a, VarNode* b);
                VarNode* ag_matmul(Tape* tape, VarNode* a, VarNode* b);
                VarNode* ag_relu(Tape* tape, VarNode* a);

                /* ── Error handling (shared with TensorEngine) ───────────────── */
                bool        tensor_check_error(void);
                const char* tensor_get_last_error(void);
                void        tensor_clear_error(void);
            ', $libPath);
        }

        return self::$ffi;
    }
}
