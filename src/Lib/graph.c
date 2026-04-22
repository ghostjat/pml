#include "graph.h"
#include <stdlib.h>
#include <string.h>

/* ── Internal helpers ───────────────────────────────────────────────────── */

/*
 * Allocate a zeroed FLOAT32 grad tensor matching v->data's shape.
 * Called only during tape_backward Phase 1 (cold path).
 */
AG_NOINLINE AG_COLD
static Tensor* make_grad(VarNode* v)
{
    /* v->data must be non-NULL at this point (invariant: forward ops set it) */
    return tensor_create(v->data->ndim, v->data->shape);
}

/* ════════════════════════════════════════════════════════════════════════════
 * Tape lifecycle
 * ════════════════════════════════════════════════════════════════════════════ */

Tape* tape_create(int ops_cap, int vars_cap)
{
    if (ops_cap  <= 0) ops_cap  = AG_DEFAULT_OPS;
    if (vars_cap <= 0) vars_cap = AG_DEFAULT_VARS;

    Tape* t = (Tape*)malloc(sizeof(Tape));
    if (AG_UNLIKELY(!t)) return NULL;

    t->ops  = (OpNode*) malloc((size_t)ops_cap  * sizeof(OpNode));
    t->vars = (VarNode*)malloc((size_t)vars_cap * sizeof(VarNode));

    if (AG_UNLIKELY(!t->ops || !t->vars)) {
        free(t->ops);
        free(t->vars);
        free(t);
        return NULL;
    }

    t->n_ops  = 0;  t->ops_cap  = ops_cap;
    t->n_vars = 0;  t->vars_cap = vars_cap;
    return t;
}

void tape_reset(Tape* t)
{
    if (AG_UNLIKELY(!t)) return;

    for (int i = 0; i < t->n_vars; i++) {
        VarNode* v = &t->vars[i];
        if (v->grad) {
            tensor_free(v->grad);
            v->grad = NULL;
        }
        if (v->data_owned && v->data) {
            tensor_free(v->data);
            v->data = NULL;
        }
    }
    t->n_ops  = 0;
    t->n_vars = 0;
}

void tape_destroy(Tape* t)
{
    if (AG_UNLIKELY(!t)) return;
    tape_reset(t);
    free(t->ops);
    free(t->vars);
    free(t);
}

void tape_clear_grads(Tape* t)
{
    if (AG_UNLIKELY(!t)) return;
    for (int i = 0; i < t->n_vars; i++) {
        VarNode* v = &t->vars[i];
        if (v->grad) tensor_fill(v->grad, 0.0f);
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Node allocation
 * ════════════════════════════════════════════════════════════════════════════ */

VarNode* tape_alloc_var(Tape* t, Tensor* data, bool requires_grad)
{
    if (AG_UNLIKELY(t->n_vars >= t->vars_cap)) return NULL;
    VarNode* v       = &t->vars[t->n_vars++];
    v->data          = data;
    v->grad          = NULL;
    v->op_idx        = -1;
    v->requires_grad = requires_grad;
    v->data_owned    = false;
    return v;
}

OpNode* tape_alloc_op(Tape* t)
{
    if (AG_UNLIKELY(t->n_ops >= t->ops_cap)) return NULL;
    OpNode* op = &t->ops[t->n_ops++];
    memset(op, 0, sizeof(OpNode));
    return op;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Backward pass
 * ════════════════════════════════════════════════════════════════════════════ */

AG_HOT
void tape_backward(Tape* t, VarNode* root)
{
    if (AG_UNLIKELY(!t || !root || !root->requires_grad)) return;

    /*
     * Phase 1 — allocate or zero all grad tensors.
     * This is the ONLY allocation window; Phase 2 (the hot loop) is zero-alloc.
     *
     * Handling existing grads is the key to supporting multiple backward passes
     * on the same tape without calling tape_reset():
     *   - New grad  → tensor_create() returns a zeroed buffer.
     *   - Old grad  → tensor_fill(..., 0.0f) resets it for re-use.
     */
    for (int i = 0; i < t->n_vars; i++) {
        VarNode* v = &t->vars[i];
        if (!v->requires_grad || !v->data) continue;

        if (AG_LIKELY(v->grad != NULL)) {
            /* Grad buffer already exists from a previous backward — just zero. */
            tensor_fill(v->grad, 0.0f);
        } else {
            /* First backward: allocate a zeroed grad buffer. */
            v->grad = make_grad(v);
            if (AG_UNLIKELY(!v->grad)) {
                tensor_set_error("tape_backward: OOM allocating grad tensor");
                return;
            }
        }
    }

    /* Seed: d(root)/d(root) = 1 for every element. */
    tensor_fill(root->grad, 1.0f);

    /*
     * Phase 2 — O(N) reverse-topological sweep. ZERO heap allocations.
     *
     * Correctness for shared nodes (diamond DAGs):
     *   ops[] is in forward topological order ⇒ reverse order guarantees that
     *   every consumer of a var runs its backward BEFORE the var's producer.
     *   By the time the producer runs, its output grad has received all upstream
     *   contributions via in-place accumulation in each consumer's backward_fn.
     */
    for (int i = t->n_ops - 1; i >= 0; i--) {
        OpNode* op = &t->ops[i];
        /* backward_fn is the first field — already in the same cache line. */
        if (AG_LIKELY(op->backward_fn && op->output && op->output->grad))
            op->backward_fn(op);
    }
}
