/*
 * graph.h — Computation tape for autograd.
 *
 * Design constraints:
 *   - Contiguous OpNode / VarNode storage; NO realloc after tape_create()
 *     so pointers into those arrays remain stable for the tape's lifetime.
 *   - Zero allocations in the backward loop (tape_backward Phase 2).
 *   - Grad tensors allocated in one pre-pass (Phase 1), freed on tape_reset().
 *   - Forward output tensors owned by the tape (data_owned = true).
 *
 * Double-backward guarantee:
 *   tape_backward() may be called multiple times on the same tape without
 *   calling tape_reset() in between. Phase 1 zeros all existing grad buffers
 *   before re-seeding root, so each call produces correct independent grads.
 *
 * Build:
 *   gcc -O2 -march=native -D_GNU_SOURCE -DAG_RUN_TESTS \
 *       graph.c autograd.c -L. -ltensor -lm -fopenmp -o ag_test
 */
#ifndef GRAPH_H
#define GRAPH_H

#include "tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Forward declarations ───────────────────────────────────────────────── */
typedef struct VarNode VarNode;
typedef struct OpNode  OpNode;
typedef struct Tape    Tape;

/* ── Compile-time limits ────────────────────────────────────────────────── */
#define AG_MAX_INPUTS    2      /* max inputs per op (add/mul/matmul = 2)   */
#define AG_DEFAULT_OPS   4096   /* default OpNode pool size                 */
#define AG_DEFAULT_VARS  8192   /* default VarNode pool size                */

/* ── Portable compiler hint macros ─────────────────────────────────────── */
#if defined(__GNUC__) || defined(__clang__)
#  define AG_LIKELY(x)     __builtin_expect(!!(x), 1)
#  define AG_UNLIKELY(x)   __builtin_expect(!!(x), 0)
#  define AG_HOT           __attribute__((hot))
#  define AG_COLD          __attribute__((cold))
#  define AG_NOINLINE      __attribute__((noinline))
#  define AG_ASSUME_ALIGNED(p, a) __builtin_assume_aligned((p), (a))
#else
#  define AG_LIKELY(x)     (x)
#  define AG_UNLIKELY(x)   (x)
#  define AG_HOT
#  define AG_COLD
#  define AG_NOINLINE
#  define AG_ASSUME_ALIGNED(p, a) (p)
#endif

/* ── OpNode ─────────────────────────────────────────────────────────────── */
/*
 * backward_fn is placed first so it is loaded in the same cache fetch as
 * the rest of the struct during the backward sweep (hot access pattern).
 */
struct OpNode {
    void   (*backward_fn)(OpNode*);         /* 8 B — NULL ⇒ no-grad op      */
    VarNode* inputs[AG_MAX_INPUTS];         /* 16 B — stable ptrs into vars[]*/
    VarNode* output;                        /* 8 B                           */
    int      n_inputs;                      /* 4 B                           */
    /* 4 B padding → 40 B total */
};

/* ── VarNode ─────────────────────────────────────────────────────────────── */
/*
 * Pointers first (8-byte alignment), then int, then bools — 24 B total,
 * no implicit padding wasted inside the struct.
 */
struct VarNode {
    Tensor*  data;           /* forward value; externally owned (leaf) or tape-owned */
    Tensor*  grad;           /* gradient; NULL until tape_backward(); tape-owned      */
    int      op_idx;         /* index of producing OpNode in tape->ops[]; -1 = leaf   */
    bool     requires_grad;
    bool     data_owned;     /* if true, tape_reset() calls tensor_free(data)         */
    /* 2 B implicit padding */
};

/* ── Tape ────────────────────────────────────────────────────────────────── */
/*
 * ops[]  — topological forward order; backward iterates in reverse.
 * vars[] — all VarNodes for this tape (leaves and op outputs).
 *
 * Pointer-stability guarantee: tape_alloc_var() / tape_alloc_op() return NULL
 * rather than reallocating, so stored VarNode* pointers inside OpNodes are
 * permanently valid for the lifetime of the tape.
 */
struct Tape {
    OpNode*  ops;
    VarNode* vars;
    int      n_ops,  ops_cap;
    int      n_vars, vars_cap;
};

/* ── Lifecycle API ───────────────────────────────────────────────────────── */

/* Allocate tape (0 ⇒ use AG_DEFAULT_* capacities). */
Tape*    tape_create(int ops_cap, int vars_cap);

/*
 * Free all grad tensors, all tape-owned data tensors; reset op/var counts.
 * Safe to call with t == NULL.
 */
void     tape_reset(Tape* t);

/* tape_reset() then free the tape struct itself. Safe with t == NULL. */
void     tape_destroy(Tape* t);

/*
 * Zero all allocated grad buffers without freeing them.
 * Cheaper than tape_reset() for re-running backward on the same graph.
 * Safe with t == NULL.
 */
void     tape_clear_grads(Tape* t);

/* ── Node allocation ─────────────────────────────────────────────────────── */

/*
 * Reserve a VarNode slot from the fixed pool.
 * Returns NULL if the pool is exhausted; data ownership stays with the caller.
 */
VarNode* tape_alloc_var(Tape* t, Tensor* data, bool requires_grad);

/*
 * Reserve an OpNode slot from the fixed pool (zero-initialised).
 * Returns NULL if the pool is exhausted.
 */
OpNode*  tape_alloc_op(Tape* t);

/* ── Backward pass ───────────────────────────────────────────────────────── */

/*
 * Run backward from root.
 *
 * Phase 1 (allocation — before the hot loop):
 *   For every requires_grad VarNode:
 *     - If no grad buffer exists → allocate (zeroed) via tensor_create().
 *     - If a grad buffer already exists → zero it (supports multi-backward).
 *   Then seed root->grad = 1.
 *
 * Phase 2 (hot loop — ZERO allocations):
 *   Iterate ops[] in reverse topological order; call backward_fn per op.
 *
 * DAG correctness:
 *   ops[] is recorded in forward topological order, so reverse order
 *   guarantees each op's output grad is fully accumulated before its
 *   backward_fn executes. Shared nodes accumulate correctly via in-place +=.
 *
 * Preconditions: t != NULL, root != NULL, root->requires_grad == true.
 */
void     tape_backward(Tape* t, VarNode* root);

#ifdef __cplusplus
}
#endif
#endif /* GRAPH_H */
