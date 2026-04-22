/*
 * autograd.h — Differentiable ops layered on top of graph.h.
 *
 * Every ag_* function:
 *   1. Validates inputs and pre-checks tape capacity (atomic: no partial state
 *      changes on failure).
 *   2. Runs the forward op (calls the underlying tensor_* primitive).
 *   3. If any input has requires_grad, records an OpNode on the tape.
 *   4. Returns the output VarNode (tape-owned data; data_owned = true).
 *
 * Returns NULL on:
 *   - NULL tape or input VarNode
 *   - NULL data pointer inside a VarNode
 *   - Shape mismatch (matmul)
 *   - Tape capacity exhausted (ops or vars pool full)
 *   - Underlying tensor op failure (OOM for forward output)
 *
 * On NULL return the tape is guaranteed to be in the same state as before
 * the call (all-or-nothing semantics).
 *
 * Supported ops and their gradients:
 *   ag_add    z  = a + b       dA += dZ,        dB += dZ
 *   ag_mul    z  = a ⊙ b       dA += dZ ⊙ B,    dB += dZ ⊙ A
 *   ag_matmul Z  = A @ B       dA += dZ @ Bᵀ,   dB += Aᵀ @ dZ
 *   ag_relu   z  = max(0, a)   dA += dZ ⊙ (A > 0)  [mask recomputed inline]
 */
#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "graph.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Wrap an existing Tensor as a leaf VarNode.  Tape does NOT take ownership. */
VarNode* ag_var(Tape* tape, Tensor* data, bool requires_grad);

/* Element-wise add: z = a + b  (shapes must match) */
VarNode* ag_add(Tape* tape, VarNode* a, VarNode* b);

/* Element-wise mul: z = a ⊙ b  (shapes must match) */
VarNode* ag_mul(Tape* tape, VarNode* a, VarNode* b);

/* Matrix multiply: Z = A @ B  (A: [M,K], B: [K,N]) */
VarNode* ag_matmul(Tape* tape, VarNode* a, VarNode* b);

/* ReLU: z = max(0, a).  Backward recomputes mask inline; no mask tensor. */
VarNode* ag_relu(Tape* tape, VarNode* a);

#ifdef __cplusplus
}
#endif
#endif /* AUTOGRAD_H */
