---
layout: default
title: Autograd Internals
---

# Autograd Internals

Autograd implements reverse-mode differentiation for tensor computations.

## Concept

- Variables wrap tensors and record operations.
- A computation tape records the forward pass.
- Backpropagation traverses the graph from outputs to inputs.

## Internal flow

```text
Forward pass
   ├─ operations create nodes
   └─ tape stores references
Backward pass
   └─ gradients propagate through nodes
```

## Memory behavior

- Gradients are stored as tensors in the autograd graph.
- The graph retains references to intermediate tensors until backward completes.
- Use `detach()` or explicit scope if intermediate reuse is not needed.

## When to use

- Use autograd for custom differentiable layers.
- Use it only when the estimator requires gradients.

## When not to use

- Do not use autograd for tree-based or non-differentiable models.
- Avoid building deep graphs for very large tensors without measuring memory usage.
