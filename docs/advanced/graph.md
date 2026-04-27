---
layout: default
title: Compute Graph
---

# Compute Graph

The compute graph represents operator dependencies during tensor computation.

## Concept

- Each operation is a node in the graph.
- Nodes maintain pointers to input tensors.
- Backward passes traverse the graph in reverse.

## Graph structure

- Nodes are created by tensor operations.
- Graphs are acyclic for feedforward workloads.
- Intermediate tensors can be shared across downstream nodes.

## Performance implications

- Graph depth and width determine backward memory usage.
- Shared intermediate results reduce recomputation but increase lifetime.
- Avoid unnecessary graph nodes in production inference.

## When to use

- Use the graph for custom gradients and model introspection.
- Use it in debugging or research workflows.

## When not to use

- Do not keep the graph alive beyond a single backward pass unless required.
- Do not enable autograd for pure inference workloads.
