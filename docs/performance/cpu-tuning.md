---
layout: default
title: CPU Tuning
---

# CPU Tuning

The native backend is CPU-first. Effective tuning aligns threads, vectorization, and memory with physical hardware.

## Thread control

- `TensorEngine` exposes OpenMP-backed kernels.
- `Tensor::configureThreading()` may set BLAS and OpenMP thread counts.
- Use a small number of threads for high-core contention environments.

## BLAS settings

- `OPENBLAS_NUM_THREADS=1` is often the best default for single-process workloads.
- For multi-process jobs, constrain BLAS threads to avoid oversubscription.

## Cache locality

- Prefer batch sizes that fit in L2 cache when possible.
- Use contiguous row-major tensors for matrix multiplication.
- Avoid random access patterns inside the hot loop.

## NUMA and affinity

- Bind processes to local NUMA nodes for large datasets.
- Keep tensor buffers on the same memory node as CPU threads.

## When to use

- Apply CPU tuning to production training and inference.
- Adjust thread counts when the same machine runs multiple workers.

## When not to use

- Do not rely on default operating system thread scheduling for peak throughput.
- Do not enable aggressive OpenMP scaling without measuring end-to-end latency.
