---
name: Performance Regression
about: Report a measurable slowdown, increased memory usage, or throughput degradation
title: "[PERF] "
labels: ["performance", "regression", "needs-triage"]
assignees: ghostjat
---

## What regressed?

<!-- Which operation, model, or workflow became slower or used more memory? -->

## Measurements

### Before

<!-- PML version, commit hash, or date when performance was last known good -->

| Metric | Value |
|---|---|
| PML version / commit | |
| Operation | |
| Time | |
| Memory (RSS) | |
| Throughput | |

### After

| Metric | Value |
|---|---|
| PML version / commit | |
| Operation | |
| Time | |
| Memory (RSS) | |
| Throughput | |

## Benchmark Script

<!-- The exact PHP or PHPBench code you used to measure this -->

```php
<?php
// benchmark script
```

Or PHPBench class:

```bash
vendor/bin/phpbench run benchmarks/MyBench.php --report=aggregate
```

## Environment

| Field | Value |
|---|---|
| PHP version | |
| OS | |
| CPU | |
| GCC version | |
| OpenBLAS version | |
| OpenMP threads | <!-- OMP_NUM_THREADS --> |

## Suspected Cause

<!-- Do you have a hypothesis? A recent commit, build flag change, dependency update? -->

## Related Code Path

<!-- Which C function or PHP method do you think is involved?
     e.g., `tensor_matmul`, `GBDTClassifier::train`, `Dense::forward` -->

## Profiling Data (if available)

<!-- Callgrind, gprof, perf record, or PHP Xdebug profiler output -->

```
paste profile output or attach file
```
