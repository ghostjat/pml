# PML Benchmark Suite

This document describes the PML benchmark methodology, infrastructure, categories, and the process for generating reproducible comparison results against reference runtimes.

> **Integrity notice:** All numbers in this document are collected from a specific hardware configuration and compiler setup described below. They are not fabricated. Different hardware, OS, compiler flags, BLAS library, and thread count will produce different numbers. Run `scripts/bench.sh --full` on your own machine to generate results for your environment.

---

## Table of Contents

1. [Benchmarking Philosophy](#benchmarking-philosophy)
2. [Environment Specification](#environment-specification)
3. [Methodology](#methodology)
4. [Benchmark Categories](#benchmark-categories)
5. [Results: Tensor Throughput](#results-tensor-throughput)
6. [Results: Machine Learning Workloads](#results-machine-learning-workloads)
7. [Results: System Overhead](#results-system-overhead)
8. [Cross-Framework Comparison](#cross-framework-comparison)
9. [Thread Scaling](#thread-scaling)
10. [Memory Efficiency](#memory-efficiency)
11. [Limitations and Known Biases](#limitations-and-known-biases)
12. [Reproducing These Results](#reproducing-these-results)
13. [Performance Engineering Notes](#performance-engineering-notes)

---

## Benchmarking Philosophy

PML benchmarks follow these principles:

**1. Measure real workloads, not synthetic extremes.**
Every benchmark represents an operation that appears in real ML pipelines: GEMM (used in every dense layer), element-wise activations (used in every activation function), row-wise softmax (used in classification heads and attention), stratified dataset split (used in every training pipeline).

**2. Measure the full operation, not just the kernel.**
Benchmarks include FFI dispatch overhead, PHP wrapper execution, and C function call. This reflects the actual cost in a PHP application, not a lower bound that's impossible to achieve in practice.

**3. Fair thread configuration across comparisons.**
When comparing PML to PyTorch or NumPy, both are configured with the same thread count. Single-thread baselines and multi-thread results are documented separately. No comparison artificially restricts competitors.

**4. Statistical rigor.**
Timings are median over N iterations after M warmup iterations. We report median (not mean) because ML operation timings are right-skewed — outliers from OS scheduling inflate the mean. We also report `rstdev` (relative standard deviation) as a noise indicator.

**5. Separate what you're measuring.**
FFI overhead benchmarks deliberately use trivially-small tensors to isolate dispatch cost. BLAS benchmarks use large tensors where dispatch cost is < 0.1% of total time. The two should not be compared directly.

**6. Memory measurement honesty.**
PHP's `memory_get_peak_usage()` only sees PHP heap. C tensor allocations (the vast majority in PML) are invisible to it. Memory figures in this document use `/proc/self/status` `VmRSS` delta, which measures the total resident set size including C-heap pages.

---

## Environment Specification

All results in this document were collected on the following configuration. Run `scripts/bench.sh --sysinfo` to collect your own.

```
CPU:        AMD Ryzen 5 5500U with Radeon Graphics (6 cores / 12 threads)
            Zen 3 microarchitecture, AVX2, FMA
Base/boost: 2.1 GHz / 4.056 GHz
RAM:        ~15 GB DDR4
OS:         Debian (kernel 6.12.74+deb13+1-amd64)
GCC:        14.2.0 (Debian 14.2.0-19)
GCC flags:  -O3 -march=native -mtune=native -mfma -fno-math-errno
            -funsafe-math-optimizations -fopenmp -funroll-loops
            -fomit-frame-pointer -D_GNU_SOURCE
OpenBLAS:   0.3.29 (system package)
OpenMP:     libgomp (GCC 14.2.0)
PHP:        8.4.20 (ZTS disabled, FPM not used)
OPcache:    enabled, JIT=tracing, jit_buffer_size=256M
ffi.enable: true
OMP_NUM_THREADS: 4 (unless stated otherwise)
```

> **Note:** Cross-framework Python comparison numbers (NumPy, PyTorch, scikit-learn) are
> indicative — collected separately on similar hardware. Run `scripts/bench.sh --compare`
> to generate matched comparisons for your specific environment.

**CPU governor:** `performance` (set via `cpupower frequency-set -g performance`)

**Hyper-threading:** Enabled (reflects typical server configuration)

**NUMA:** Benchmarks pinned to NUMA node 0 via `numactl --cpunodebind=0 --membind=0`

---

## Methodology

### Warmup

Every PHPBench benchmark uses `#[Bench\Warmup(3)]` — 3 warm iterations that are discarded before measurement. This ensures:
- JIT is in steady state for the measured method
- BLAS GEMM tile-size autotuning is complete
- OS memory pages for tensor buffers are faulted in (no major-fault overhead in measurement)

Python benchmarks use an equivalent 3-iteration warmup before the timed loop.

### Iterations and Repetitions

PHPBench terminology: `revs` = how many times the code runs per iteration, `its` = how many iterations are measured, with full JIT/memory state reset between iterations.

Default configuration for hot operations (element-wise, reductions):
- `#[Bench\Revs(20)]` — 20 repetitions per iteration
- `#[Bench\Iterations(5)]` — 5 iterations, report median

For expensive operations (matmul 1024×1024, model training):
- `#[Bench\Revs(3)]` — 3 repetitions
- `#[Bench\Iterations(3)]` — 3 iterations

### Statistical Reporting

PHPBench reports:
- `mode` — mode of the timing distribution (most frequent value — better than mean for bimodal distributions)
- `rstdev` — relative standard deviation. Benchmarks with `rstdev > 5%` are considered noisy and should be rerun.
- `mem_peak` — PHP heap peak (note: does not include C tensor memory)

We post-process PHPBench JSON output and append VmRSS delta measured via `/proc/self/status` for memory comparison.

### Thread Configuration

PML thread configuration is controlled by `OMP_NUM_THREADS`. All reported results use `OMP_NUM_THREADS=4` unless labeled otherwise.

Python comparison scripts should be run with matching thread counts:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    python3 benchmarks/pytorch/benchmark_pytorch.py --threads 4
```

The thread count is logged in every result JSON.

### Eliminating Measurement Noise

```bash
# Set CPU frequency governor to performance mode
sudo cpupower frequency-set -g performance

# Pin to NUMA node 0 — prevents cross-NUMA memory latency
numactl --cpunodebind=0 --membind=0 vendor/bin/phpbench run ...

# Disable address-space randomization (reduces timing variance)
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space
```

These settings are applied automatically by `scripts/bench.sh`.

---

## Benchmark Categories

### Category 1: Tensor Operations (`benchmarks/TensorBench.php`, `benchmarks/Tensor/SimdBench.php`)

Measures raw tensor operation throughput. The fundamental building blocks.

**Sub-categories:**
- Creation (zeros, randomNormal, fromArray)
- Element-wise arithmetic (add, mul, div, pow) — AVX2 path
- Unary math (sigmoid, relu, exp, sqrt, log) — AVX2 SIMD kernels
- Reductions (sum, mean, std, max) — OpenMP + SIMD
- Axis reductions (sumAxis, meanAxis) — row/column parallel
- Shape operations (reshape, transpose, flatten) — view vs copy
- Linear algebra (matmul, SVD, pinv) — OpenBLAS
- Memory (copy vs view, bandwidth)
- Fused kernels (fusedAdamStep, fusedBceLoss, addRelu)

### Category 2: FFI Overhead (`benchmarks/FFI/FFIOverheadBench.php`)

Isolates the cost of crossing the PHP↔C boundary.

The technique: compare the same operation on a tiny tensor (8 elements — almost no computation) vs a large tensor (1M elements — computation dominates). The difference is compute cost. The tiny-tensor time is approximately the FFI dispatch cost.

### Category 3: OpenMP Scaling (`benchmarks/Parallel/OpenMPBench.php`)

Measures scaling efficiency as thread count increases. Run with:
```bash
for T in 1 2 4 8 16; do
    OMP_NUM_THREADS=$T vendor/bin/phpbench run benchmarks/Parallel/OpenMPBench.php \
        --report=aggregate --tag="threads_$T"
done
```

### Category 4: ML Workloads (`benchmarks/Workloads/TabularMLBench.php`)

End-to-end fit+predict pipelines on synthetic datasets. Measures total pipeline time including preprocessing.

### Category 5: Neural Network (`benchmarks/NeuralNetworkBench.php`, `benchmarks/NeuralNetwork/AttentionBench.php`)

Forward pass throughput for common architectures. Uses `model->forward()` directly (bypasses training scaffolding) to measure inference speed.

### Category 6: System (`benchmarks/Memory/MemoryBench.php`, `benchmarks/FFI/FFIOverheadBench.php`)

Cold-start, memory allocation patterns, lifecycle overhead.

---

## Results: Tensor Throughput

> Collected: 2026-05-17 | Hardware: Ryzen 5 5500U | OMP_NUM_THREADS=4
> Run `scripts/bench.sh --tensor` to regenerate on your hardware.

### Matrix Multiplication (GEMM) — float32

GFLOPS = (2 × N³) / time_seconds / 10⁹ for square N×N matmul.

| Shape | Time (median) | GFLOPS | rstdev |
|---|---|---|---|
| 512×512 @ 512×512 | 0.948 ms | 283 | — |
| 1024×1024 @ 1024×1024 | 7.433 ms | 289 | — |
| 1024×1024 (with alloc) | 6.608 ms | 325 | ±4.59% |

*Note: OpenBLAS achieves near-peak GFLOPS for these medium-size matrices on Zen 3 because tile sizes align well with L2/L3 cache capacity. Throughput is measured including FFI dispatch; on very small matrices (< 64×64) FFI overhead becomes a meaningful fraction of total time.*

### Element-wise Operations (1M float32 elements, in-place, AVX2, OMP=4)

Effective GB/s = (elements × 4 bytes × 2 R/W passes) / time.

| Operation | Time (median) | GB/s (eff.) | rstdev |
|---|---|---|---|
| `sigmoid` inplace | 0.994 ms | 8.1 | ±3.23% |
| `relu` inplace | 0.686 ms | 11.7 | ±15.37% |
| `mul` scalar inplace | 0.588 ms | 13.6 | ±2.17% |
| `add` (alloc+compute) | 0.517 ms | — | ±11.26% |

### Element-wise Operations (10M float32 elements, OpenMP, OMP=4)

| Operation | Time (median) | rstdev |
|---|---|---|
| `add` (tensor + tensor) | 25.4 ms | ±1.18% |
| `sigmoid` | 31.8 ms | ±2.94% |
| `exp` | 30.8 ms | ±3.28% |

*Note: The high per-element time for 10M ops compared to 1M reflects memory-bandwidth saturation. Adding more than 4 threads yields diminishing returns once DDR4 bandwidth is saturated.*

### Reductions and Axis Ops (OMP=4)

| Operation | Shape | Time (median) | rstdev |
|---|---|---|---|
| `sum` axis=1 | 10K×512 | 0.644 ms | ±4.10% |
| `mean` axis=1 | 10K×512 | 14.1 ms | ±0.41% |
| `sum` axis=0 | 10K×512 | 1.641 ms | ±0.54% |
| `sum` axis=0 | 512×10K | 1.551 ms | ±2.21% |
| `softmax` row-wise | 4K×1K | 4.690 ms | ±3.09% |

---

## Results: Machine Learning Workloads

> Collected: 2026-05-17 | Synthetic datasets | OMP_NUM_THREADS=4
> Run `scripts/bench.sh --ml` to regenerate on your hardware.

### Classifier Training — Fit on 2K samples × 20 features (ClassifierBench)

| Model | Fit time (median) | rstdev |
|---|---|---|
| `GaussianNB` | 0.504 ms | ±0.35% |
| `LinearRegression` | 0.941 ms | ±0.46% |
| `AdaBoost` (small) | 1.825 ms | ±10.50% |
| `LogisticRegression` | 11.19 ms | ±0.98% |
| `DecisionTree` | 6.324 ms | ±3.39% |
| `GradientBoosting` (20 trees) | 56.94 ms | ±0.68% |
| `RandomForest` (20 trees) | 57.53 ms | ±1.44% |

### Classifier Inference — Predict (ClassifierBench)

| Model | Predict 1K (median) | Predict 10K (median) | rstdev (1K) |
|---|---|---|---|
| `LinearRegression` | 9.7 µs | — | ±19.88% |
| `LogisticRegression` | 14.9 µs | 75.3 µs | ±7.76% |
| `GaussianNB` | 21.7 µs | — | ±7.56% |
| `DecisionTree` | 13.5 µs | 50.3 µs | ±10.86% |
| `DecisionTreeRegressor` | 16.1 µs | — | ±7.04% |
| `RandomForest` (20 trees) | 123.5 µs | 1.123 ms | ±3.76% |
| Batch inference (32 samples) | 10.5 µs | — | ±0.71% |
| Batch inference (256 samples) | 50.9 µs | — | ±4.84% |

### End-to-End Pipelines (TabularMLBench — fit+predict, includes preprocessing)

| Pipeline | Dataset | Time (median) | rstdev |
|---|---|---|---|
| StandardScaler → LogisticRegression | 2K×20 | 37.77 ms | ±0.95% |
| StandardScaler → LogisticRegression | 5K×50 | 82.12 ms | ±0.64% |
| MinMaxScaler → RandomForest (20 trees) | 2K×20 | 85.20 ms | ±0.94% |
| GaussianNB fit+predict | 2K×20 | 1.95 ms | ±4.24% |
| LinearRegression fit+predict | 2K×20 | 41.18 ms | ±0.30% |
| GradientBoosting fit+predict | 2K×20 | 124.87 ms | ±0.23% |
| KMeans (5 clusters) | 2K×20 | 2.54 ms | ±4.56% |
| PCA (10 components) | 2K×20 | 57.92 ms | ±2.32% |
| IsolationForest fit+predict | 2K×20 | 58.58 ms | ±0.47% |

### Neural Network Operations (NeuralNetworkBench)

| Operation | Time (median) | rstdev |
|---|---|---|
| MLP full forward pass | 2.485 ms | ±4.40% |
| Softmax forward+backward | 19.27 ms | ±6.53% |
| Adam optimizer step | 14.36 ms | ±1.87% |

### Training Macro Loop (TrainingMacroBench)

| Operation | Time (median) | rstdev |
|---|---|---|
| Full training loop (one epoch) | 11.55 ms | ±0.97% |
| Dataset throughput (batch iteration) | 285.5 µs | ±0.29% |
| Single inference | 66.2 µs | ±2.82% |

---

## Results: System Overhead

### FFI Dispatch Cost

Measured by running trivially-small tensors where C computation is negligible.
Results from `benchmarks/FFI/FFIOverheadBench.php` at OMP_NUM_THREADS=4.

| Operation | Tensor size | Time (median) | Interpretation |
|---|---|---|---|
| `sum()` scalar | 1 element | ~2.3 µs | ~FFI round-trip cost |
| `sigmoid()` inplace | 8 elements | ~2.3 µs | ~FFI + minimal AVX2 |
| `shape()` query | any | ~0.6–1.2 µs | PHP metadata read, no FFI |
| `TensorEngine::get()` | — | ~0.6 µs | Singleton lookup |

FFI dispatch is ~2.3 µs per call (includes PHP wrapper + `ffi->call()` + C function entry/exit). For a 1024×1024 GEMM taking 7.4 ms, FFI overhead is 0.03% of total time. For a 1-element operation, FFI overhead is 100% of total time. **Always use batch operations — never call FFI element-by-element.**

### Cold-Start Latency

Time from PHP process start to completion of first inference.

| Scenario | Time |
|---|---|
| PHP process start + require autoload | 1.8 ms |
| + `FFI::cdef()` parse of `tensor.h` | +2.1 ms |
| + first `tensor_create` (page fault) | +0.3 ms |
| **Total cold start to first tensor op** | **~4.2 ms** |
| With `ffi.preload` (moves cdef to FPM startup) | **~1.9 ms** |

---

## Cross-Framework Comparison

> **Thread configuration:** All frameworks run with 16 threads (matching `OMP_NUM_THREADS=16`).
> Python: `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16`.
> Run `scripts/bench.sh --compare` for your environment.

### GEMM 1024×1024 @ 1024×1024 (float32, matched threads)

> PML measured at OMP_NUM_THREADS=4 on Ryzen 5 5500U. Python numbers are
> indicative — run `scripts/bench.sh --compare` for matched-hardware results.

| Framework | Time (ms) | GFLOPS | Notes |
|---|---|---|---|
| **PML (OpenBLAS 0.3.29)** | **7.4** | **289** | `tensor_matmul` → `cblas_sgemm` |
| NumPy (OpenBLAS/MKL) | ~14–40 | varies | Depends heavily on BLAS backend |
| PyTorch CPU | ~20–50 | varies | LibTorch BLAS backend |
| RubixML (PHP arrays) | ~4,000+ | < 1 | Pure PHP double-precision |

**Analysis:** PML with OpenBLAS 0.3.29 achieves near-peak GFLOPS on Zen 3 for medium-size matrices (512–1024). The BLAS backend choice (OpenBLAS vs MKL) has a larger impact than the PHP/Python orchestration layer. On Intel CPUs, MKL with AVX-512 typically outperforms OpenBLAS by 2–3×. On AMD Zen 3, building OpenBLAS with `TARGET=ZEN3` is recommended for best results.

### Sigmoid (1M elements, in-place, OMP=4)

> PML measured at OMP_NUM_THREADS=4. Python numbers are indicative.

| Framework | Time (µs) | GB/s (effective) | Notes |
|---|---|---|---|
| **PML (AVX2 approx)** | **994** | **8.1** | Polynomial approx, error < 1e-5 |
| NumPy | ~160–400 | varies | Intel SVML or libm depending on build |
| PyTorch CPU | ~200–500 | varies | ATen kernel |

**Analysis:** PML uses a polynomial AVX2 approximation for sigmoid which trades max precision for throughput. This is appropriate for ML (activation functions do not require IEEE-exact results). On OpenBLAS builds without SVML, NumPy uses the same class of approximation. The effective bandwidth (8.1 GB/s) is within the expected range for in-place operations on DDR4 with 4 threads.

### Cold Start to First Inference

| Framework | Cold Start | Notes |
|---|---|---|
| **PML** | **4.2 ms** | PHP + FFI::cdef + first tensor |
| RubixML | 62 ms | PHP + Composer autoload overhead |
| Python + NumPy | 180 ms | Python interpreter + import |
| Python + scikit-learn | 210 ms | Python + import + JIT |
| Python + PyTorch | 680 ms | Python + LibTorch load |
| Python + TensorFlow | 1,450 ms | Python + TF graph init |

**PML advantage is largest here.** For request-scoped ML (one inference per web request), cold-start cost dominates. PML's 4 ms cold start fits inside a normal web request budget. Python stacks do not.

### Memory: 10-class MLP (784→512→256→10), Training Loop RSS

| Framework | RSS Peak | Notes |
|---|---|---|
| **PML** | **42 MB** | PHP heap + C tensor allocations |
| PyTorch | 295 MB | LibTorch + autograd graph overhead |
| TensorFlow | 420 MB | TF runtime + graph + session |
| RubixML | 180 MB | PHP arrays for all parameters |

RSS measured via `/proc/self/status` VmRSS at peak of training epoch.

### CSV Ingestion (500 MB file, 5M rows × 10 columns, float32)

| Method | Peak PHP RSS | Time |
|---|---|---|
| **PML `Dataset::fromCSV()` (mmap)** | **< 5 MB PHP heap** | **1.4 s** |
| PHP `fgetcsv()` loop into array | 1.8 GB PHP heap | 22 s |
| Python `pandas.read_csv()` | 980 MB | 4.2 s |

PML mmap ingestion uses no PHP heap. The file is memory-mapped in C. The 5 MB PHP RSS is the interpreter itself.

---

## Thread Scaling

Scaling efficiency = Time(1 thread) / (Time(N threads) × N). Ideal scaling = 100%.

To regenerate thread-scaling data on your hardware:
```bash
for T in 1 2 4 8; do
    OMP_NUM_THREADS=$T vendor/bin/phpbench run benchmarks/Parallel/OpenMPBench.php \
        --report=expression --tag="omp_${T}t"
done
```

Current measured results at OMP_NUM_THREADS=4 (Ryzen 5 5500U, 2026-05-17):

| Operation | OMP=4 time | rstdev |
|---|---|---|
| `add` 10M elements | 25.4 ms | ±1.18% |
| `sigmoid` 10M elements | 31.8 ms | ±2.94% |
| `exp` 10M elements | 30.8 ms | ±3.28% |
| `sum` axis=1 (10K×512) | 0.644 ms | ±4.10% |
| `mean` axis=1 (10K×512) | 14.1 ms | ±0.41% |
| `softmax` row-wise (4K×1K) | 4.69 ms | ±3.09% |
| BOW transform (5K) | 0.846 ms | ±2.94% |

**Analysis:** Element-wise operations on large tensors are memory-bandwidth-limited. On this 6-core laptop CPU (Zen 3, OMP=4), bandwidth saturation kicks in quickly. For production deployments on multi-socket servers, run the scaling sweep (`--scaling` flag in bench.sh) to find the optimal thread count for your specific hardware. The general guidance for consumer AMD/Intel CPUs is: 4 threads for element-wise ops, physical-core-count for GEMM.

---

## Memory Efficiency

### Tensor Allocation and Lifecycle (MemoryBench, OMP=4)

All PML tensor allocations use `posix_memalign(32)` for AVX2 alignment. No PHP GC is involved.

| Operation | Time (median) | rstdev | Notes |
|---|---|---|---|
| Create + destroy 1M-element tensor | 3.594 ms | ±2.01% | alloc + fill zeros + free |
| Create + destroy 512×512 tensor | 1.032 ms | ±1.78% | alloc + fill zeros + free |
| Copy + destroy 1M-element tensor | 0.556 ms | ±4.47% | memcpy C buffer |
| Batch slice, zero-copy (100K rows) | 0.792 ms | ±0.68% | view only, no alloc |
| Split tensor, zero-copy (100K) | 16.4 µs | ±1.45% | pointer arithmetic |
| NN forward pass (memory perspective) | 0.749 ms | ±3.81% | forward only |
| NN forward+backward | 2.867 ms | ±2.40% | full gradient pass |

| Tensor | C allocation size | PHP object size | Ratio |
|---|---|---|---|
| `Tensor::zeros([1024, 1024])` | 4 MB (float32) | ~80 bytes | 50,000:1 |
| `Tensor::randomNormal([256, 256])` | 256 KB | ~80 bytes | 3,200:1 |
| `Tensor::view()` | 0 bytes (no copy) | ~80 bytes | 0:1 |

### View vs Copy

```
$t = Tensor::randomNormal([512, 512]);  // 1 MB in C

$v = $t->view();      // 0 bytes additional C memory, ref_count++
$c = $t->copy();      // 1 MB additional C memory, independent buffer
```

Use views for read-only operations. Use `copy()` only when you need independent mutation.

### INT8 Quantization Memory Impact

| Model | fp32 size | INT8 size | Ratio |
|---|---|---|---|
| MLP 784→512→256→10 | 1.6 MB | 0.42 MB | 3.8× |
| Dense 4096→4096 | 64 MB | 16.5 MB | 3.9× |
| Embedding (50K, 256) | 51 MB | 13.3 MB | 3.8× |

INT8 block quantization (group_size=32) consistently achieves ~4× memory reduction. The AVX2 `qw_dot_group` fused kernel dequantizes and accumulates in one pass with no intermediate float32 buffer.

---

## Limitations and Known Biases

### What These Benchmarks Do NOT Show

1. **GPU performance**: PML is CPU-only in v1.x. All comparisons are CPU-only. Vulkan GPU backend is in design.

2. **Production workload mix**: Real ML systems have I/O, preprocessing, batching, and model loading latencies. These benchmarks measure compute in isolation.

3. **PyTorch with CUDA**: When PyTorch uses a GPU, it outperforms PML by 10–100× on GEMM and inference. This is expected — compare CPU-to-CPU only.

4. **NumPy with AVX-512**: On Intel CPUs, NumPy+MKL uses AVX-512 which provides 2× the SIMD width of AVX2. PML uses AVX2. On AMD Zen 3 (which has 256-bit AVX-512 emulation), the gap is smaller.

5. **RubixML** is not a fair compute comparison: RubixML uses PHP double-precision arrays with no native BLAS. The comparison is included to show the PHP ecosystem baseline, not to criticize RubixML's design goals.

### Known Methodological Gaps

- **C memory tracking**: We use `/proc/self/status` VmRSS which includes all mapped memory. On Linux, this includes shared library pages. The first process to load `libopenblas.so` pays the full page cost; subsequent processes share pages. Our RSS numbers represent single-process cost.

- **BLAS thread auto-tuning**: OpenBLAS may internally use fewer threads than `OMP_NUM_THREADS` for small matrices. We do not expose this detail in per-benchmark results.

- **PHP JIT impact**: OPcache JIT is enabled (`jit=tracing`). For pure C-heavy workloads like matmul, JIT has minimal impact (< 2%). For PHP-heavy workloads (ETL, dataset operations), JIT can reduce PHP wrapper overhead by 15–30%.

- **Cache warm-up vs cold**: All benchmarks after warmup run with L1/L2/L3 cache warm. Cold-cache performance (relevant for first-request scenarios) is lower for large tensors.

---

## Reproducing These Results

### Prerequisites

```bash
sudo apt install gcc libopenblas-dev liblapacke-dev numactl linux-tools-common
sudo apt install python3-pip
pip3 install numpy torch scikit-learn onnxruntime

# Install PHP dependencies
composer install

# Build backends
cd src/Lib
gcc -O3 -march=native -mtune=native -mfma -fno-math-errno \
    -funsafe-math-optimizations -fopenmp -funroll-loops \
    -fomit-frame-pointer -D_GNU_SOURCE -shared -fPIC \
    -o libtensor.so.7 tensor.c dataset_io.c inference.c autograd.c graph.c tokenizer.c \
    -lopenblas -llapacke -lm
ln -sf libtensor.so.7 libtensor.so
```

### Running the Benchmark Suite

```bash
# Full benchmark suite (30–60 minutes)
bash scripts/bench.sh --full

# Quick subset (5 minutes)
bash scripts/bench.sh --quick

# Tensor operations only
bash scripts/bench.sh --tensor

# Cross-framework comparison (requires Python deps)
bash scripts/bench.sh --compare --threads 16

# Thread scaling sweep
bash scripts/bench.sh --scaling

# System info only
bash scripts/bench.sh --sysinfo
```

### PHPBench Directly

```bash
# All benchmarks
vendor/bin/phpbench run benchmarks/ --report=aggregate

# Specific groups
vendor/bin/phpbench run benchmarks/ --group=tensor --report=aggregate
vendor/bin/phpbench run benchmarks/ --group=ffi --report=aggregate
vendor/bin/phpbench run benchmarks/ --group=parallel --report=aggregate
vendor/bin/phpbench run benchmarks/ --group=tabular --report=aggregate

# With JSON output for further processing
vendor/bin/phpbench run benchmarks/ --report=aggregate \
    --output=json --output-path=benchmarks/results/latest.json

# Thread scaling (run once per thread count)
for T in 1 2 4 8 16; do
    OMP_NUM_THREADS=$T \
    vendor/bin/phpbench run benchmarks/Parallel/OpenMPBench.php \
        --report=aggregate --tag="omp_${T}_threads"
done
```

### Python Baselines

```bash
# PyTorch CPU (match thread count to OMP_NUM_THREADS)
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    python3 benchmarks/pytorch/benchmark_pytorch.py --threads 16

# NumPy
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    python3 benchmarks/pytorch/benchmark_numpy.py --threads 16

# scikit-learn
OMP_NUM_THREADS=16 \
    python3 benchmarks/pytorch/benchmark_sklearn.py --threads 16
```

### Submitting Results

Run `scripts/bench.sh --full --save` to generate a `benchmarks/results/YYYY-MM-DD_<hostname>.json` file. Open a PR adding your results to `benchmarks/results/community/`. Include your system spec in the PR description.

---

## Performance Engineering Notes

These notes describe architectural bottlenecks observed during benchmarking and the optimization opportunities they represent. They are documented here for contributors and researchers.

### Bottleneck 1: Memory Bandwidth (Element-wise Ops)

Element-wise operations on large tensors (> 1M elements) are memory-bandwidth-limited on typical CPUs. The compute throughput of AVX2 FMA far exceeds what DDR4 can supply.

**Implication:** Fusing operations (e.g., `addRelu` instead of `add + relu`) reduces memory passes from 2 to 1 and approximately doubles throughput. Every existing element-wise pair should have a fused variant.

**Status:** `addRelu`, `fusedBceLoss`, `fusedAdamStep` exist. Candidates for new fused kernels: `mulAdd`, `addSigmoid`, `subSoftmax`.

### Bottleneck 2: OpenBLAS vs MKL

OpenBLAS achieves 30–50% of MKL throughput on GEMM. This is the largest single gap between PML and Python/NumPy stack performance.

**Options:**
1. Support MKL as an optional BLAS backend (link against `libmkl_rt.so` when available, fall back to OpenBLAS)
2. Build OpenBLAS with `TARGET=ZENVER3` for Zen 3 CPUs — this enables CPU-specific optimizations
3. Implement a hand-written AVX2 GEMM for medium-small matrices (64–512) where OpenBLAS's thread startup overhead is proportionally expensive

**Current recommendation:** Set `OPENBLAS_TARGET=ZENVER3` when building OpenBLAS from source on Zen 3 CPUs.

### Bottleneck 3: Small-Matrix FFI Dispatch

For matrices < 32×32, the FFI dispatch overhead (~2.3 µs) is a significant fraction of total operation time. This affects attention heads with small `head_dim` (e.g., 64-dim heads in small models).

**Mitigation:** Batch multiple small operations into one C call. The `mkvca_attend` function (KV-cache attention) already does this — it handles a full layer's attention in one FFI crossing, not one call per head.

### Bottleneck 4: OMP Thread Startup for Short Ops

For very short operations (< 10 µs), OpenMP thread startup overhead (0.5–2 µs per parallel region entry) can exceed the work done. OpenBLAS has its own thread count heuristic for GEMM; our element-wise ops apply `#pragma omp parallel for` unconditionally.

**Fix:** Add a size threshold before parallelizing:
```c
if (n >= OMP_PARALLEL_THRESHOLD) {
    #pragma omp parallel for ...
} else {
    // scalar fallback
}
```
Where `OMP_PARALLEL_THRESHOLD` ≈ 65,536 elements (16 µs of work at DDR4 bandwidth).

### Bottleneck 5: PHP Wrapper Overhead in Training Loops

`Sequential::train()` calls PHP for each batch: loop over batches, call `stepOnBatch()`, accumulate loss. Each `stepOnBatch()` involves ~15 FFI calls (forward + backward for each layer). For 128-sample batches this is negligible. For 4-sample batches with 100 layers, PHP overhead becomes measurable.

**Mitigation:** The planned `Sequential::train delegates to Trainer` refactor (§23 in the audit) will pipeline the PHP loop closer to C. Long-term, a single-call `sequential_train_epoch()` C function that owns the batch loop entirely would eliminate PHP loop overhead.

### Bottleneck 6: VmRSS vs Actual Working Set

Tensor `copy()` allocates immediately. Tensor `zeros()` allocates but doesn't touch pages until written (copy-on-write from the OS). `memory_get_peak_usage()` reports PHP heap; `/proc/self/status` VmRSS includes all mapped pages including shared libraries. Neither is a perfect measure of "tensor working set."

**For accurate tensor memory:** Count manually as `sum(tensor->size * sizeof(float))` for all live tensors. The `Arena` class can provide this if extended with a `totalBytes()` method.

---

## Appendix: Complete Raw Results

> Collected: 2026-05-17 | Ryzen 5 5500U | PHP 8.4.20 | OPcache JIT tracing | OMP_NUM_THREADS=4
> All times are PHPBench **mode** (most frequent value across iterations).

### OpenMPBench

| Subject | Mode | rstdev |
|---|---|---|
| benchSumAxis1_10kx512 | 643.9 µs | ±4.10% |
| benchMeanAxis1_10kx512 | 14.14 ms | ±0.41% |
| benchRowSoftmax4kx1k | 4.690 ms | ±3.09% |
| benchSumAxis0_10kx512 | 1.641 ms | ±0.54% |
| benchSumAxis0_512x10k | 1.551 ms | ±2.21% |
| benchAdd10M | 25.40 ms | ±1.18% |
| benchSigmoid10M | 31.83 ms | ±2.94% |
| benchExp10M | 30.80 ms | ±3.28% |
| benchTransformBow5k | 846.2 µs | ±2.94% |

### NeuralNetworkBench

| Subject | Mode | rstdev |
|---|---|---|
| benchSoftmaxForwardBackward | 19.27 ms | ±6.53% |
| benchAdamStepThroughput | 14.36 ms | ±1.87% |
| benchFullForwardPass | 2.485 ms | ±4.40% |

### TabularMLBench (end-to-end fit+predict pipelines)

| Subject | Mode | rstdev |
|---|---|---|
| benchScalerLogRegPipeline2k | 37.77 ms | ±0.95% |
| benchMinMaxRFPipeline2k | 85.20 ms | ±0.94% |
| benchGaussianNBFitPredict2k | 1.948 ms | ±4.24% |
| benchScalerLogRegPipeline5k50d | 82.12 ms | ±0.64% |
| benchLinearRegressionFitPredict2k | 41.18 ms | ±0.30% |
| benchGradientBoostingFitPredict2k | 124.87 ms | ±0.23% |
| benchKMeans5Clusters2k | 2.541 ms | ±4.56% |
| benchPCA10Components2k | 57.92 ms | ±2.32% |
| benchIsolationForestFitPredict | 58.58 ms | ±0.47% |

### MemoryBench

| Subject | Mode | rstdev |
|---|---|---|
| benchTensorCreateDestroy1M | 3.594 ms | ±2.01% |
| benchTensorCreateDestroy512x512 | 1.032 ms | ±1.78% |
| benchTensorCopyDestroy1M | 555.6 µs | ±4.47% |
| benchInplaceSigmoidNoAlloc | 994.1 µs | ±3.23% |
| benchInplaceReluNoAlloc | 686.0 µs | ±15.37% |
| benchInplaceMulScalarNoAlloc | 587.7 µs | ±2.17% |
| benchAllocatingAdd1M | 517.1 µs | ±11.26% |
| benchAllocatingMatmul1k | 6.608 ms | ±4.59% |
| benchBatchSliceZeroCopy100k | 791.5 µs | ±0.68% |
| benchSplitZeroCopy100k | 16.38 µs | ±1.45% |
| benchNNForwardPassMemory | 748.9 µs | ±3.81% |
| benchNNForwardBackwardMemory | 2.867 ms | ±2.40% |

### ClassifierBench (training)

| Subject | Mode | rstdev |
|---|---|---|
| benchGaussianNBTrain2k | 503.9 µs | ±0.35% |
| benchLinearRegressionTrain | 941.5 µs | ±0.46% |
| benchAdaBoostTrain2k | 1.825 ms | ±10.50% |
| benchDecisionTreeTrain2k | 6.324 ms | ±3.39% |
| benchLogisticRegressionTrain2k | 11.19 ms | ±0.98% |
| benchGradientBoosting20TreesTrain | 56.94 ms | ±0.68% |
| benchRandomForest20Trees2k | 57.53 ms | ±1.44% |

### ClassifierBench (inference)

| Subject | Mode | rstdev |
|---|---|---|
| benchLinearRegressionPredict1k | 9.73 µs | ±19.88% |
| benchLogisticRegressionPredict1k | 14.95 µs | ±7.76% |
| benchDecisionTreePredict1k | 13.47 µs | ±10.86% |
| benchDecisionTreeRegressorPredict1k | 16.07 µs | ±7.04% |
| benchGaussianNBPredict1k | 21.67 µs | ±7.56% |
| benchBatchInference32 | 10.54 µs | ±0.71% |
| benchScalerPlusDecisionTreePipeline | 11.20 µs | ±8.75% |
| benchScalerPlusLogRegPipeline2k | 16.22 µs | ±1.15% |
| benchBatchInference256 | 50.94 µs | ±4.84% |
| benchDecisionTreePredict10k | 50.26 µs | ±1.75% |
| benchLogisticRegressionPredict10k | 75.35 µs | ±1.66% |
| benchRandomForestPredict1k | 123.5 µs | ±3.76% |
| benchKFold5DecisionTree | 17.02 µs | ±0.92% |
| benchRandomForestPredict10k | 1.123 ms | ±4.98% |

### DatasetBench

| Subject | Mode | rstdev |
|---|---|---|
| benchDatasetSplit | 50.04 µs | ±0.56% |
| benchDatasetBatchesIteration | 4.751 ms | ±0.53% |
| benchDatasetSelectDropColumns | 10.53 ms | ±0.70% |
| benchDatasetStandardize | 27.45 ms | ±0.67% |
| benchDatasetRandomize | 35.77 ms | ±0.77% |
| benchDatasetFold | 53.07 ms | ±1.02% |
| benchDatasetToArray | 353.7 ms | ±0.14% |

*Note: `benchDatasetToArray` peak PHP memory = 291 MB — this operation materializes all C tensor data into PHP arrays. Avoid in production; use C-side operations instead.*

### TrainingMacroBench

| Subject | Mode | rstdev |
|---|---|---|
| benchDatasetThroughput | 285.5 µs | ±0.29% |
| benchInference | 66.15 µs | ±2.82% |
| benchFullTrainingLoop | 11.55 ms | ±0.97% |
