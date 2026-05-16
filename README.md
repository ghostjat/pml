<div align="center">

# PML — PHP Machine Learning

**A production-grade CPU-first AI runtime and machine learning infrastructure framework for PHP.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PHP](https://img.shields.io/badge/PHP-%3E%3D8.1-8892BF.svg)](https://php.net)
[![Build](https://img.shields.io/github/actions/workflow/status/ghostjat/pml/ci.yml?label=CI)](https://github.com/ghostjat/pml/actions)
[![Packagist](https://img.shields.io/packagist/v/ghostjat/pml)](https://packagist.org/packages/ghostjat/pml)
[![Stars](https://img.shields.io/github/stars/ghostjat/pml?style=social)](https://github.com/ghostjat/pml/stargazers)
[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-ea4aaa.svg)](https://github.com/sponsors/ghostjat)

</div>

---

> PML is to PHP what llama.cpp is to C++ — a high-performance native runtime that brings serious AI computation into an ecosystem the rest of the industry ignores.

---

## What is PML?

PML is a **native-accelerated machine learning and AI inference runtime** built for PHP. It combines a hand-optimized C tensor engine with a clean PHP orchestration layer, delivering production-grade ML without Python, without CUDA, and without sacrificing throughput.

The architecture is built on a single philosophy: **PHP orchestrates, C computes**.

```
Your PHP Application
        │
        ▼
   Pml\Tensor / Pml\Dataset      ← zero-copy PHP wrappers
        │
        ▼  PHP FFI (single boundary crossing per op)
   libtensor.so                  ← C tensor engine
        │
        ├── OpenBLAS              ← BLAS / LAPACK kernels
        ├── LAPACKE               ← eigendecomposition, SVD
        ├── OpenMP                ← multi-threaded batch ops
        └── AVX2                  ← SIMD acceleration
```

Every tensor lives as a `TensorC*` in C memory. PHP holds a reference pointer — never a copy. There are no PHP arrays in any hot path.

---

## Why PML Exists

Modern ML stacks assume Python. This assumption carries hidden costs in PHP-first environments:

| Pain Point | Python Stack | PML |
|---|---|---|
| Cold-start overhead | 200–800 ms (interpreter + runtime imports) | < 5 ms (PHP + FFI) |
| Memory per inference | 150–400 MB baseline | 8–20 MB baseline |
| Deployment surface | Python runtime + venv + pip | PHP + one `.so` file |
| PHP integration | IPC, REST, or subprocess | Native function call |
| CPU parallelism | GIL-constrained | OpenMP, zero-GIL |

If you run PHP backends, PML lets you **embed ML directly** — same process, same memory space, same request lifecycle.

---

## Technical Highlights

### Zero-Copy Tensor Architecture

```php
// CSV loaded via mmap into C memory — no PHP arrays
$ds = Dataset::fromCSV('/data/train.csv');

// Tensor wraps TensorC* — no PHP-side copy
$X  = $ds->samples();  // Pml\Tensor → TensorC* view

// All math crosses FFI exactly once per operation
$out = $X->matmul($W)->add($b)->relu();
```

`Tensor` is a thin PHP object holding a `\FFI\CData` pointer. Slices, views, and column extractions reuse the same memory buffer with reference counts tracked entirely inside C.

### Native C Tensor Engine

`libtensor.so` provides:

- **500+ exported C functions** across tensor ops, dataset I/O, inference, autograd, graph execution, and tokenization
- **Fused kernels**: `addRelu`, `fusedAdamStep`, `fusedBceLoss`, `qw_dot_group` (INT8 + fp32 scale)
- **AVX2 SIMD** sigmoid, tanh, exp, INT8 dot product
- **OpenBLAS SGEMM** for all matmul on contiguous float32 tensors
- **OpenMP** threaded batch operations, tree predictions, image pipelines
- **mmap CSV loader**: ingests multi-GB datasets without touching PHP memory

### LLM Inference Engine

```php
$tok     = Tokenizer::fromJson('/models/llama3-8b/tokenizer.json');
$session = InferenceSession::load('/models/llama3-8b', tok: $tok);

// GQA forward pass, KV-cache, streaming tokens
foreach ($session->generate("Explain AVX2:", maxNewTokens: 200) as $token) {
    echo $token;
}
```

- LLaMA / Mistral / Phi architecture support
- **GQA** (Grouped Query Attention) natively in C
- **Multi-layer KV-cache** (`MultiKVCache`) — eliminates O(T²) decode cost
- Milakov online-softmax: O(head_dim) working memory
- SafeTensors mmap weight loading — zero-copy model ingestion
- **INT8 block quantization** (Q8_0-class): 4× memory reduction, AVX2 fused kernel

### Classical ML at Native Speed

```php
$pipeline = new Pipeline(
    transformers: [new StandardScaler(), new PolynomialExpander(degree: 2)],
    estimator:    new GBDTClassifier(trees: 500, maxDepth: 6)
);

$pipeline->train($dataset);
echo $pipeline->score($test);  // accuracy, AUC, F1
```

GBDT with histogram subtraction + PQ leaf-wise growth. All split-finding runs in C.

---

## Feature Matrix

| Module | Description |
|---|---|
| **Tensor** | 200+ ops: creation, arithmetic, linear algebra, shape, reductions, fused kernels |
| **Dataset** | Zero-copy mmap CSV, ETL/DataFrame mode, stratified splits, DataLoader, streaming |
| **Estimators** | 19 classifiers, 15 regressors, 6 anomaly detectors, 5 clusterers, decomposition |
| **Transformers** | Scalers, encoders, NLP vectorizers, image transforms, feature selection, imputers |
| **Neural Networks** | 29 layer types, 9 optimizers, 5 losses, early stopping, callbacks, mixed precision |
| **Quantization** | INT8 block quantization, QuantizedTensor, Dense::quantize(), Sequential::quantize() |
| **Inference** | LLM forward pass, GQA, KV-cache, BPE tokenizer, SafeTensors I/O, streaming |
| **Vision** | 106 C functions: image I/O, augmentation, MobileNetV3, YOLO11n, NanoDet, FastSAM |
| **Pipeline** | Transformer composition, 6 CV strategies, GridSearch, ensemble, BootstrapAggregator |
| **Autograd** | Reverse-mode AD, compute graph, Variable API |

---

## Installation

### Requirements

| Dependency | Version | Purpose |
|---|---|---|
| PHP | ≥ 8.1 | Runtime |
| ext-ffi | any | C bridge |
| GCC | ≥ 11 | Compile backend |
| libopenblas-dev | any | BLAS kernels |
| liblapacke-dev | any | Linear algebra |
| Linux x86_64 | — | AVX2 / OpenMP |

```bash
# Ubuntu / Debian
sudo apt install gcc libopenblas-dev liblapacke-dev

# Install PHP library
composer require ghostjat/pml

# Build the C backend (once per machine)
cd vendor/ghostjat/pml/src/Lib
gcc -O3 -march=native -mfma -fopenmp -funroll-loops -fomit-frame-pointer \
    -D_GNU_SOURCE -shared -fPIC -funsafe-math-optimizations \
    -o libtensor.so.7 tensor.c dataset_io.c inference.c autograd.c graph.c tokenizer.c \
    -lopenblas -llapacke -lm
ln -sf libtensor.so.7 libtensor.so
```

**`php.ini` settings:**

```ini
ffi.enable        = true
memory_limit      = 2G
opcache.jit       = tracing
opcache.jit_buffer_size = 128M
```

---

## Quick Start

### Classical Classification

```php
<?php
require 'vendor/autoload.php';

use Pml\Dataset;
use Pml\Pipeline;
use Pml\Transformers\StandardScaler;
use Pml\Estimators\Classifiers\RandomForestClassifier;

$dataset = Dataset::fromCSV('iris.csv', hasHeader: true)
    ->withLabelColumn('species')
    ->dropNans();

[$train, $test] = $dataset->stratifiedSplit(testRatio: 0.2);

$pipeline = new Pipeline(
    transformers: [new StandardScaler()],
    estimator:    new RandomForestClassifier(trees: 200)
);

$pipeline->train($train);
echo "Accuracy: " . $pipeline->score($test) . PHP_EOL;
$pipeline->save('/models/iris');
```

### Deep Learning (MLP with early stopping)

```php
<?php
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\{Dense, BatchNormalization, Dropout, ReLU, Softmax};
use Pml\NeuralNetwork\Optimizers\Adam;
use Pml\NeuralNetwork\Losses\CrossEntropyLoss;
use Pml\Training\{Trainer, TrainingArguments};

$model = new Sequential([
    new Dense(784, 512), new BatchNormalization(), new ReLU(), new Dropout(0.3),
    new Dense(512, 256), new BatchNormalization(), new ReLU(), new Dropout(0.2),
    new Dense(256, 10),  new Softmax(),
], new Adam(lr: 1e-3), new CrossEntropyLoss());

$trainer = new Trainer($model, new TrainingArguments(
    epochs: 30, batchSize: 128, patience: 5,
));

$result = $trainer->train($trainDataset, $valDataset);
echo "Best accuracy: {$result->bestMetric}" . PHP_EOL;
```

### INT8 Quantized Deployment

```php
<?php
// Quantize after training — 4× memory reduction, same API
$model->quantize(groupSize: 32);
$predictions = $model->predict($testDataset);
```

### LLM Inference (LLaMA / Mistral)

```php
<?php
use Pml\Inference\{InferenceSession, Tokenizer};

$tok     = Tokenizer::fromJson('/models/mistral-7b/tokenizer.json');
$session = InferenceSession::load('/models/mistral-7b', tok: $tok);

foreach ($session->generate("Write a PHP FFI binding:", maxNewTokens: 300) as $token) {
    echo $token;
    flush();
}
```

### Computer Vision

```php
<?php
use Pml\Vision\{Image, Yolo11n, MobileNetV3};

$detector   = new Yolo11n('/models/yolo11n.weights', confidenceThresh: 0.5);
$classifier = new MobileNetV3('/models/mobilenetv3.weights');

$img  = Image::fromFile('scene.jpg');
$dets = $detector->detect($img);

foreach ($dets as $box) {
    $label = $classifier->classify($img->crop(...$box->rect));
    echo "{$label} @ {$box->confidence}" . PHP_EOL;
}
```

---

## Benchmarks

> Benchmarks run on AMD Ryzen 9 5950X, 64 GB DDR4-3600, Ubuntu 22.04, GCC 13, PHP 8.3.
> Full methodology in [BENCHMARKS.md](BENCHMARKS.md).

### Tensor Throughput — GEMM 1024×1024

| Runtime | Time | GFLOPS |
|---|---|---|
| **PML (OpenBLAS + AVX2)** | **18 ms** | **116** |
| RubixML (PHP arrays) | 4,200 ms | 0.5 |
| NumPy (MKL) | 14 ms | 150 |
| PyTorch CPU | 22 ms | 95 |

### Cold-Start to First Inference

| Runtime | Cold Start |
|---|---|
| **PML** | **4 ms** |
| Python + scikit-learn | 210 ms |
| Python + PyTorch | 680 ms |

### Memory: 10-class MLP Training (50K samples)

| Runtime | RSS Peak |
|---|---|
| **PML** | **38 MB** |
| PyTorch | 290 MB |
| TensorFlow | 410 MB |

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design document.

```
┌─────────────────────────────────────────────────────────┐
│                   Your PHP Application                   │
└─────────────────────────────┬───────────────────────────┘
                              │  PSR-4 autoload
┌─────────────────────────────▼───────────────────────────┐
│                     PML PHP Layer                        │
│  Tensor · Dataset · Pipeline · Sequential ·              │
│  InferenceSession · Vision · Estimators · Transformers   │
└─────────────────────────────┬───────────────────────────┘
                              │  FFI::cdef() — one crossing per op
┌─────────────────────────────▼───────────────────────────┐
│              libtensor.so  (C tensor engine)             │
│  tensor.c · dataset_io.c · inference.c · autograd.c      │
│  graph.c · tokenizer.c                                   │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐   │
│  │  OpenBLAS    │  │  LAPACKE    │  │   OpenMP     │   │
│  └──────────────┘  └─────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Roadmap Preview

| Version | Focus | Status |
|---|---|---|
| v1.0–1.3 | Tensor engine, classical ML, deep learning, LLM inference, INT8, vision | ✅ Done |
| v2.0 | Vulkan GPU backend (cross-vendor: NVIDIA / AMD / Intel / Apple) | 🔄 Design |
| v2.1 | ONNX model import, fp16 tensors, Flash Attention | 📋 Planned |
| v3.0 | Distributed training, sharded datasets, agent runtime | 📋 Planned |

Full roadmap: [ROADMAP.md](ROADMAP.md)

---

## Comparison

| | PML | scikit-learn | PyTorch CPU | RubixML |
|---|---|---|---|---|
| Language | PHP + C | Python + C | Python + C++ | PHP |
| Tensor engine | Native C (libtensor.so) | NumPy | LibTorch | PHP arrays |
| Zero-copy I/O | ✅ mmap | ✗ | ✗ | ✗ |
| PHP-native API | ✅ | ✗ | ✗ | ✅ |
| LLM inference | ✅ GQA, KV-cache | ✗ | ✗ | ✗ |
| INT8 quantization | ✅ AVX2 fused | ✗ | ✅ | ✗ |
| Vision (detection) | ✅ YOLO11n, NanoDet | ✗ | ✗ | ✗ |
| Cold-start | 4 ms | 210 ms | 680 ms | 60 ms |
| Deployment | `.so` file | Python env | Python env | Composer |

---

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. Key rules:

- PHP orchestrates — heavy loops must stay in C
- Preserve zero-copy semantics everywhere possible
- New C functions must be declared in `tensor.h` and bound in `TensorEngine.php`
- All PRs require PHPUnit + PHPBench results
- Performance regressions block merge

```bash
composer install
vendor/bin/phpunit --colors=always
vendor/bin/phpbench run --report=aggregate
```

---

## Sponsors

PML is an independent open-source project. Sponsorship funds C kernel development, GPU backend work, documentation, and infrastructure.

**[❤ Become a Sponsor](https://github.com/sponsors/ghostjat)**

See [SPONSORS.md](SPONSORS.md) for tier details and benefits.

---

## License

[MIT](LICENSE) — Copyright (c) 2024 Shubham Chaudhary

---

<div align="center">

*PHP orchestrates. C computes. Zero compromises.*

</div>
