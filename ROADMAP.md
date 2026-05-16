# PML Roadmap

This document tracks the evolution of PML from a high-performance PHP ML library toward a full AI infrastructure platform. Work items are grouped into phases with concrete technical targets.

---

## Current State — v1.x (Shipped)

### Core Engine
- [x] `TensorC` with zero-copy PHP↔C memory model
- [x] 200+ tensor operations (arithmetic, reductions, linalg, shape, fused kernels)
- [x] OpenBLAS SGEMM, LAPACKE SVD/eigendecomposition
- [x] AVX2 SIMD: sigmoid, tanh, exp, INT8 dot product
- [x] OpenMP parallel batch ops, tree predictions
- [x] mmap CSV ingestion (no PHP heap)

### Classical ML
- [x] 19 classifiers, 15 regressors, 6 anomaly detectors, 5 clusterers
- [x] GBDT with histogram subtraction + PQ leaf-wise growth
- [x] SVM (kernel: RBF, poly, linear) via C SVM implementation
- [x] k-NN with KD-tree and Ball-tree acceleration
- [x] PCA, t-SNE
- [x] 35+ feature transformers (scalers, encoders, NLP, image)
- [x] Pipeline composition with 6 cross-validation strategies
- [x] GridSearch, RandomSearch, BootstrapAggregator, StackingRegressor

### Neural Networks
- [x] Sequential model with 29 layer types
- [x] 9 optimizers: SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, Nadam, LARS, LAMB
- [x] 5 loss functions: MSE, BCE, CrossEntropy, Huber, focal
- [x] BatchNorm, LayerNorm, Dropout, Noise
- [x] Conv2D, DepthwiseConv2D, InvertedResidual, SEBlock
- [x] LSTM, RNN, Mamba
- [x] CausalSelfAttention (training + prefill + decode paths)
- [x] Trainer + TrainingArguments + EarlyStopping + callbacks
- [x] GradScaler for mixed precision training

### LLM Inference
- [x] GQA (Grouped Query Attention) forward pass in C
- [x] BPE tokenizer (zero-malloc, HuggingFace tokenizer.json compatible)
- [x] SafeTensors mmap weight loader
- [x] Multi-layer KV-cache (`MultiKVCache`) — O(1) decode memory
- [x] Milakov online-softmax in `mkvca_attend`
- [x] Streaming generation (`generate()` PHP Generator)
- [x] LLaMA / Mistral / Phi architecture configs
- [x] InferenceSession API: `generate`, `chat`, `forward`, `generateIds`

### INT8 Quantization
- [x] Q8_0-class block quantization (group size 32, per-group fp32 scale)
- [x] `QuantizedWeight` C struct with AVX2 fused `qw_dot_group` kernel
- [x] `Dense::quantize()` — 4× memory reduction
- [x] `Sequential::quantize()` — whole-model quantization
- [x] `getStateDict()` dequantizes for export; `loadStateDict()` re-quantizes on load

### Vision
- [x] `libvision.so` with 106 C functions
- [x] Image I/O (stb_image): JPEG, PNG, BMP, TIFF
- [x] Resize (bilinear, bicubic, nearest, lanczos), crop, rotate, flip, pad
- [x] Color space conversion, pixel format conversion, layout conversion
- [x] Augmentation: brightness, contrast, hue, cutout, MixUp, CutMix
- [x] MobileNetV3 classifier forward pass
- [x] YOLO11n, NanoDet, PicoDet, SSDLite detection
- [x] FastSAM segmentation

### Autograd
- [x] Reverse-mode automatic differentiation
- [x] Compute graph with topological sort
- [x] Variable API: add, mul, matmul, relu, backward

---

## Phase 2.0 — Vulkan GPU Backend

**Target: Cross-vendor GPU execution without CUDA dependency.**

Vulkan runs on NVIDIA, AMD, Intel, and Apple Silicon (via MoltenVK). This phase makes GPU acceleration available to any PML deployment without requiring a proprietary runtime.

### Architecture
```
tensor_matmul(a, b)
  │
  ├─ if (a->on_gpu && b->on_gpu): vk_dispatch_gemm(a, b, out)
  │        │
  │        └─ Vulkan compute queue → SPIR-V GEMM kernel
  └─ else: cblas_sgemm(...)  ← current CPU path
```

`TensorC` gains: `VkBuffer vk_buffer`, `VkDeviceMemory vk_memory`, `bool on_gpu`, `bool dirty_cpu`.

PHP API is **unchanged**. `$tensor->gpu()` uploads; `$tensor->cpu()` downloads. Everything else is automatic.

### Kernel Priority

| Kernel | Impact | Priority |
|---|---|---|
| GEMM (fp32, fp16) | All matmul, inference | Critical |
| Linear (GEMM + bias) | Dense, attention projections | Critical |
| QLinear (INT8) | Quantized inference | Critical |
| Attention (fused QKV) | LLM decode speed | Critical |
| Softmax | Attention + classification | High |
| LayerNorm / RMSNorm | Transformer layers | High |
| FusedAdamW | Training throughput | High |
| Element-wise (relu, gelu, silu) | All models | Medium |
| Reduce (sum, max, mean) | Loss, pooling | Medium |
| Conv2d | Vision models | Medium |

### Implementation Phases

**2.0-alpha**: Foundation
- Vulkan instance, device selection, compute queue
- Buffer allocation (`vk_tensor_upload`, `vk_tensor_download`)
- SPIR-V GEMM kernel (fp32)
- `Tensor::gpu()`, `Tensor::cpu()` PHP API
- Basic test: `$c = $a->gpu()->matmul($b->gpu())->cpu()`

**2.0-beta**: LLM Inference Kernels
- fp16 tensor type
- GQA attention kernel (fused QKV project + scaled dot-product + output project)
- RoPE kernel
- SwiGLU FFN kernel
- RMSNorm kernel
- Full LLaMA-3 8B inference on GPU: target > 30 t/s on RTX 3080

**2.0-rc**: Training Kernels
- FusedAdamW
- Cross-entropy with softmax
- Backward pass kernels for Dense, LayerNorm
- Mixed-precision training (fp16 forward, fp32 accumulation)

**2.0**: Platform Breadth
- MoltenVK support (Apple Silicon)
- AMD RDNA support
- CPU↔GPU auto-migration for tensors over threshold size
- Benchmark suite vs CUDA baseline

---

## Phase 2.1 — ONNX Import + fp16

**Target: Run any ONNX model inside PHP without Python.**

- Parse ONNX protobuf (use a minimal C parser, no external protobuf dep)
- Map ONNX ops to PML tensor operations
- Support: MatMul, Conv, Relu, Sigmoid, Softmax, Reshape, Transpose, Gather, LayerNorm, Attention
- fp16 (`DTYPE_FLOAT16`) throughout the tensor engine
- Automatic upcasting for CPU path (fp16 → fp32 before computation), native fp16 for GPU path
- Flash Attention (Dao et al. 2022) implementation — O(N) memory attention

---

## Phase 2.2 — Extended Quantization

**Target: Match llama.cpp quantization quality.**

- [ ] Q4_0 (4-bit symmetric)
- [ ] Q4_K_M (4-bit with k-quant grouping)
- [ ] Q5_0, Q5_1 (5-bit)
- [ ] GGUF weight format reader (direct llama.cpp model loading)
- [ ] GPTQ-compatible weight import
- [ ] AWQ-compatible weight import
- [ ] Quantization-aware training (QAT) hooks in autograd

---

## Phase 3.0 — Agent Runtime + Tool Infrastructure

**Target: PHP-native agent loop with LLM orchestration.**

- [ ] `Agent` class: system prompt, tool registry, message history, loop driver
- [ ] Tool definition interface: `interface Tool { function call(array $args): mixed; }`
- [ ] JSON Schema tool spec generation from PHP type hints
- [ ] Parallel tool execution via `pcntl_fork` + Redis message queue
- [ ] Streaming token-level observation during agent turns
- [ ] Memory backends: in-process vector store, Redis, SQLite
- [ ] Simple RAG pipeline: `Embedder` → `VectorIndex` → `Retriever`

---

## Phase 3.1 — Distributed Training

**Target: Multi-process data-parallel training across PHP workers.**

- [ ] Gradient all-reduce via Redis pub/sub (parameter server pattern)
- [ ] Sharded dataset loading: each worker owns a dataset shard
- [ ] Checkpoint coordination: atomic writes via Redis locks
- [ ] `DistributedTrainer` class wrapping existing `Trainer`
- [ ] No changes to model code — distribution is transparent at the trainer level

---

## Phase 3.2 — JIT Tensor Compiler

**Target: Kernel fusion at runtime without manual C changes.**

- [ ] Trace PHP `Tensor` operation sequences into a compute graph IR
- [ ] Simple fusion rules: consecutive element-wise ops, add+relu, matmul+bias
- [ ] Emit fused C code at runtime, compile with `gcc -O2 -shared -fPIC`
- [ ] Cache compiled kernels by graph hash
- [ ] Fall back to sequential ops on compilation failure

This is a research-quality feature. No production timeline yet.

---

## Phase 4.0 — Edge AI Deployment

**Target: PML as a deployable AI runtime for constrained environments.**

- [ ] WASM backend via Emscripten (browser or Node.js deployment)
- [ ] Static binary mode: bundle PHP + libtensor.so into a single executable
- [ ] Model quantization to < 100 MB for edge devices
- [ ] ARM NEON SIMD path (Raspberry Pi, mobile, edge servers)
- [ ] PHP extension mode (`ext-pml`) for zero-FFI overhead in performance-critical deployments
- [ ] Android / iOS deployment guide via PHP for Android

---

## Long-Term Research Directions

These are directional, not committed:

- **Sparse tensor support**: COO/CSR sparse formats for NLP embedding layers
- **Symbolic math engine**: Differentiate arbitrary PHP expressions
- **Hardware-aware autotuning**: Profile GEMM tile sizes and choose optimal kernel at init
- **Multi-modal models**: Vision + language embedding alignment (CLIP-style)
- **Federated learning**: Privacy-preserving training with differential privacy noise
- **Custom accelerator backends**: NPU (Intel OpenVINO, Qualcomm QNN) via a plugin interface

---

## Version Policy

| Component | Versioning | Breaking changes |
|---|---|---|
| PHP API (`src/`) | Semantic (major.minor.patch) | Only in major versions |
| C ABI (`tensor.h`) | Versioned `.so` (libtensor.so.N) | New `.so` version |
| PHP FFI bindings | Follows C ABI version | With C ABI |
| CLI (`bin/automl`) | Semantic | Only in major versions |

---

## How to Influence the Roadmap

- Open a [Discussion](https://github.com/ghostjat/pml/discussions) with the tag `roadmap`
- Submit an RFC issue with the label `rfc`
- Sponsor a specific feature via [GitHub Sponsors](https://github.com/sponsors/ghostjat) with a note

Priority is given to items that have community demand, existing sponsor backing, or strategic infrastructure value.
