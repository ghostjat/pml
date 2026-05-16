# PML Architecture

## Design Philosophy

PML is built on four invariants that must hold at every layer:

1. **PHP orchestrates, C computes.** No ML math in PHP. PHP is the API surface; the C backend is the compute surface.
2. **Zero-copy PHP↔C data flow.** `Tensor` holds a `TensorC*` pointer. PHP never duplicates the underlying buffer.
3. **Single FFI boundary crossing per operation.** No per-element FFI calls. Every operator dispatches exactly one C function.
4. **Explicit memory ownership.** C allocates, C frees. PHP destructors call `tensor_free()`. Reference counting is managed in C.

---

## Layer Map

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Application                          │
│                (your PHP code, controllers, CLI)                 │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│                    PML PHP API Layer  (src/)                      │
│                                                                  │
│  Tensor.php          ─── wraps TensorC*                         │
│  Dataset.php         ─── wraps DataFrame* / TensorC*            │
│  Pipeline.php        ─── Transformer[] → Learner                │
│  Sequential.php      ─── layer stack, forward/backward          │
│  InferenceSession.php─── LLM session, KV-cache lifecycle        │
│  KVCache.php         ─── wraps MultiKVCache*                    │
│  QuantizedTensor.php ─── wraps QuantizedWeight*                 │
│  Estimators/         ─── classifiers, regressors, clusterers    │
│  Transformers/       ─── scalers, encoders, vectorizers         │
│  Vision/             ─── image, detection, segmentation         │
│  Training/           ─── Trainer, TrainingArguments, callbacks  │
│  CrossValidation/    ─── KFold, StratifiedKFold, MonteCarlo     │
│  Autograd/           ─── Variable, reverse-mode AD              │
└──────────────────────────────┬───────────────────────────────────┘
                               │  PHP FFI::cdef() — one crossing per op
┌──────────────────────────────▼───────────────────────────────────┐
│                  FFI Bridge  (src/Lib/TensorEngine.php)          │
│                                                                  │
│  TensorEngine::get() → TensorFFI singleton                      │
│  Declares all TensorC, DataFrame, QuantizedWeight,              │
│  MultiKVCache, KVCache, SafeTensorsMap structs via cdef()        │
└──────────────────────────────┬───────────────────────────────────┘
                               │  dlopen()
┌──────────────────────────────▼───────────────────────────────────┐
│                libtensor.so  (src/Lib/*.c)                        │
│                                                                  │
│  tensor.c      — TensorC CRUD, arithmetic, fused kernels        │
│  dataset_io.c  — mmap CSV, DataFrame, ETL ops                   │
│  inference.c   — GQA forward pass, KV-cache, attention          │
│  autograd.c    — Variable, backward pass, gradient accumulation │
│  graph.c       — compute graph, topological sort, eval          │
│  tokenizer.c   — BPE tokenizer, encode/decode                   │
│                                                                  │
│  ┌────────────┐  ┌───────────┐  ┌──────────┐  ┌────────────┐  │
│  │  OpenBLAS  │  │  LAPACKE  │  │  OpenMP  │  │    AVX2    │  │
│  │   SGEMM    │  │  SVD/eig  │  │ threads  │  │   SIMD     │  │
│  └────────────┘  └───────────┘  └──────────┘  └────────────┘  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                libquant.so  (src/Lib/quant.c)                    │
│  INT8 block quantization, QuantizedWeight, qw_dot_group AVX2    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│              libvision.so  (src/Lib/vision/)                     │
│  Image I/O (stb_image), resize, crop, augment, color spaces,   │
│  MobileNetV3 CNN forward, YOLO11n, NanoDet, PicoDet, FastSAM   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Core Data Structures

### TensorC (C)

```c
typedef struct TensorC {
    float*   data;         // raw float32 buffer (owned by C)
    int*     shape;        // dimension sizes [d0, d1, ..., dn-1]
    int*     strides;      // byte strides (enables non-contiguous views)
    int      ndim;         // number of dimensions
    int      size;         // total element count
    int      ref_count;    // shared ownership count
    DType    dtype;        // DTYPE_FLOAT32, DTYPE_INT8, DTYPE_INT64, etc.
    bool     owns_data;    // false for views/slices
} TensorC;
```

Views (`tensor_view`, `tensor_row`, `tensor_slice`) set `owns_data = false` and increment `ref_count`. `tensor_free` decrements the ref count and only frees `data` when it reaches zero.

### QuantizedWeight (C)

```c
typedef struct QuantizedWeight {
    int8_t*  data;         // INT8 quantized weights [rows × cols]
    float*   scales;       // per-group fp32 scales  [rows × num_groups]
    int      rows, cols;
    int      group_size;   // typically 32 (Q8_0-class)
    int      num_groups;   // cols / group_size
} QuantizedWeight;
```

Symmetric INT8 quantization. Scale is per group of 32 columns per row. The fused `qw_dot_group` AVX2 kernel dequantizes and accumulates in a single pass.

### MultiKVCache (C)

```c
typedef struct MultiKVCache {
    void**   caches;       // KVCache*[nLayers * nHeads]
    int      nLayers;
    int      nHeads;
    int      maxSeqLen;
    int      headDim;
    int      seqLen;       // current fill level
} MultiKVCache;
```

`mkvca_prefill` processes the prompt in one shot. `mkvca_append` extends by 1 token. `mkvca_attend` runs Milakov online-softmax with O(head_dim) working memory.

---

## Memory Model

### Allocation

All tensor memory is allocated inside `libtensor.so` via `malloc` / `posix_memalign` (32-byte aligned for AVX2). PHP never calls `malloc` for tensor data.

### Lifetime

```
tensor_create()  →  ref_count = 1
tensor_view()    →  ref_count++, owns_data = false
tensor_free()    →  ref_count--
                     if ref_count == 0 && owns_data: free(data)
                     always: free(shape), free(strides), free(TensorC)
```

PHP `__destruct()` calls `tensor_free()`. No GC involvement in the hot path.

### Zero-Copy Paths

| Operation | PHP side | C side |
|---|---|---|
| `Dataset::fromCSV()` | stores `DataFrame*` pointer | mmap CSV, build column buffers |
| `$ds->samples()` | creates `Tensor` with existing pointer | `tensor_view` of DataFrame buffer |
| `$t->row(i)` | creates `Tensor` with offset pointer | `tensor_view` with stride offset |
| `$t->slice(a, b)` | creates `Tensor` with pointer + shape | `tensor_view` with new shape |
| SafeTensors load | stores mmap pointer | mmap file, parse header |

### Known Non-Zero-Copy Paths

- `Tensor::toFlatArray()` — converts C buffer to PHP array (intentionally, for user output)
- `Tensor::fromArray()` — copies PHP array into C buffer (intentionally, for user input)
- `Dataset::materialize()` — ETL→Tensor mode conversion copies to contiguous tensor layout

---

## FFI Bridge Design

`TensorEngine.php` is a singleton that calls `FFI::cdef()` once at first access:

```php
final class TensorEngine {
    private static ?TensorFFI $instance = null;

    public static function get(): TensorFFI {
        if (self::$instance === null) {
            self::$instance = new TensorFFI(\FFI::cdef(
                file_get_contents(__DIR__ . '/tensor.h'),
                __DIR__ . '/libtensor.so'
            ));
        }
        return self::$instance;
    }
}
```

`TensorFFI` is a typed wrapper class around `\FFI` (not a subclass of `\FFI`). All PHP callers obtain the singleton via `TensorEngine::get()`. IDE-level type checking is handled by stubs in `stubs/`.

### Why Not Preload?

FFI preload (`ffi.preload`) is supported and recommended for production deployments. It moves the `cdef()` parse from request time into PHP-FPM startup, reducing first-request latency to near-zero.

---

## C Backend Modules

### tensor.c

The core module. Provides:

- Tensor lifecycle: `tensor_create`, `tensor_create_dtype`, `tensor_view`, `tensor_free`, `tensor_copy`
- Arithmetic: `tensor_add`, `tensor_sub`, `tensor_mul`, `tensor_div`, `tensor_pow`, scalar variants
- In-place: `*_inplace` variants for all arithmetic ops
- Element-wise: `tensor_exp`, `tensor_log`, `tensor_sqrt`, `tensor_abs`, activation functions
- Reductions: `tensor_sum`, `tensor_mean`, `tensor_max`, `tensor_argmax`, `tensor_std`, `tensor_sum_axis`
- Linear algebra: `tensor_matmul`, `tensor_linear`, `tensor_svd`, `tensor_ridge_solve`
- Shape: `tensor_reshape`, `tensor_transpose`, `tensor_concat`, `tensor_stack`, `tensor_pad`
- Fused: `tensor_fused_adam_step`, `tensor_fused_adamw_step`, `tensor_add_relu`, `tensor_fused_bce_loss`
- EDA stats: `tensor_col_stats`, `tensor_correlation_matrix`, `tensor_mutual_info_cols`, `tensor_spearman_cols`
- I/O: `tensor_save`, `tensor_load`, `tensor_save_safetensors`, `tensor_load_safetensors`

### dataset_io.c

- `tensor_dataset_from_csv`: mmap-backed CSV ingestion with type inference
- `dataframe_*`: 40+ DataFrame operations (select, drop, join, groupBy, sort, encode)
- `dataframe_one_hot_encode`, `dataframe_target_encode_fit/transform`, `dataframe_freq_encode_fit/transform`

### inference.c

- `inference_forward`: full GQA Transformer forward pass (pre-norm, RoPE, GQA attention, SwiGLU FFN)
- `mkvca_create/free/reset/prefill/append/attend`: multi-layer KV-cache management
- `attention_forward_causal`: causal multi-head attention with masking
- `rope_embed`: RoPE positional embedding

### autograd.c

- `ag_variable_create/free`: Variable lifecycle
- `ag_add/mul/matmul/relu/backward`: differentiable ops
- Gradient accumulation with topological sort in `graph.c`

### tokenizer.c

- `tokenizer_load_json`: parse HuggingFace tokenizer.json (BPE)
- `tokenizer_encode/decode/encode_batch`: BPE encode/decode, returns `int64_t[]`
- Zero-malloc design: pre-allocated merge buffer, no per-token allocation

---

## Build System

### libtensor.so (main backend)

```bash
cd src/Lib
gcc -O3 -march=native -mtune=native -mfma -fno-math-errno \
    -funsafe-math-optimizations -fopenmp -funroll-loops \
    -fomit-frame-pointer -D_GNU_SOURCE -shared -fPIC \
    -o libtensor.so.7 tensor.c dataset_io.c inference.c autograd.c graph.c tokenizer.c \
    -lopenblas -llapacke -lm
ln -sf libtensor.so.7 libtensor.so
```

### libquant.so (INT8 quantization)

Compiled separately to keep the quantization surface isolated.

```bash
gcc -O3 -march=native -mfma -funsafe-math-optimizations \
    -fopenmp -funroll-loops -fomit-frame-pointer -D_GNU_SOURCE \
    -shared -fPIC -o libquant.so.1 quant.c -lopenblas -lm
ln -sf libquant.so.1 libquant.so
```

### libvision.so (vision backend)

```bash
gcc -O3 -march=native -mfma -funsafe-math-optimizations \
    -fopenmp -funroll-loops -D_GNU_SOURCE -shared -fPIC \
    -o libvision.so.1 vision/*.c -lopenblas -lm -ljpeg -lpng
ln -sf libvision.so.1 libvision.so
```

---

## Namespace Map

| Namespace | Location | Purpose |
|---|---|---|
| `Pml\` | `src/` | Root namespace |
| `Pml\Lib\` | `src/Lib/` | FFI bridge, C helpers |
| `Pml\NeuralNetwork\` | `src/NeuralNetwork/` | Layers, optimizers, losses |
| `Pml\Inference\` | `src/Inference/` | LLM inference session |
| `Pml\Vision\` | `src/Vision/` | Computer vision |
| `Pml\Estimators\` | `src/Estimators/` | Classical ML models |
| `Pml\Transformers\` | `src/Transformers/` | Feature transformers |
| `Pml\CrossValidation\` | `src/CrossValidation/` | CV strategies |
| `Pml\Training\` | `src/Training/` | Trainer, callbacks, LR |
| `Pml\Autograd\` | `src/Autograd/` | Automatic differentiation |
| `Pml\SLM\` | `src/SLM/` | Small LM utilities |

---

## Interface Hierarchy

```
Estimator
  └── Learner
        ├── Probabilistic
        ├── Scoring
        ├── Online
        └── RanksFeatures
Transformer
  └── FitTransformable
Persistable
  └── (Learner, Transformer implementations)
Verbose (PSR-3 logger injection)
Stateful (reset/warm state)
TrainableWithOptions (Sequential::train options)
Quantizable (quantize/isQuantized)
```

---

## Vulkan GPU Roadmap

The GPU backend is designed as an additive layer — **no PHP API changes required**.

```
Tensor::matmul($w)
  │
  └─ tensor_matmul(a, b)   ← same C function
       │
       ├─ if (a->on_gpu && b->on_gpu): vk_dispatch_gemm(a, b)
       └─ else: cblas_sgemm(...)     ← current path
```

`TensorC` will gain three fields: `vk_buffer`, `vk_memory`, `on_gpu`. Upload/download will be lazy: tensors move to GPU on first GPU-bound op, move back on `toFlatArray()` or save. GLSL compute shaders are compiled to SPIR-V at build time. See [ROADMAP.md](ROADMAP.md) for the full GPU implementation plan.
