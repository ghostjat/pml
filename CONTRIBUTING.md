# Contributing to PML

Thank you for your interest in contributing. PML is a systems-level project — contributions are welcome at every layer: C kernels, PHP API, tests, benchmarks, documentation, and examples.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Architecture Overview](#architecture-overview)
- [Contribution Types](#contribution-types)
- [C Development Guidelines](#c-development-guidelines)
- [PHP Development Guidelines](#php-development-guidelines)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Pull Request Process](#pull-request-process)
- [Performance Contract](#performance-contract)

---

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be professional, constructive, and respectful.

---

## Development Setup

### Prerequisites

```bash
# Ubuntu / Debian
sudo apt install gcc g++ make libopenblas-dev liblapacke-dev php8.2-cli php8.2-ffi

# Verify PHP FFI
php -m | grep FFI

# Verify OpenBLAS
dpkg -l libopenblas-dev
```

### Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/pml.git
cd pml
composer install
```

### Build the C Backend

```bash
cd src/Lib

# Main backend
gcc -O3 -march=native -mtune=native -mfma -fno-math-errno \
    -funsafe-math-optimizations -fopenmp -funroll-loops \
    -fomit-frame-pointer -D_GNU_SOURCE -shared -fPIC \
    -o libtensor.so.7 tensor.c dataset_io.c inference.c autograd.c graph.c tokenizer.c \
    -lopenblas -llapacke -lm
ln -sf libtensor.so.7 libtensor.so

# Quantization backend
gcc -O3 -march=native -mfma -funsafe-math-optimizations \
    -fopenmp -funroll-loops -fomit-frame-pointer -D_GNU_SOURCE \
    -shared -fPIC -o libquant.so.1 quant.c -lopenblas -lm
ln -sf libquant.so.1 libquant.so

# Verify exports
nm -D libtensor.so | grep tensor_
nm -D libquant.so  | grep quant_
```

### PHP Configuration

```ini
; php.ini (or php-cli.ini)
ffi.enable = true
memory_limit = 2G
opcache.jit = tracing
opcache.jit_buffer_size = 128M
```

### Run Tests

```bash
vendor/bin/phpunit --colors=always
```

---

## Architecture Overview

Read [ARCHITECTURE.md](ARCHITECTURE.md) before contributing. The key invariants:

1. **PHP orchestrates, C computes.** Do not add ML math in PHP.
2. **Zero-copy.** PHP holds `TensorC*` pointers. Do not copy tensor data into PHP arrays in hot paths.
3. **One FFI crossing per op.** Do not call FFI in a loop. Write a batch C function instead.
4. **C owns memory.** PHP destructors call `tensor_free()`. Never `free()` from PHP.

---

## Contribution Types

### C Kernel Contributions (highest impact)

- New tensor operations in `tensor.c`
- Fused kernels (combine two+ ops into one pass)
- AVX2 SIMD optimizations
- OpenMP parallelization of batch loops
- New model architectures in `inference.c`
- Dataset I/O improvements in `dataset_io.c`

**Requirements:**
- Add function declaration to `tensor.h`
- Add PHP binding in `TensorEngine.php` (cdef block + method)
- Add PHPUnit test
- Add PHPBench benchmark
- Document in `docs/`
- No memory leaks (run with `valgrind --leak-check=full`)

### PHP API Contributions

- New estimator implementations
- New transformer implementations
- Cross-validation strategies
- Training callbacks
- Serialization / persistence improvements

**Requirements:**
- Implement the relevant interface (`Learner`, `Transformer`, `Persistable`, etc.)
- Delegate all numeric work to existing C functions via `TensorEngine::get()`
- Add PHPUnit test with real data
- Follow existing naming conventions

### Documentation

- Corrections or improvements to `docs/*.html`
- New examples in `example/`
- Benchmark reproductions
- README improvements

---

## C Development Guidelines

### Memory Rules

```c
// CORRECT: allocate inside C, return pointer
TensorC* my_op(TensorC* a) {
    TensorC* out = tensor_create(a->ndim, a->shape);
    // ... fill out->data ...
    return out;  // caller owns, must call tensor_free()
}

// CORRECT: view (no new allocation)
TensorC* my_view(TensorC* a, int* shape, int ndim) {
    TensorC* v = tensor_view(a, shape, ndim);
    return v;  // ref_count of a->data incremented
}

// WRONG: allocate with malloc, return raw pointer without TensorC wrapper
float* bad_op(TensorC* a) { ... }
```

### Error Handling

```c
// Use assertions for invariants
assert(a->ndim == 2 && "matmul requires 2D tensor");

// Return NULL for runtime failures (bad input)
if (a->shape[1] != b->shape[0]) {
    fprintf(stderr, "[tensor] matmul: shape mismatch\n");
    return NULL;
}
```

### AVX2 Kernels

```c
#ifdef __AVX2__
#include <immintrin.h>

void my_avx2_kernel(float* dst, const float* src, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        // ... compute ...
        _mm256_storeu_ps(dst + i, result);
    }
    // scalar tail
    for (; i < n; i++) dst[i] = scalar_fn(src[i]);
}
#else
// fallback for non-AVX2 builds
void my_avx2_kernel(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] = scalar_fn(src[i]);
}
#endif
```

### OpenMP

```c
// Parallel outer loop, sequential inner
#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
for (int i = 0; i < batch_size; i++) {
    compute_row(out->data + i * cols, in->data + i * cols, cols);
}
```

Do not nest `omp parallel` inside another parallel region.

### Naming Conventions

| Prefix | Scope |
|---|---|
| `tensor_` | TensorC operations |
| `dataframe_` | DataFrame / ETL operations |
| `inference_` | LLM forward pass |
| `mkvca_` | Multi-layer KV-cache |
| `ag_` | Autograd |
| `qweight_` | QuantizedWeight / INT8 |
| `tok_` | Tokenizer |

---

## PHP Development Guidelines

### Interface Compliance

Every estimator must implement at minimum `Estimator` and `Learner`. Check `src/Interfaces/` for the full list. Do not add methods to PHP classes that duplicate what C already does.

### FFI Calls

```php
// CORRECT: get singleton once, store in local
$ffi = TensorEngine::get();
$out = $ffi->tensor_my_op($this->ptr, $other->ptr);
return new Tensor($out);

// WRONG: call TensorEngine::get() inside a loop
for ($i = 0; $i < $n; $i++) {
    TensorEngine::get()->tensor_my_op(...); // unnecessary repeated lookup
}
```

### Null Safety

PHP C pointers returned from FFI can be `null` on allocation failure. Check:

```php
$out = $ffi->tensor_my_op($this->ptr);
if ($out === null) {
    throw new \RuntimeException('tensor_my_op: C allocation failed');
}
```

### No PHP Numeric Loops

```php
// WRONG: PHP-level row loop
for ($i = 0; $i < $n; $i++) {
    $sum += $tensor->get($i, 0);
}

// CORRECT: delegate to C
$sum = $tensor->sum()->toScalar();
```

### Naming

- Classes: `PascalCase`
- Methods: `camelCase`
- Constants: `UPPER_SNAKE_CASE`
- Private fields: `$camelCase` (no underscore prefix)

---

## Testing

### PHPUnit

```bash
vendor/bin/phpunit --colors=always
```

Test files live in `tests/`. Mirror the `src/` structure:

```
tests/
  NeuralNetwork/
    Layers/
      DenseTest.php
  Estimators/
    Classifiers/
      RandomForestTest.php
  TensorTest.php
```

### Writing a Test

```php
final class DenseTest extends TestCase {
    public function testForwardShape(): void {
        $layer = new Dense(4, 8);
        $input = Tensor::randomNormal([2, 4]);
        $out   = $layer->forward($input);

        $this->assertSame([2, 8], $out->shape());
    }

    public function testNoMemoryLeak(): void {
        // Instantiate and destroy multiple times — valgrind catches leaks in CI
        for ($i = 0; $i < 100; $i++) {
            $t = Tensor::randomNormal([64, 64]);
            unset($t);
        }
        $this->assertTrue(true); // leak detection is valgrind-side
    }
}
```

### Memory Leak Check

```bash
valgrind --leak-check=full --error-exitcode=1 \
    php vendor/bin/phpunit tests/TensorTest.php
```

CI runs valgrind on the core tensor test suite on every PR.

---

## Benchmarking

### Run Benchmarks

```bash
vendor/bin/phpbench run --report=aggregate
```

### Write a Benchmark

```php
// benchmarks/MyFeatureBench.php
use PhpBench\Attributes as Bench;

class MyFeatureBench {
    private Tensor $a;
    private Tensor $b;

    public function setUp(): void {
        $this->a = Tensor::randomNormal([512, 512]);
        $this->b = Tensor::randomNormal([512, 512]);
    }

    #[Bench\Revs(10)]
    #[Bench\Iterations(3)]
    #[Bench\Warmup(1)]
    public function benchMyOp(): void {
        $this->a->myOp($this->b);
    }
}
```

### Performance Regression Policy

If your PR degrades performance of an existing benchmark by more than **5%**, it will be held for review. Include benchmark results in your PR description.

---

## Pull Request Process

1. **Fork** and create a feature branch: `git checkout -b feature/my-kernel`
2. **Build** the C backend and verify `nm -D libtensor.so | grep my_fn` shows the new export
3. **Test**: `vendor/bin/phpunit --colors=always` must pass
4. **Benchmark**: include before/after PHPBench results if changing a hot path
5. **Open PR**: use the [PR template](.github/PULL_REQUEST_TEMPLATE.md)
6. **CI** runs automatically: PHPUnit + lint + valgrind subset
7. **Review**: maintainer reviews within 5 business days

### Commit Style

```
type(scope): short description

Longer explanation if needed. Wrap at 72 chars.

- bullet points for multi-item changes
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`

Examples:
```
feat(tensor): add tensor_cumsum with OpenMP reduction
perf(inference): fuse rope_embed into attention_forward
fix(dataset): handle UTF-8 BOM in mmap CSV loader
test(quantization): add INT8 matmul accuracy regression test
```

---

## Performance Contract

Every contribution must uphold:

- **Zero-copy invariant**: PHP↔C boundary crossings must not introduce buffer copies in hot paths
- **No PHP-level matrix math**: Any operation on tensor data must go through FFI
- **Memory leak free**: `tensor_free()` must be called on every `TensorC*` that PHP owns, exactly once
- **Thread safety**: C functions called from PHP must be re-entrant (no global mutable state outside init)
- **OpenMP nesting**: Do not call a parallel region from within another parallel region

Violations of these rules are blocking issues and will prevent merge.

---

## Getting Help

- Open a [Discussion](https://github.com/ghostjat/pml/discussions) for design questions
- Open an [Issue](https://github.com/ghostjat/pml/issues) for bugs or unexpected behavior
- Tag `good first issue` items are good starting points for new contributors

Thank you for helping build serious AI infrastructure for PHP.
