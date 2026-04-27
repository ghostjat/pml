# Tensor API

`Pml\Tensor` is the numeric core of the framework. It wraps a native C tensor and exposes fast vectorized operators, linear algebra, and zero-copy views.

## Overview

The tensor API is designed for high-performance numeric work with a PHP-friendly wrapper.

The native C backend supports:

- dense tensor creation and views
- arithmetic and reductions
- BLAS-backed matrix multiplication
- convolution and attention primitives
- SafeTensors serialization and mmap-backed loading
- zero-copy dataset ingestion from CSV

## API Signature

```php
final class Tensor
```

### Construction

```php
public function __construct(array $shape = [], int $dtype = self::DTYPE_FLOAT32, ?\FFI\CData $arena = null)
public static function wrap(?\FFI\CData $ptr, ?self $parent = null): self
public static function fromArray(array $data, int $dtype = self::DTYPE_FLOAT32): self
public static function zeros(int ...$shape): self
public static function ones(int ...$shape): self
public static function randomNormal(array $shape, float $mean = 0.0, float $stddev = 1.0, ?\FFI\CData $arena = null): self
public static function randomUniform(array $shape, float $minVal = 0.0, float $maxVal = 1.0, ?\FFI\CData $arena = null): self
public static function range(float $start, float $end, float $step = 1.0): self
public static function linspace(float $start, float $end, int $steps): self
```

### Metadata and shape

```php
public function dtype(): int
public function shape(): array
public function size(): int
public function ndim(): int
public function isContiguous(): bool
public function contiguous(): self
```

### Zero-copy views

```php
public function view(): self
public function slice(int $axis, int $start, int $length): self
public function sliceStep(int $axis, int $start, int $end, int $step): self
public function row(int $row): self
public function col(int $col): self
```

### Shape and layout operations

```php
public function reshape(int ...$newShape): self
public function flatten(): self
public function expandDims(int $axis): self
public function squeeze(): self
public function transpose(): self
public function transposeNd(array $axes): self
public function swapaxes(int $axis1, int $axis2): self
```

### Arithmetic

```php
public function add(Tensor $b): self
public function sub(Tensor $b): self
public function mul(Tensor $b): self
public function div(Tensor $b): self
public function addScalar(float $val): self
public function mulScalar(float $val): self
public function pow(Tensor $b): self
public function clip(float $min, float $max): self
```

### In-place arithmetic

```php
public function addInplace(Tensor $b): self
public function subInplace(Tensor $b): self
public function mulInplace(Tensor $b): self
public function divInplace(Tensor $b): self
public function addScalarInplace(float $val): self
public function mulScalarInplace(float $val): self
```

### Unary functions

```php
public function sqrt(): self
public function square(): self
public function abs(): self
public function sign(): self
public function exp(): self
public function log(): self
public function log1p(): self
public function round(): self
public function floor(): self
public function ceil(): self
public function sigmoid(): self
public function tanh(): self
public function relu(): self
public function sin(): self
public function cos(): self
public function tan(): self
public function asin(): self
public function acos(): self
public function atan(): self
```

### Logical operations

```php
public function equal(Tensor $b): self
public function notEqual(Tensor $b): self
public function greater(Tensor $b): self
public function greaterEqual(Tensor $b): self
public function less(Tensor $b): self
public function lessEqual(Tensor $b): self
public function logicalNot(): self
```

### NaN / Inf handling

```php
public function isNan(): self
public function isInf(): self
public function nanToNumInplace(float $nan_val = 0.0, float $posinf = 1e38, float $neginf = -1e38): self
public function any(): bool
public function all(): bool
```

### Indexing and selection

```php
public function where(Tensor $x, Tensor $y): self
public function booleanIndex(Tensor $mask): self
public function take(Tensor $indices, int $axis): self
public function unique(): self
public function bincount(): self
```

### Concatenation and padding

```php
public static function concat(array $tensors, int $axis): self
public function pad(array $padWidth, float $constantValue = 0.0): self
```

### Sorting and aggregation

```php
public function argsort(int $axis = -1): self
public function sort(int $axis = -1): self
public function topk(int $k, int $axis = -1): self
public function sum(): float
public function product(): float
public function mean(): float
public function min(): float
public function max(): float
public function argmin(): int
public function argmax(): int
public function variance(): float
public function std(): float
public function median(): float
public function sumAxis(int $axis): self
public function meanAxis(int $axis): self
public function maxAxis(int $axis): self
public function minAxis(int $axis): self
public function cumsum(int $axis): self
public function sumMulti(array $axes): self
public function meanMulti(array $axes): self
public function maxMulti(array $axes): self
```

### Normalization

```php
public function normalize(): self
public function standardize(): self
public function normalizeInplace(): self
public function standardizeInplace(): self
```

### Linear algebra

```php
public function dot(Tensor $b): float
public function trace(): float
public function matmul(Tensor $b, bool $transA = false, bool $transB = false): self
public function matmulInto(Tensor $A, Tensor $B, bool $transA = false, bool $transB = false): void
public function sumAxisInto(Tensor $A, int $axis): void
public function bmm(Tensor $b): self
public function inverse(): self
public function pinv(): self
public function solve(Tensor $B): self
public function cholesky(): self
public function lu(): array
public function svd(): array
public function eigenSym(): array
public function ref(): self
public function rref(): self
```

### Deep learning primitives

```php
public function linear(Tensor $W, ?Tensor $bias = null): self
public function addRelu(Tensor $b): self
public function mulAdd(Tensor $B, Tensor $C): self
public function im2col(int $kernel_h, int $kernel_w, int $stride_h, int $stride_w, int $pad_h, int $pad_w): self
public function col2im(int $batch, int $channels, int $height, int $width, int $kernel_h, int $kernel_w, int $stride_h, int $stride_w, int $pad_h, int $pad_w): self
public function conv2d(Tensor $W, ?Tensor $bias = null, int $stride_h = 1, int $stride_w = 1, int $pad_h = 0, int $pad_w = 0): self
public function conv2dBackward(Tensor $X, Tensor $W, int $stride_h = 1, int $stride_w = 1, int $pad_h = 0, int $pad_w = 0): array
public function embeddingLookup(Tensor $weights): self
```

### Transformer and attention primitives

```php
public function rmsnorm(float $eps = 1e-5): void
public function applyRope(Tensor $k, int $headDim, int $pos, float $baseFreq = 10000.0, float $scale = 1.0): void
public function softmaxInplace(): void
public function attention(Tensor $k, Tensor $v): self
public function attentionKV(KVCache $cache): self
```

### Persistence and I/O

```php
public function save(string $filepath): void
public static function load(string $filepath): self
public static function saveSafetensors(string $filepath, string $jsonHeader, array $tensors): int
public static function datasetFromCsv(string $filepath, int $labelCol = -1, bool $hasHeader = true): array
public static function fromMmap(string $filepath, int $byteOffset, array $shape, int $dtype = 0): self
public function mmapFree(): void
public function copyFrom(Tensor $src): void
public function toFlatArray(): array
public function buffer(): \FFI\CData
```

### Advanced utilities

```php
public static function configureThreading(int $ompThreads, int $blasThreads = 1): void
```

## What it does

`Tensor` wraps the native `TensorC` pointer and offers a PHP-friendly API for high-performance numeric operations. Most methods call C kernels and return new tensors with C-managed data.

## When to use it

- Use `Tensor` for matrix math and neural network computation.
- Use zero-copy views for slicing and batching.
- Use `fromArray()` to convert small PHP arrays into native tensors.
- Use `toFlatArray()` only for final export or debugging.

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `shape` | `array` | Tensor dimensions, e.g. `[2, 3]`.
| `dtype` | `int` | One of `Tensor::DTYPE_FLOAT32`, `DTYPE_INT32`, `DTYPE_INT64`.
| `axis` | `int` | Axis index for slicing or aggregation.
| `indices` | `Tensor` | Tensor of integer positions.
| `mask` | `Tensor` | Boolean mask tensor.
| `padWidth` | `array` | Padding width per axis.
| `k` | `int` | Number of top elements to select.

## Return values

- Most operations return a new `Tensor` object backed by native C memory.
- In-place methods return the same tensor instance.
- `sum()`, `mean()`, and similar aggregations return scalars.
- `lu()` and `svd()` return arrays of tensors.

## Example Usage

```php
use Pml\Tensor;

$x = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
$w = Tensor::ones(2, 2);
$y = $x->matmul($w)->relu();

$view = $y->slice(0, 0, 1); // first row, zero-copy
print_r($view->toFlatArray());

$z = $y->sumAxis(0);
var_export($z->toFlatArray());
```

## Performance notes

- `Tensor::fromArray()` packs the PHP values into a binary string and performs a single `FFI::memcpy`.
- Views such as `slice()` retain a parent reference to avoid freeing shared memory.
- `configureThreading()` controls OpenMP and BLAS so you can prevent CPU oversubscription.
- `saveSafetensors()` writes tensor bytes via the native kernel rather than through PHP.

## Common mistakes

- Calling `toFlatArray()` inside the training loop forces a full copy.
- Forgetting to retain the parent reference for view tensors can lead to use-after-free bugs; `Tensor` handles that internally.
- Assuming `reshape()` modifies the tensor in-place; it returns a new view.
- Passing `Tensor` objects of mismatched dtype to operations that require the same dtype.
