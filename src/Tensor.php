<?php

declare(strict_types=1);

namespace Pml;

/**
 * Tensor: The foundation of PhpTensor.
 *
 * Design Principles:
 * - SINGLE 1D flat buffer in C memory (float32 OR int8_t). NO PHP arrays for data.
 * - Shape + strides live in PHP-land as int[]. They are cheap metadata.
 * - GC-managed: buffer is owned by PHP GC (FFI::new(..., true)).
 * - Zero-copy operations use strides/transpose flags rather than re-allocating.
 * - Data never leaves C memory unless explicitly requested via toArray().
 *
 * Int8 quantisation:
 * - When dtype === self::INT8, $buffer is int8_t[N] holding symmetric quantised values.
 * - $quantScale is the per-tensor scale: float32_val ≈ int8_val * scale.
 * - Ops::matmul() detects INT8 inputs and JIT-dequantises before calling sgemm.
 */
class Tensor
{
    // ── Dtype constants ───────────────────────────────────────────────────
    public const FLOAT32 = 0;
    public const INT8    = 1;

    public readonly \FFI\CData $buffer;
    public readonly array $shape;    // e.g. [batch, seq, d_model]
    public readonly array $strides;  // row-major strides (in elements, not bytes)
    public readonly int   $size;     // total element count = product(shape)

    // ── Quantisation metadata (only meaningful when dtype === INT8) ────────
    public readonly int   $dtype;         // FLOAT32 | INT8
    public readonly float $quantScale;    // float32 ≈ int8 * scale
    public readonly int   $quantZeroPoint;// asymmetric zero-point (0 for symmetric)

    // Transpose metadata — avoids re-allocating on transposition
    /** @internal */
    public bool  $_transposed = false;
    /** @internal */
    public array $_transposedShape = [];

    // ── Autograd graph ────────────────────────────────────────────────────
    //
    // Design (reverse-mode automatic differentiation / "autograd tape"):
    //
    //   Every Tensor that participates in a differentiable computation carries:
    //
    //   $requiresGrad — true for learnable parameters and any tensor derived
    //                   from them.  Tensors without this flag are treated as
    //                   constants: no grad buffer is allocated and no backward
    //                   closure is ever registered for them.
    //
    //   $grad         — Float32 FFI buffer of the same size as $buffer.
    //                   Allocated lazily by initGrad() on the first backward
    //                   pass; starts at zero (FFI::new zero-initialises).
    //                   Gradients *accumulate* here (+=) so you must call
    //                   zeroGrad() (or $optimizer->zeroGrad()) before each
    //                   training step.
    //
    //   $_prev        — Tensor[] of immediate parents in the computational
    //                   graph.  Only populated when $requiresGrad is true and
    //                   a differentiable Op produced this tensor.
    //
    //   $_backward    — Closure registered by the Op that produced this tensor.
    //                   When called it reads $this->grad (the upstream gradient
    //                   flowing into this node) and accumulates the appropriate
    //                   contribution into each parent's $grad buffer.
    //
    // Leaf tensors (e.g. model weights created by the user) have empty $_prev
    // and null $_backward — they are the endpoints of backpropagation.

    /** Whether gradient tracking is active for this tensor. */
    public bool $requiresGrad = false;

    /** Gradient buffer: float[$size], null until initGrad() is first called. */
    public ?\FFI\CData $grad = null;

    /**
     * Immediate parent tensors in the computational graph.
     * @var Tensor[]
     */
    public array $_prev = [];

    /**
     * Backward closure registered by the Op that produced this tensor.
     * Called during backward() in reverse topological order.
     */
    public ?\Closure $_backward = null;

    // ── Construction ──────────────────────────────────────────────────────

    /**
     * @param array           $shape     e.g. [3, 4] or [2, 3, 4]
     * @param \FFI\CData|null $buffer    If provided, tensor is a VIEW into existing memory.
     * @param int             $dtype     self::FLOAT32 (default) or self::INT8
     * @param float           $scale     Quantisation scale (INT8 only)
     * @param int             $zeroPoint Quantisation zero-point (INT8 only, 0 = symmetric)
     */
    public function __construct(
        array        $shape,
        ?\FFI\CData  $buffer    = null,
        int          $dtype     = self::FLOAT32,
        float        $scale     = 1.0,
        int          $zeroPoint = 0
    ) {
        if (empty($shape)) {
            throw new \InvalidArgumentException('Tensor shape must not be empty.');
        }
        foreach ($shape as $dim) {
            if ($dim <= 0) throw new \InvalidArgumentException("All dimensions must be > 0, got {$dim}.");
        }

        $this->shape          = array_map('intval', $shape);
        $this->size           = (int) array_product($this->shape);
        $this->strides        = self::computeStrides($this->shape);
        $this->dtype          = $dtype;
        $this->quantScale     = $scale;
        $this->quantZeroPoint = $zeroPoint;

        if ($buffer !== null) {
            $this->buffer = $buffer;
        } elseif ($dtype === self::INT8) {
            // int8_t[N] — 1 byte per element; 4× smaller than float32
            $this->buffer = BlasEngine::get()->allocInt8($this->size, true);
        } else {
            // GC-owned float[N]: PHP will call FFI destructor when Tensor goes out of scope
            $this->buffer = BlasEngine::get()->allocFloat($this->size, true);
        }
    }

    // ── Int8 quantisation factory methods ─────────────────────────────────

    /**
     * Quantise a FLOAT32 tensor to INT8 using symmetric per-tensor quantisation.
     *
     * Algorithm:
     *   scale = max(|x|) / 127
     *   q[i]  = clamp( round(x[i] / scale), -128, 127 )
     *
     * This is a one-time, load-time cost.  At inference time Ops::matmul()
     * will dequantise on the fly into a temporary F32 buffer, call sgemm,
     * and free the temporary.
     *
     * @param float|null $scale  Override the auto-computed scale (e.g. from
     *                           a stored calibration file).
     */
    public static function quantize(self $src, ?float $scale = null, int $zeroPoint = 0): self
    {
        if ($src->dtype !== self::FLOAT32) {
            throw new \InvalidArgumentException('quantize() requires a FLOAT32 source tensor.');
        }

        // Find max absolute value to compute the scale
        $absMax = 0.0;
        for ($i = 0; $i < $src->size; $i++) {
            $v = abs((float) $src->buffer[$i]);
            if ($v > $absMax) $absMax = $v;
        }

        $scale ??= $absMax / 127.0;
        if ($scale < 1e-9) $scale = 1e-9; // avoid division by zero for zero tensors

        $out      = new self($src->shape, null, self::INT8, $scale, $zeroPoint);
        $invScale = 1.0 / $scale;

        for ($i = 0; $i < $src->size; $i++) {
            $q = (int) round(((float) $src->buffer[$i]) * $invScale - $zeroPoint);
            $out->buffer[$i] = max(-128, min(127, $q));
        }

        return $out;
    }

    /**
     * Dequantise an INT8 tensor back to FLOAT32.
     *
     *   float32[i] = (int8[i] + zeroPoint) * scale
     *
     * Ops::matmul() calls this internally; you rarely need it directly.
     */
    public function dequantize(): self
    {
        if ($this->dtype !== self::INT8) {
            throw new \InvalidArgumentException('dequantize() requires an INT8 tensor.');
        }

        $out    = new self($this->shape); // FLOAT32 by default
        $scale  = $this->quantScale;
        $zp     = $this->quantZeroPoint;

        for ($i = 0; $i < $this->size; $i++) {
            $out->buffer[$i] = ((float)((int) $this->buffer[$i]) + $zp) * $scale;
        }

        return $out;
    }

    // ── Factory Methods ───────────────────────────────────────────────────

    /**
     * Create a Tensor from a PHP array (any depth, must be rectangular).
     * Uses binary pack() → FFI::memcpy for maximum speed.
     */
    public static function fromArray(array $data, ?array $shape = null): self
    {
        $flat  = self::flattenNested($data);
        $shape = $shape ?? self::inferShape($data);
        $t     = new self($shape);

        // Binary pack float32 → C buffer in one syscall
        $bytes = pack('f*', ...$flat);
        \FFI::memcpy($t->buffer, $bytes, $t->size * 4);

        return $t;
    }

    /**
     * All-zeros tensor.
     */
    public static function zeros(array $shape): self
    {
        return new self($shape); // FFI::new zero-initializes
    }

    /**
     * All-ones tensor.
     */
    public static function ones(array $shape): self
    {
        $t = new self($shape);
        BlasEngine::get()->ffi->cblas_sscal($t->size, 0.0, $t->buffer, 1); // zero first (already zero)
        // saxpy: y = alpha*x + y → ones: fill via sscal trick
        // Fastest: unpack with binary string
        $bytes = pack('f*', ...array_fill(0, $t->size, 1.0));
        \FFI::memcpy($t->buffer, $bytes, $t->size * 4);
        return $t;
    }

    /**
     * Scalar-filled tensor.
     */
    public static function full(array $shape, float $value): self
    {
        $t = new self($shape);
        if ($value === 0.0) return $t; // already zero
        $bytes = pack('f*', ...array_fill(0, $t->size, $value));
        \FFI::memcpy($t->buffer, $bytes, $t->size * 4);
        return $t;
    }

    /**
     * Identity matrix (2D only).
     */
    public static function eye(int $n): self
    {
        $t = self::zeros([$n, $n]);
        for ($i = 0; $i < $n; $i++) {
            $t->buffer[$i * $n + $i] = 1.0;
        }
        return $t;
    }

    /**
     * Standard normal (Gaussian) random tensor — N(0, 1).
     * Uses Box-Muller transform for true Gaussian samples.
     */
    public static function randn(array $shape, float $mean = 0.0, float $std = 1.0): self
    {
        $t = new self($shape);
        $n = $t->size;

        for ($i = 0; $i < $n; $i += 2) {
            // Box-Muller
            do { $u1 = mt_rand() / mt_getrandmax(); } while ($u1 === 0.0);
            $u2 = mt_rand() / mt_getrandmax();

            $mag = $std * sqrt(-2.0 * log($u1));
            $t->buffer[$i]        = (float)($mag * cos(2.0 * M_PI * $u2) + $mean);
            if ($i + 1 < $n) {
                $t->buffer[$i + 1] = (float)($mag * sin(2.0 * M_PI * $u2) + $mean);
            }
        }
        return $t;
    }

    /**
     * Uniform random tensor U[low, high).
     */
    public static function uniform(array $shape, float $low = 0.0, float $high = 1.0): self
    {
        $t     = new self($shape);
        $range = $high - $low;
        for ($i = 0; $i < $t->size; $i++) {
            $t->buffer[$i] = (float)($low + (mt_rand() / mt_getrandmax()) * $range);
        }
        return $t;
    }

    /**
     * He (Kaiming) initialization — best for ReLU/GELU layers.
     * std = sqrt(2 / fan_in)
     */
    public static function heInit(array $shape): self
    {
        $fanIn = $shape[count($shape) - 2] ?? $shape[0];
        return self::randn($shape, 0.0, sqrt(2.0 / $fanIn));
    }

    /**
     * Xavier (Glorot) initialization — best for tanh/sigmoid.
     * std = sqrt(2 / (fan_in + fan_out))
     */
    public static function xavierInit(array $shape): self
    {
        $fanIn  = $shape[count($shape) - 2] ?? $shape[0];
        $fanOut = $shape[count($shape) - 1] ?? $shape[0];
        return self::randn($shape, 0.0, sqrt(2.0 / ($fanIn + $fanOut)));
    }

    /**
     * Arange: [start, start+step, start+2*step, ...] as flat 1D tensor.
     */
    public static function arange(float $start, float $stop, float $step = 1.0): self
    {
        $values = [];
        for ($v = $start; $v < $stop; $v += $step) {
            $values[] = $v;
        }
        return self::fromArray($values, [count($values)]);
    }

    /**
     * Linspace: N evenly spaced values from start to stop (inclusive).
     */
    public static function linspace(float $start, float $stop, int $num): self
    {
        $values = [];
        $step   = ($stop - $start) / max(1, $num - 1);
        for ($i = 0; $i < $num; $i++) {
            $values[] = $start + $i * $step;
        }
        return self::fromArray($values, [$num]);
    }

    // ── Shape Operations ──────────────────────────────────────────────────

    /**
     * Reshape: Returns a new Tensor view with different shape, same buffer.
     * Zero-copy — shares the underlying C buffer.
     */
    public function reshape(array $newShape): self
    {
        $newSize = (int) array_product($newShape);
        if ($newSize !== $this->size) {
            throw new \InvalidArgumentException(
                "Reshape: cannot change total size from {$this->size} to {$newSize}."
            );
        }
        return new self($newShape, $this->buffer);
    }

    /**
     * Flatten to 1D. Zero-copy view.
     */
    public function flatten(): self
    {
        return $this->reshape([$this->size]);
    }

    /**
     * Transpose a 2D tensor.
     *
     * Strategy: Rather than copying data, we set a _transposed flag that is
     * consumed by Ops::matmul() to flip the CblasNoTrans/CblasTrans flag.
     * This is truly zero-copy for the GEMM path.
     *
     * For non-GEMM consumers (element access, etc.) we provide a physical
     * transposed copy via transposePhysical().
     */
    public function T(): self
    {
        if (count($this->shape) !== 2) {
            throw new \RuntimeException('Logical transpose only supported for 2D tensors. Use transposeAxes() for N-D.');
        }

        $view               = new self(array_reverse($this->shape), $this->buffer);
        $view->_transposed  = true;
        $view->_transposedShape = $this->shape; // original shape before flip
        return $view;
    }

    /**
     * Physical (copying) transpose for 2D — required when the tensor is
     * used outside GEMM (e.g. element-wise ops, printing).
     */
    public function transposePhysical(): self
    {
        if (count($this->shape) !== 2) {
            throw new \RuntimeException('transposePhysical() only supports 2D tensors.');
        }
        [$m, $n] = $this->shape;
        $out = new self([$n, $m]);

        for ($i = 0; $i < $m; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $out->buffer[$j * $m + $i] = $this->buffer[$i * $n + $j];
            }
        }
        return $out;
    }

    /**
     * Physical (copying) 2-D transpose: shape [M, N] → [N, M].
     *
     * Alias for transposePhysical() — provided for NumPy naming parity
     * (np.ndarray.transpose / np.transpose).
     *
     * Re-lays out data so that result->buffer[col × M + row] = this->buffer[row × N + col].
     * For BLAS GEMM, prefer T() which is zero-copy and passes CblasTrans instead of copying.
     *
     * @throws \RuntimeException on non-2D tensor.
     */
    public function transpose(): self
    {
        return $this->transposePhysical();
    }

    /**
     * Permute axes for N-D tensors (physical copy).
     * e.g. $t->permuteAxes([0, 2, 1]) on shape [2,3,4] → [2,4,3]
     */
    public function permuteAxes(array $axes): self
    {
        $ndim = count($this->shape);
        if (count($axes) !== $ndim) {
            throw new \InvalidArgumentException('Number of axes must match tensor ndim.');
        }
        $newShape = [];
        foreach ($axes as $ax) {
            $newShape[] = $this->shape[$ax];
        }
        $out      = new self($newShape);
        $oldStrides = $this->strides;
        $newStrides = self::computeStrides($newShape);

        // Iterate over all element indices
        for ($flatIdx = 0; $flatIdx < $this->size; $flatIdx++) {
            // Compute multi-index in new shape
            $remainder  = $flatIdx;
            $newMultiIdx = [];
            foreach ($newShape as $dimSize) {
                $newMultiIdx[] = intdiv($remainder, $dimSize > 0 ? (int)($this->size / $dimSize) : 1);
                $remainder     = $remainder % ($dimSize > 0 ? (int)($this->size / $dimSize) : 1);
            }
            // Map back to old multi-index via axes permutation
            $oldFlatIdx = 0;
            foreach ($axes as $newAx => $oldAx) {
                $oldFlatIdx += $newMultiIdx[$newAx] * $oldStrides[$oldAx];
            }
            $out->buffer[$flatIdx] = $this->buffer[$oldFlatIdx];
        }
        return $out;
    }

    /**
     * Squeeze: remove all dimensions of size 1.
     */
    public function squeeze(): self
    {
        $newShape = array_values(array_filter($this->shape, fn($d) => $d !== 1));
        if (empty($newShape)) $newShape = [1];
        return $this->reshape($newShape);
    }

    /**
     * Unsqueeze: insert a size-1 dimension at the given axis.
     */
    public function unsqueeze(int $axis): self
    {
        $newShape = $this->shape;
        array_splice($newShape, $axis, 0, [1]);
        return $this->reshape($newShape);
    }

    // ── Row / Column Extraction ───────────────────────────────────────────

    /**
     * Extract row $index from a 2D tensor. Returns a 1D tensor.
     * Uses BLAS scopy — zero additional PHP loops.
     */
    public function getRow(int $index): self
    {
        if (count($this->shape) !== 2) {
            throw new \RuntimeException('getRow() requires a 2D tensor.');
        }
        $cols = $this->shape[1];
        $out  = new self([$cols]);
        $src  = \FFI::cast('float*', \FFI::addr($this->buffer[$index * $cols]));
        BlasEngine::get()->ffi->cblas_scopy($cols, $src, 1, $out->buffer, 1);
        return $out;
    }

    /**
     * Slice rows [$start, $end) from a 2D tensor. Returns a 2D tensor view (zero-copy).
     */
    public function sliceRows(int $start, int $end): self
    {
        if (count($this->shape) !== 2) {
            throw new \RuntimeException('sliceRows() requires a 2D tensor.');
        }
        $cols   = $this->shape[1];
        $nRows  = $end - $start;
        $srcPtr = \FFI::cast('float*', \FFI::addr($this->buffer[$start * $cols]));
        return new self([$nRows, $cols], $srcPtr);
    }

    /**
     * Slice along axis 0, returning a physical copy of the selected rows.
     *
     * Works for any N-D tensor; axis 0 is sliced and all trailing axes are
     * preserved intact.  The result is a new allocation — safe to hold after
     * $this is freed or mutated.
     *
     * Byte arithmetic (row-major float32, 4 bytes per element):
     *   stride₀        = $this->strides[0]   — elements per axis-0 index
     *   first element  = $start  × stride₀
     *   bytes to copy  = $length × stride₀ × 4
     *
     * @param int      $start  First axis-0 index (0-indexed, inclusive).
     * @param int|null $length Rows to copy; null = from $start to the last row.
     * @return self  New tensor with shape [$length, …original tail shape].
     * @throws \InvalidArgumentException if the range is out-of-bounds.
     */
    public function slice(int $start, ?int $length = null): self
    {
        $axis0  = $this->shape[0];
        $length ??= $axis0 - $start;

        if ($start < 0 || $length < 1 || $start + $length > $axis0) {
            throw new \InvalidArgumentException(
                "slice(): out-of-bounds — tensor has {$axis0} rows on axis 0, "
                . "requested start={$start}, length={$length}."
            );
        }

        $newShape    = $this->shape;
        $newShape[0] = $length;

        // stride₀ = elements per axis-0 index (product of all trailing dimensions)
        $stride0 = $this->strides[0];

        // Bytes to copy: $length rows × stride₀ elements/row × 4 bytes/float32
        $nBytes = $length * $stride0 * 4;
        $out    = new self($newShape);

        // \FFI::addr($buffer[$offset]) returns a C pointer to element $offset,
        // letting memcpy treat the slice as a contiguous byte block.
        $srcPtr = \FFI::cast('float*', \FFI::addr($this->buffer[$start * $stride0]));
        \FFI::memcpy($out->buffer, $srcPtr, $nBytes);

        return $out;
    }

    /**
     * Copy columns [$start, $end) from a 2D tensor. Returns a new [rows, end-start] tensor.
     */
    public function sliceCols(int $start, int $end): self
    {
        if (count($this->shape) !== 2) {
            throw new \RuntimeException('sliceCols() requires a 2D tensor.');
        }
        $rows    = $this->shape[0];
        $cols    = $this->shape[1];
        $width   = $end - $start;
        $out     = new self([$rows, $width]);
        $blas    = BlasEngine::get()->ffi;
        for ($r = 0; $r < $rows; $r++) {
            $src = \FFI::cast('float*', \FFI::addr($this->buffer[$r * $cols + $start]));
            $dst = \FFI::cast('float*', \FFI::addr($out->buffer[$r * $width]));
            $blas->cblas_scopy($width, $src, 1, $dst, 1);
        }
        return $out;
    }

    /**
     * Write a [rows, width] tensor into this tensor's columns starting at $start.
     */
    public function setColSlice(int $start, Tensor $src): void
    {
        if (count($this->shape) !== 2) {
            throw new \RuntimeException('setColSlice() requires a 2D tensor.');
        }
        $rows  = $this->shape[0];
        $cols  = $this->shape[1];
        $width = $src->shape[1];
        $blas  = BlasEngine::get()->ffi;
        for ($r = 0; $r < $rows; $r++) {
            $s = \FFI::cast('float*', \FFI::addr($src->buffer[$r * $width]));
            $d = \FFI::cast('float*', \FFI::addr($this->buffer[$r * $cols + $start]));
            $blas->cblas_scopy($width, $s, 1, $d, 1);
        }
    }

    /**
     * Set row $index from a 1D tensor. Uses BLAS scopy.
     */
    public function setRow(int $index, Tensor $row): void
    {
        if (count($this->shape) !== 2) {
            throw new \RuntimeException('setRow() requires a 2D tensor.');
        }
        $cols = $this->shape[1];
        $dst  = \FFI::cast('float*', \FFI::addr($this->buffer[$index * $cols]));
        BlasEngine::get()->ffi->cblas_scopy($cols, $row->buffer, 1, $dst, 1);
    }

    /**
     * Vertically stack $other below $this (concatenate along axis 0).
     *
     * Both tensors must have identical shapes on every axis after axis 0.
     * For 2D [M₁, C] and [M₂, C] this means column count C must match;
     * for 1D there is no tail axis to validate.
     *
     * Two FFI::memcpy() calls perform the full concatenation without any
     * PHP element-level loops:
     *
     *   Copy 1 — top half:
     *     src  = $this->buffer  (element 0)
     *     dst  = $out->buffer   (element 0)
     *     size = $this->size × 4 bytes
     *
     *   Copy 2 — bottom half:
     *     src  = $other->buffer  (element 0)
     *     dst  = $out->buffer    (element $this->size)   ← pointer arithmetic
     *     size = $other->size × 4 bytes
     *
     * @param Tensor $other Tensor to append below $this.
     * @return self  New tensor with shape [$this->shape[0] + $other->shape[0], …tail].
     * @throws \InvalidArgumentException if the tail shapes differ.
     */
    public function vstack(self $other): self
    {
        $tailThis  = array_slice($this->shape,  1);
        $tailOther = array_slice($other->shape, 1);

        if ($tailThis !== $tailOther) {
            throw new \InvalidArgumentException(
                'vstack(): tail shape mismatch — '
                . 'this=[' . implode(', ', $this->shape) . '], '
                . 'other=[' . implode(', ', $other->shape) . '].'
            );
        }

        $newShape    = $this->shape;
        $newShape[0] = $this->shape[0] + $other->shape[0];
        $out         = new self($newShape);

        // Copy 1: $this fills the top half of $out — simple pointer-to-pointer copy.
        \FFI::memcpy($out->buffer, $this->buffer, $this->size * 4);

        // Copy 2: $other fills the bottom half.
        // \FFI::addr($out->buffer[$this->size]) is the C pointer to the first
        // element that belongs to $other's data, cast to float* for memcpy.
        $dstPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$this->size]));
        \FFI::memcpy($dstPtr, $other->buffer, $other->size * 4);

        return $out;
    }

    // ── Element Access ────────────────────────────────────────────────────

    /**
     * Get element by multi-dimensional index.
     * e.g. $t->get(1, 2) for a 2D tensor.
     */
    public function get(int ...$indices): float
    {
        return (float) $this->buffer[$this->flatIndex($indices)];
    }

    /**
     * Set element by multi-dimensional index.
     */
    public function set(float $value, int ...$indices): void
    {
        $this->buffer[$this->flatIndex($indices)] = $value;
    }

    // ── Reduction ─────────────────────────────────────────────────────────

    /**
     * Sum of all elements. Uses BLAS sasum on abs values — not general.
     * For signed sum we use a simple accumulation loop (BLAS has no signed sum).
     */
    public function sum(): float
    {
        $total = 0.0;
        for ($i = 0; $i < $this->size; $i++) {
            $total += $this->buffer[$i];
        }
        return $total;
    }

    /**
     * Mean of all elements.
     */
    public function mean(): float
    {
        return $this->sum() / $this->size;
    }

    /**
     * Max element value.
     */
    public function max(): float
    {
        $ffi = BlasEngine::get()->ffi;
        $idx = $ffi->cblas_isamax($this->size, $this->buffer, 1);
        return (float) $this->buffer[$idx];
    }

    /**
     * Argmax (index of maximum element).
     */
    public function argmax(): int
    {
        // cblas_isamax finds max of abs values — for ML logits (positive after softmax)
        // this is fine; for general use we do a PHP scan.
        $maxIdx = 0;
        $maxVal = $this->buffer[0];
        for ($i = 1; $i < $this->size; $i++) {
            if ($this->buffer[$i] > $maxVal) {
                $maxVal = $this->buffer[$i];
                $maxIdx = $i;
            }
        }
        return $maxIdx;
    }

    /**
     * L2 norm (Euclidean). Uses BLAS snrm2.
     */
    public function norm(): float
    {
        return (float) BlasEngine::get()->ffi->cblas_snrm2($this->size, $this->buffer, 1);
    }

    // ── In-Place Scalar Operations ────────────────────────────────────────

    /**
     * Scale all elements by a scalar in-place. Uses BLAS sscal.
     */
    public function scaleInPlace(float $alpha): self
    {
        BlasEngine::get()->ffi->cblas_sscal($this->size, $alpha, $this->buffer, 1);
        return $this;
    }

    /**
     * Add scalar to all elements in-place.
     */
    public function addScalarInPlace(float $scalar): self
    {
        for ($i = 0; $i < $this->size; $i++) {
            $this->buffer[$i] += $scalar;
        }
        return $this;
    }

    /**
     * Clip values in-place: elements clamped to [min, max].
     */
    public function clipInPlace(float $min, float $max): self
    {
        for ($i = 0; $i < $this->size; $i++) {
            $v = (float) $this->buffer[$i];
            if ($v < $min) $this->buffer[$i] = $min;
            elseif ($v > $max) $this->buffer[$i] = $max;
        }
        return $this;
    }

    // ── Data Export ───────────────────────────────────────────────────────

    /**
     * Export to a flat PHP float array. Use sparingly — crosses FFI boundary.
     */
    public function toArray(): array
    {
        $bytes = \FFI::string($this->buffer, $this->size * 4);
        return array_values(unpack('f*', $bytes));
    }

    /**
     * Export to a nested PHP array matching the tensor's shape.
     */
    public function toNestedArray(): array
    {
        $flat = $this->toArray();
        return self::nestArray($flat, $this->shape);
    }

    /**
     * Clone into a new Tensor with its own buffer copy.
     */
    public function clone(): self
    {
        $copy = new self($this->shape);
        \FFI::memcpy($copy->buffer, $this->buffer, $this->size * 4);
        return $copy;
    }

    // ── Autograd Methods ──────────────────────────────────────────────────

    /**
     * Allocate the gradient buffer if it doesn't exist yet.
     *
     * Called by backward closures on the PARENT tensors before accumulating
     * into them with saxpy.  FFI::new zero-initialises, so the first backward
     * pass starts accumulating from 0.0 — correct behaviour.
     *
     * Idempotent: a second call does nothing.
     */
    public function initGrad(): void
    {
        if ($this->grad === null) {
            // GC-owned float[size] — initialised to 0.0 by the C allocator
            $this->grad = BlasEngine::get()->allocFloat($this->size, true);
        }
    }

    /**
     * Zero the gradient buffer in-place using FFI memset.
     *
     * Call this (or AdamW::zeroGrad()) before every training step so that
     * gradients from the previous step do not accumulate into the current one.
     *
     * IEEE 754: the bit-pattern 0x00000000 is exactly 0.0f, so a byte-level
     * memset to zero is safe for float32 buffers.
     */
    public function zeroGrad(): void
    {
        if ($this->grad !== null) {
            \FFI::memset($this->grad, 0, $this->size * 4);
        }
    }

    /**
     * Run reverse-mode automatic differentiation from this (scalar) tensor.
     *
     * Algorithm:
     *   1. Topologically sort all tensors reachable via _prev links.
     *   2. Seed $this->grad = 1.0  (dL/dL = 1).
     *   3. Walk the sorted list in REVERSE order, calling each _backward
     *      closure.  Each closure reads its own $grad (the upstream signal)
     *      and uses BLAS to accumulate contributions into its parents' $grad.
     *
     * Only call this on the final scalar loss tensor (size === 1).
     *
     * @throws \LogicException if the tensor does not require grad or is not scalar.
     */
    public function backward(): void
    {
        if (!$this->requiresGrad) {
            throw new \LogicException(
                'backward() called on a tensor with requiresGrad=false. '
                . 'Mark leaf parameters with $tensor->requiresGrad = true.'
            );
        }
        if ($this->size !== 1) {
            throw new \LogicException(
                "backward() requires a scalar (size=1) loss tensor, got size={$this->size}. "
                . 'Call backward() on the scalar loss returned by CrossEntropyLoss::forward().'
            );
        }

        // ── 1. Topological sort ───────────────────────────────────────────
        $topo    = [];
        $visited = new \SplObjectStorage();
        self::buildTopo($this, $topo, $visited);

        // ── 2. Seed gradient: dL/dL = 1.0 ────────────────────────────────
        $this->initGrad();
        $this->grad[0] = 1.0;

        // ── 3. Reverse-mode gradient propagation ─────────────────────────
        foreach (array_reverse($topo) as $node) {
            if ($node->_backward !== null) {
                ($node->_backward)();
            }
        }
    }

    /**
     * Return a copy of this tensor detached from the computational graph.
     *
     * The returned tensor shares the same DATA buffer (zero-copy) but has
     * requiresGrad=false and no backward closure.  Use this to stop gradient
     * flow (e.g. when passing targets to a loss function).
     */
    public function detach(): self
    {
        $view               = new self($this->shape, $this->buffer);
        $view->requiresGrad = false;
        return $view;
    }

    // ── Utility ───────────────────────────────────────────────────────────

    public function ndim(): int    { return count($this->shape); }
    public function dtype(): string { return 'float32'; }

    public function __toString(): string
    {
        $shapeStr = implode('×', $this->shape);
        $flat     = $this->toArray();
        $preview  = array_slice($flat, 0, min(8, $this->size));
        $previewStr = implode(', ', array_map(fn($v) => number_format($v, 4), $preview));
        if ($this->size > 8) $previewStr .= ', ...';
        return "Tensor(shape=[{$shapeStr}], dtype=float32, [{$previewStr}])";
    }

    // ── Internal Helpers ──────────────────────────────────────────────────

    private function flatIndex(array $indices): int
    {
        if (count($indices) !== count($this->shape)) {
            throw new \InvalidArgumentException(
                'Index count ' . count($indices) . ' does not match ndim ' . count($this->shape)
            );
        }
        $flat = 0;
        foreach ($this->strides as $dim => $stride) {
            $flat += $indices[$dim] * $stride;
        }
        return $flat;
    }

    /**
     * Iterative post-order DFS to build a topological ordering of the graph.
     *
     * We use an explicit stack to avoid PHP call-stack limits on deep graphs.
     * The SplObjectStorage acts as a visited-set with O(1) identity lookups.
     *
     * Post-order: a node is appended to $topo only AFTER all its children
     * have been visited.  Reversing $topo then gives a valid reverse-topo
     * order for the backward pass.
     *
     * @param Tensor[]              $topo    Output: nodes in post-order
     * @param \SplObjectStorage     $visited Set of already-processed nodes
     */
    private static function buildTopo(self $root, array &$topo, \SplObjectStorage $visited): void
    {
        // Each stack entry: [Tensor $node, int $childIndex]
        // We process children one at a time so we can resume after recursing.
        $stack = [[$root, 0]];

        while (!empty($stack)) {
            /** @var array{0:Tensor,1:int} $frame */
            $frame  = &$stack[count($stack) - 1];
            $node   = $frame[0];
            $childI = $frame[1];

            if (!$visited->contains($node) && $childI < count($node->_prev)) {
                // Advance to the next unvisited child
                $frame[1]++;
                $child = $node->_prev[$childI];
                if (!$visited->contains($child)) {
                    $stack[] = [$child, 0];
                }
            } else {
                // All children done — finalise this node
                array_pop($stack);
                if (!$visited->contains($node)) {
                    $visited->attach($node);
                    $topo[] = $node;
                }
            }
        }
    }

    private static function computeStrides(array $shape): array
    {
        $ndim    = count($shape);
        $strides = array_fill(0, $ndim, 1);
        for ($i = $ndim - 2; $i >= 0; $i--) {
            $strides[$i] = $strides[$i + 1] * $shape[$i + 1];
        }
        return $strides;
    }

    private static function flattenNested(array $data): array
    {
        $result = [];
        array_walk_recursive($data, function($v) use (&$result) {
            $result[] = (float)$v;
        });
        return $result;
    }

    private static function inferShape(array $data): array
    {
        $shape = [];
        $node  = $data;
        while (is_array($node)) {
            $shape[] = count($node);
            $node    = $node[0] ?? null;
        }
        return $shape;
    }

    private static function nestArray(array $flat, array $shape): array
    {
        if (count($shape) === 1) return array_slice($flat, 0, $shape[0]);
        $chunk = (int)(count($flat) / $shape[0]);
        $rest  = array_slice($shape, 1);
        $out   = [];
        for ($i = 0; $i < $shape[0]; $i++) {
            $out[] = self::nestArray(array_slice($flat, $i * $chunk, $chunk), $rest);
        }
        return $out;
    }
}