<?php

declare(strict_types=1);

namespace Pml;

use Pml\Lib\TensorEngine;

/**
 * INT8 symmetric block-quantized weight matrix.
 *
 * Wraps the C QuantizedWeight struct, which stores:
 *   - data:   [rows × cols]  int8_t  (64-byte aligned)
 *   - scales: [rows × num_groups] float32  (one scale per group of cols)
 *
 * Memory reduction vs fp32: ~4× for group_size=32, exact for group_size=cols.
 *
 * Usage:
 *   $qw = QuantizedTensor::fromTensor($weights, groupSize: 32);
 *   $y  = $qw->linear($x, $bias);  // X [batch, cols] → Y [batch, rows]
 *   $w  = $qw->toTensor();          // dequantize back to fp32 (for export)
 *
 * Created by Dense::quantize() — do not construct directly unless you control
 * the QuantizedWeight* lifetime.
 */
final class QuantizedTensor
{
    /** @var \FFI\CData  QuantizedWeight* */
    public readonly \FFI\CData $ptr;

    private function __construct(\FFI\CData $ptr)
    {
        $this->ptr = $ptr;
    }

    /**
     * Quantize a fp32 [rows, cols] Tensor to INT8 block format.
     *
     * @param Tensor $w         Source weight matrix [rows, cols] fp32
     * @param int    $groupSize Elements per quantization group (32 = Q8_0-class)
     */
    public static function fromTensor(Tensor $w, int $groupSize = 32): self
    {
        $ffi = TensorEngine::get();
        $ptr = $ffi->qweight_quantize($w->ptr, $groupSize);
        if ($ffi->tensor_check_error()) {
            $msg = \FFI::string($ffi->tensor_get_last_error());
            $ffi->tensor_clear_error();
            throw new \RuntimeException("QuantizedTensor::fromTensor — {$msg}");
        }
        if (\FFI::isNull($ptr)) {
            throw new \RuntimeException('QuantizedTensor::fromTensor — qweight_quantize returned NULL');
        }
        return new self($ptr);
    }

    /**
     * Dequantize back to a fp32 Tensor [rows, cols].
     * Allocates a new Tensor — use only for checkpoint export, not in hot paths.
     */
    public function toTensor(): Tensor
    {
        $ffi = TensorEngine::get();
        $ptr = $ffi->qweight_dequantize($this->ptr);
        if ($ffi->tensor_check_error()) {
            $msg = \FFI::string($ffi->tensor_get_last_error());
            $ffi->tensor_clear_error();
            throw new \RuntimeException("QuantizedTensor::toTensor — {$msg}");
        }
        if (\FFI::isNull($ptr)) {
            throw new \RuntimeException('QuantizedTensor::toTensor — NULL result');
        }
        return Tensor::wrap($ptr);
    }

    /**
     * Quantized linear forward: Y = X @ W^T + bias.
     *
     * Hot path for LLM decode (batch=1): AVX2 fused int8→fp32 dot product,
     * OpenMP-parallel over output rows — no temporary fp32 weight allocation.
     *
     * @param  Tensor      $X    [batch, cols] or [cols] input activations (fp32)
     * @param  Tensor|null $bias [rows] bias vector (fp32, optional)
     * @return Tensor            [batch, rows] output (new fp32 Tensor)
     */
    public function linear(Tensor $X, ?Tensor $bias = null): Tensor
    {
        $ffi  = TensorEngine::get();
        $bPtr = $bias !== null ? $bias->ptr : null;
        $ptr  = $ffi->qweight_linear($X->ptr, $this->ptr, $bPtr);
        if ($ffi->tensor_check_error()) {
            $msg = \FFI::string($ffi->tensor_get_last_error());
            $ffi->tensor_clear_error();
            throw new \RuntimeException("QuantizedTensor::linear — {$msg}");
        }
        if (\FFI::isNull($ptr)) {
            throw new \RuntimeException('QuantizedTensor::linear — NULL result');
        }
        return Tensor::wrap($ptr);
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    public function rows(): int      { return (int)$this->ptr->rows;       }
    public function cols(): int      { return (int)$this->ptr->cols;       }
    public function groupSize(): int { return (int)$this->ptr->group_size; }
    public function numGroups(): int { return (int)$this->ptr->num_groups; }

    /**
     * Bytes used by int8 data + scale buffers (excludes struct header).
     * Compare with rows() × cols() × 4 to see the compression ratio.
     */
    public function memoryBytes(): int
    {
        return (int) TensorEngine::get()->qweight_memory($this->ptr);
    }

    /**
     * Compression ratio vs an equivalent fp32 matrix.
     */
    public function compressionRatio(): float
    {
        $fp32Bytes = $this->rows() * $this->cols() * 4;
        $qBytes    = $this->memoryBytes();
        return $fp32Bytes > 0 ? $fp32Bytes / $qBytes : 1.0;
    }

    public function __destruct()
    {
        TensorEngine::get()->qweight_free($this->ptr);
    }
}
