<?php 

namespace Pml;

class Tensor
{
    public \FFI\CData $buffer;
    public readonly array $shape;
    public readonly int $size;

    public function __construct(array $shape, ?\FFI\CData $buffer = null)
    {
        $this->shape = $shape;
        $this->size = array_product($shape);
        $ffi = BlasEngine::get()->ffi;

        if ($buffer !== null) {
            $this->buffer = $buffer;
        } else {
            // $owned = true: PHP's GC will automatically free the C memory when Tensor is destroyed
            $this->buffer = $ffi->new("float[{$this->size}]", true);
        }
    }

    /**
     * Rapidly loads a flat PHP array into the C buffer using binary packing.
     * Orders of magnitude faster than a foreach loop over FFI indices.
     */
    public static function fromArray(array $data, array $shape): self
    {
        $tensor = new self($shape);
        
        // 'f*' packs into machine-byte-order single-precision floats (32-bit)
        $binaryString = pack('f*', ...$data); 
        
        // Copy raw bytes directly into the C struct (4 bytes per float32)
        \FFI::memcpy($tensor->buffer, $binaryString, $tensor->size * 4);
        
        return $tensor;
    }

    /**
     * Plucks a specific row out of a 2D tensor (e.g., getting a token embedding)
     * Returns a NEW Tensor, but uses FFI pointer arithmetic to avoid copying 
     * if we are just reading. For safety in an ML loop, we usually copy.
     */
    public function getRow(int $index): self
    {
        if (count($this->shape) !== 2) throw new \Exception("getRow only supports 2D Tensors");
        
        $cols = $this->shape[1];
        $rowTensor = new self([$cols]);
        $ffi = BlasEngine::get()->ffi;
        
        // Pointer arithmetic to find the start of the row
        $sourcePtr = \FFI::cast("float*", \FFI::addr($this->buffer[$index * $cols]));
        
        // scopy(N, X, incX, Y, incY)
        $ffi->cblas_scopy($cols, $sourcePtr, 1, $rowTensor->buffer, 1);
        
        return $rowTensor;
    }
    
    public function fill(float $val): void
    {
        for ($i = 0; $i < $this->size; $i++) {
            $this->buffer[$i] = $val;
        }
    }
}