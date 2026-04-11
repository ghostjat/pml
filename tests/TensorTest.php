<?php

declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Tensor;
use InvalidArgumentException;

final class TensorTest extends TestCase
{
    public function testCreatesContiguousBufferFromFlatArray(): void
    {
        $data = [1.0, 2.0, 3.0, 4.0];
        $tensor = Tensor::fromArray($data);

        $this->assertSame([4], $tensor->shape());
        $this->assertSame(4, $tensor->size());
        $this->assertSame(Tensor::DTYPE_FLOAT32, $tensor->dtype());
        
        $flat = $tensor->toFlatArray();
        foreach ($data as $i => $val) {
            $this->assertEqualsWithDelta($val, $flat[$i], 0.0001);
        }
    }

    public function testRejectsIrregularJaggedArrays(): void
    {
        $this->expectException(InvalidArgumentException::class);

        $jaggedData = [
            [1.0, 2.0],
            [3.0] // Missing element destroys the uniform tensor shape
        ];
        
        Tensor::fromArray($jaggedData);
    }

    public function testIntegerDTypeForNLPTokenizers(): void
    {
        // Mock token IDs from a tokenizer
        $data = [8725, 291, 1024];
        
        // Instruct the FFI to allocate an INT32 tensor
        $tensor = Tensor::fromArray($data, Tensor::DTYPE_INT32);
        
        $this->assertSame(Tensor::DTYPE_INT32, $tensor->dtype());
        $this->assertSame([3], $tensor->shape());
        
        // Ensure the binary string unpacks natively back into exact integers
        $flat = $tensor->toFlatArray();
        $this->assertSame(8725, $flat[0]);
        $this->assertSame(291, $flat[1]);
        $this->assertSame(1024, $flat[2]);
    }

    public function testLLMEmbeddingLookupBridging(): void
    {
        // 1. The Tokens (INT32)
        $tokens = Tensor::fromArray([1, 0, 2], Tensor::DTYPE_INT32); // shape [3]
        
        // 2. The Model's Pre-trained Embedding Weights (FLOAT32)
        // Vocab Size: 3, Embedding Dim: 2
        $weights = Tensor::fromArray([
            [0.1, 0.2], // Token 0
            [0.3, 0.4], // Token 1
            [0.5, 0.6]  // Token 2
        ], Tensor::DTYPE_FLOAT32);
        
        // 3. Execute the lookup
        $embeddings = $tokens->embeddingLookup($weights);
        
        // Assert the output is a 3x2 Matrix of floats
        $this->assertSame([3, 2], $embeddings->shape());
        $this->assertSame(Tensor::DTYPE_FLOAT32, $embeddings->dtype());
        
        $flat = $embeddings->toFlatArray();
        
        // token 1 -> [0.3, 0.4]
        $this->assertEqualsWithDelta(0.3, $flat[0], 0.0001);
        $this->assertEqualsWithDelta(0.4, $flat[1], 0.0001);
        
        // token 0 -> [0.1, 0.2]
        $this->assertEqualsWithDelta(0.1, $flat[2], 0.0001);
        $this->assertEqualsWithDelta(0.2, $flat[3], 0.0001);
        
        // token 2 -> [0.5, 0.6]
        $this->assertEqualsWithDelta(0.5, $flat[4], 0.0001);
        $this->assertEqualsWithDelta(0.6, $flat[5], 0.0001);
    }
}