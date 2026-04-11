<?php

declare(strict_types=1);

namespace Pml\Benchmarks;

use PhpBench\Attributes as Bench;
use Pml\Tensor;

/**
 * Full performance profile of the PML Tensor engine.
 * Compares FFI + OpenBLAS + AVX2 + NLP Lookups throughput.
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Warmup(1)]
#[Bench\Revs(5)]
#[Bench\Iterations(3)]
final class FullTensorBench
{
    private Tensor $vectorA;
    private Tensor $vectorB;
    
    private Tensor $matrixA;
    private Tensor $matrixB;
    private Tensor $matrixC; 
    
    private Tensor $broadMatrix;
    private Tensor $broadVector;

    private Tensor $cnnInput;
    private Tensor $cnnWeights;

    private Tensor $nlpTokens;
    private Tensor $nlpWeights;

    public function setUp(): void
    {
        // Basic Math
        $this->vectorA = Tensor::randomUniform([1000000], -1.0, 1.0);
        $this->vectorB = Tensor::randomUniform([1000000], -1.0, 1.0);
        $this->matrixA = Tensor::randomUniform([1000, 1000], -1.0, 1.0);
        $this->matrixB = Tensor::randomUniform([1000, 1000], -1.0, 1.0);

        // Positive Definite Matrix for Decompositions
        $temp = Tensor::randomNormal([500, 1000], 0.0, 1.0);
        $tempT = $temp->transpose()->copy(); 
        $this->matrixC = $temp->matmul($tempT);

        // Broadcasting Targets
        $this->broadMatrix = Tensor::randomUniform([1000, 1000], 0.1, 1.0);
        $this->broadVector = Tensor::randomUniform([1000], 0.1, 1.0);

        // CNN Setup: Batch=16, Channels=3, 224x224 Image
        $this->cnnInput = Tensor::randomNormal([16, 3, 224, 224]);
        $this->cnnWeights = Tensor::randomNormal([64, 3, 3, 3]);

        // LLM Setup: Batch=32, Seq_Len=128, Vocab=30,000, Dim=768
        $tokens = [];
        for ($i = 0; $i < 32; $i++) {
            $seq = [];
            for ($j = 0; $j < 128; $j++) $seq[] = rand(0, 29999);
            $tokens[] = $seq;
        }
        $this->nlpTokens = Tensor::fromArray($tokens, Tensor::DTYPE_INT32);
        $this->nlpWeights = Tensor::randomNormal([30000, 768], 0.0, 1.0);
    }

    #[Bench\Groups(['unary', 'math'])]
    #[Bench\Assert('mode(variant.time.avg) < 15ms')]
    public function benchExpVector1M(): void
    {
        $this->vectorA->exp();
    }

    #[Bench\Groups(['unary', 'math'])]
    public function benchSinVector1M(): void
    {
        $this->vectorA->sin();
    }

    #[Bench\Groups(['unary', 'math'])]
    public function benchSigmoidVector1M(): void
    {
        $this->vectorA->sigmoid();
    }

    // ========================================================================
    // 2. BINARY MATH & BROADCASTING
    // ========================================================================
    #[Bench\Groups(['binary', 'math'])]
    #[Bench\Assert('mode(variant.time.avg) < 15ms')] 
    public function benchAddVectors1M(): void
    {
        $this->vectorA->add($this->vectorB);
    }

    #[Bench\Groups(['binary', 'math', 'inplace'])]
    public function benchAddVectorsInplace1M(): void
    {
        // Demonstrates zero-allocation speed
        $this->vectorA->addInplace($this->vectorB);
    }

    #[Bench\Groups(['binary', 'broadcasting'])]
    public function benchBroadcastMatrixAddVector(): void
    {
        // Adds a [1000] vector to a [1000, 1000] matrix seamlessly using Stride=0
        $this->broadMatrix->add($this->broadVector);
    }

    // ========================================================================
    // 3. REDUCTIONS (OpenMP Accelerated)
    // ========================================================================
    #[Bench\Groups(['reductions'])]
    #[Bench\Assert('mode(variant.time.avg) < 2ms')]
    public function benchSumMatrix1M(): void
    {
        $this->matrixA->sum();
    }

    #[Bench\Groups(['reductions'])]
    public function benchMaxMatrix1M(): void
    {
        $this->matrixA->max();
    }

    #[Bench\Groups(['reductions'])]
    public function benchArgmaxMatrix1M(): void
    {
        $this->matrixA->argmax();
    }

    #[Bench\Groups(['reductions', 'axis'])]
    public function benchSumAxisMatrix(): void
    {
        // Collapse columns to a single row
        $this->matrixA->sumAxis(0);
    }

    // ========================================================================
    // 4. LINEAR ALGEBRA (OpenBLAS / LAPACKE)
    // ========================================================================
    #[Bench\Groups(['linalg', 'cblas'])]
    #[Bench\Assert('mode(variant.time.avg) < 150ms')]
    public function benchMatmul1000x1000(): void
    {
        // O(N^3) Operation natively executed in OpenBLAS
        $this->matrixA->matmul($this->matrixB);
    }

    #[Bench\Groups(['linalg', 'lapack'])]
    public function benchInverse500x500(): void
    {
        $this->matrixC->inverse();
    }

    #[Bench\Groups(['linalg', 'lapack'])]
    public function benchCholeskyDecomp500x500(): void
    {
        $this->matrixC->cholesky();
    }

    #[Bench\Groups(['linalg', 'lapack'])]
    public function benchSVD500x500(): void
    {
        $this->matrixC->svd();
    }

    // ========================================================================
    // 5. DEEP LEARNING (CNN Primitives)
    // ========================================================================
        #[Bench\Groups(['dl', 'cnn'])]
    public function benchIm2ColResNetInput(): void
    {
        // Benchmarks the critical memory unrolling step for CNNs
        // Kernel: 3x3, Stride: 1, Pad: 1
        $this->cnnInput->im2col(3, 3, 1, 1, 1, 1);
    }
    
    #[Bench\Groups(['dl', 'cnn'])]
    public function benchConv2DResNetForward(): void
    {
        // Full Forward Pass of a Convolutional Layer
        $this->cnnInput->conv2d($this->cnnWeights, null, 1, 1, 1, 1);
    }

    #[Bench\Groups(['dl', 'nlp'])]
    public function benchEmbeddingLookupLLM(): void
    {
        // Executes 4,096 integer lookups across a 30k vocabulary matrix
        // and safely copies 3,145,728 floats (12 MB) into the new tensor natively in C
        $this->nlpTokens->embeddingLookup($this->nlpWeights);
    }
}