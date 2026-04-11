<?php

declare(strict_types=1);

namespace Pml\Benchmarks;

use PhpBench\Attributes as Bench;
use Pml\Tensor;

#[Bench\Groups(['tensor', 'ffi', 'memory'])]
final class TensorBench
{
    private array $large1D;
    private array $large2D;
    private array $deep3D;
    
    private Tensor $matrixA;
    private Tensor $matrixB;

    public function __construct()
    {
        // 1,000,000 elements
        $this->large1D = array_fill(0, 1000000, 1.5);
        
        // 1000 x 1000 elements (1,000,000 total)
        $this->large2D = array_fill(0, 1000, array_fill(0, 1000, 1.5));
        
        // 100 x 100 x 100 elements (1,000,000 total)
        $this->deep3D = array_fill(0, 100, array_fill(0, 100, array_fill(0, 100, 1.5)));

        // Pre-allocate matrices for OpenBLAS operations
        $this->matrixA = Tensor::fromArray(array_fill(0, 500, array_fill(0, 500, 2.0)));
        $this->matrixB = Tensor::fromArray(array_fill(0, 500, array_fill(0, 500, 3.0)));
    }

    // 1. Adjust the assertion to a realistic 25ms threshold for userland FFI writes
    #[Bench\Revs(10)]
    #[Bench\Iterations(5)]
    #[Bench\Assert('mode(variant.time.avg) < 25ms')] 
    public function benchFlatArraySplIteratorConversion(): void
    {
        Tensor::fromArray($this->large1D);
    }

    #[Bench\Revs(10)]
    #[Bench\Iterations(5)]
    #[Bench\Assert('mode(variant.mem.peak) < 100mb')]
    public function bench2DMatrixSplIteratorConversion(): void
    {
        Tensor::fromArray($this->large2D);
    }

    #[Bench\Revs(10)]
    #[Bench\Iterations(5)]
    public function bench3DTensorSplIteratorConversion(): void
    {
        Tensor::fromArray($this->deep3D);
    }

    // 2. Add Warmup iterations to stabilize the CPU cache and C library variance
    #[Bench\Revs(5)]
    #[Bench\Iterations(5)]
    #[Bench\Warmup(2)] 
    #[Bench\Groups(['openblas', 'math'])]
    public function benchOpenBLASMatrixMultiplication(): void
    {
        $this->matrixA->matmul($this->matrixB);
    }
}