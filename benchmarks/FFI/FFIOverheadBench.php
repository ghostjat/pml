<?php
declare(strict_types=1);

namespace Pml\Benchmarks\FFI;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\Lib\TensorEngine;

/**
 * FFI boundary crossing overhead benchmarks.
 *
 * Measures how much time is spent in PHP→C dispatch versus actual compute.
 * Compare tiny-tensor vs large-tensor versions of the same op — the delta
 * is the compute; the tiny version isolates FFI dispatch cost.
 *
 * Groups:
 *   ffi         — all FFI overhead benchmarks
 *   dispatch    — pure boundary crossing (trivial ops)
 *   amortized   — large-work ops where FFI cost is amortized
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['ffi', 'overhead'])]
final class FFIOverheadBench
{
    private static Tensor $scalar;
    private static Tensor $vec8;
    private static Tensor $vec1k;
    private static Tensor $vec1M;
    private static Tensor $mat32x32;
    private static Tensor $mat512x512;
    private static Tensor $mat1kx1k;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }
        self::$scalar     = Tensor::randomNormal([1]);
        self::$vec8       = Tensor::randomNormal([8]);
        self::$vec1k      = Tensor::randomNormal([1000]);
        self::$vec1M      = Tensor::randomNormal([1_000_000]);
        self::$mat32x32   = Tensor::randomNormal([32, 32]);
        self::$mat512x512 = Tensor::randomNormal([512, 512]);
        self::$mat1kx1k   = Tensor::randomNormal([1000, 1000]);
        self::$initialized = true;
    }

    // =========================================================================
    // DISPATCH COST — trivial work, almost all time is FFI boundary
    // =========================================================================

    #[Bench\Iterations(10), Bench\Revs(200)]
    #[Bench\Groups(['ffi', 'dispatch'])]
    public function benchFfiScalarSum(): void
    {
        self::$scalar->sum();
    }

    #[Bench\Iterations(10), Bench\Revs(200)]
    #[Bench\Groups(['ffi', 'dispatch'])]
    public function benchFfiScalarSigmoidInplace(): void
    {
        // vec8 = 8 floats — all time is dispatch + setup, not AVX throughput
        self::$vec8->sigmoidInplace();
    }

    #[Bench\Iterations(10), Bench\Revs(200)]
    #[Bench\Groups(['ffi', 'dispatch'])]
    public function benchFfiScalarReshape(): void
    {
        $r = self::$vec8->reshape(2, 4);
        unset($r);
    }

    // =========================================================================
    // SCALING — compare same op at 1k / 1M elements; slope = compute cost
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(100)]
    #[Bench\Groups(['ffi', 'amortized'])]
    public function benchSigmoid1k(): void
    {
        $r = self::$vec1k->sigmoid();
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(50)]
    #[Bench\Groups(['ffi', 'amortized'])]
    public function benchSigmoid1M(): void
    {
        $r = self::$vec1M->sigmoid();
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(100)]
    #[Bench\Groups(['ffi', 'amortized'])]
    public function benchAdd1k(): void
    {
        $r = self::$vec1k->add(self::$vec1k);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(50)]
    #[Bench\Groups(['ffi', 'amortized'])]
    public function benchAdd1M(): void
    {
        $r = self::$vec1M->add(self::$vec1M);
        unset($r);
    }

    // =========================================================================
    // MATMUL SCALING — FFI cost ~constant; BLAS cost O(N^3)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(50)]
    #[Bench\Groups(['ffi', 'amortized'])]
    public function benchMatmul32x32(): void
    {
        $r = self::$mat32x32->matmul(self::$mat32x32);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['ffi', 'amortized'])]
    public function benchMatmul512x512(): void
    {
        $r = self::$mat512x512->matmul(self::$mat512x512);
        unset($r);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['ffi', 'amortized'])]
    public function benchMatmul1kx1k(): void
    {
        $r = self::$mat1kx1k->matmul(self::$mat1kx1k);
        unset($r);
    }

    // =========================================================================
    // ACCESSOR OVERHEAD — shape/size queries are pure PHP metadata reads
    // =========================================================================

    #[Bench\Iterations(10), Bench\Revs(500)]
    #[Bench\Groups(['ffi', 'dispatch'])]
    public function benchShapeQuery(): void
    {
        self::$mat512x512->shape();
    }

    #[Bench\Iterations(10), Bench\Revs(500)]
    #[Bench\Groups(['ffi', 'dispatch'])]
    public function benchSizeQuery(): void
    {
        self::$mat512x512->size();
    }

    #[Bench\Iterations(10), Bench\Revs(500)]
    #[Bench\Groups(['ffi', 'dispatch'])]
    public function benchNdimQuery(): void
    {
        self::$mat512x512->ndim();
    }
}
