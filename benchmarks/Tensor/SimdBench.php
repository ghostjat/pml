<?php
declare(strict_types=1);

namespace Pml\Benchmarks\Tensor;

use PhpBench\Attributes as Bench;
use Pml\Tensor;

/**
 * SIMD path benchmarks — exercises the AVX2/AVX512 hot paths added during
 * the 2026-04-19 performance sprint: softmax, activations, sum_axis 2D fast paths.
 *
 * Groups:
 *   simd       — all AVX2/AVX512 ops
 *   softmax    — numerically-stable row softmax
 *   activations — sigmoid / tanh / relu / exp / sqrt
 *   reductions  — sum_axis axis=0 / axis=1 2D fast paths
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['simd', 'tensor', 'avx2'])]
final class SimdBench
{
    private static Tensor $vec1M;
    private static Tensor $vecPos1M;
    private static Tensor $mat512x512;
    private static Tensor $mat1kx128;
    private static Tensor $mat128x1k;
    private static Tensor $logits1kx1k;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }
        self::$vec1M       = Tensor::randomNormal([1_000_000]);
        self::$vecPos1M    = Tensor::randomUniform([1_000_000], 0.01, 5.0);
        self::$mat512x512  = Tensor::randomNormal([512, 512]);
        self::$mat1kx128   = Tensor::randomNormal([1000, 128]);
        self::$mat128x1k   = Tensor::randomNormal([128, 1000]);
        self::$logits1kx1k = Tensor::randomNormal([1000, 1000]);
        self::$initialized = true;
    }

    // =========================================================================
    // SOFTMAX (numerically-stable row softmax — AVX2/AVX512 4-wide unrolled)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['softmax'])]
    public function benchRowSoftmax1kx1k(): void
    {
        $t = self::$logits1kx1k->copy();
        $t->rowSoftmaxInplace();
        unset($t);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['softmax'])]
    public function benchRowSoftmax1kx128(): void
    {
        $t = self::$mat1kx128->copy();
        $t->rowSoftmaxInplace();
        unset($t);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['softmax'])]
    public function benchRowSoftmax512x512(): void
    {
        $t = self::$mat512x512->copy();
        $t->rowSoftmaxInplace();
        unset($t);
    }

    // =========================================================================
    // SIGMOID (explicit AVX2 avx2_sigmoidf)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['activations'])]
    public function benchSigmoidInplace1M(): void
    {
        $t = self::$vec1M->copy();
        $t->sigmoidInplace();
        unset($t);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['activations'])]
    public function benchSigmoidAlloc1M(): void
    {
        $r = self::$vec1M->sigmoid();
        unset($r);
    }

    // =========================================================================
    // TANH (explicit AVX2 avx2_tanhf)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['activations'])]
    public function benchTanhInplace1M(): void
    {
        $t = self::$vec1M->copy();
        $t->tanhInplace();
        unset($t);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['activations'])]
    public function benchTanhAlloc1M(): void
    {
        $r = self::$vec1M->tanh();
        unset($r);
    }

    // =========================================================================
    // RELU (AVX512 + AVX2 paths)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['activations'])]
    public function benchReluInplace1M(): void
    {
        $t = self::$vec1M->copy();
        $t->reluInplace();
        unset($t);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['activations'])]
    public function benchReluAlloc1M(): void
    {
        $r = self::$vec1M->relu();
        unset($r);
    }

    // =========================================================================
    // EXP (explicit AVX2 avx2_expf via Cephes polynomial)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(15)]
    #[Bench\Groups(['activations'])]
    public function benchExpInplace1M(): void
    {
        $t = self::$vec1M->copy();
        $t->expInplace();
        unset($t);
    }

    #[Bench\Iterations(5), Bench\Revs(15)]
    #[Bench\Groups(['activations'])]
    public function benchExpAlloc1M(): void
    {
        $r = self::$vec1M->exp();
        unset($r);
    }

    // =========================================================================
    // SQRT (explicit AVX2 _mm256_sqrt_ps)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['activations'])]
    public function benchSqrtInplace1M(): void
    {
        $t = self::$vecPos1M->copy();
        $t->sqrtInplace();
        unset($t);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['activations'])]
    public function benchSqrtAlloc1M(): void
    {
        $r = self::$vecPos1M->sqrt();
        unset($r);
    }

    // =========================================================================
    // SUM_AXIS 2D fast paths (axis=0: tiled column accumulation; axis=1: AVX2 horizontal)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['reductions'])]
    public function benchSumAxis0_512x512(): void
    {
        $r = self::$mat512x512->sumAxis(0);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['reductions'])]
    public function benchSumAxis1_512x512(): void
    {
        $r = self::$mat512x512->sumAxis(1);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['reductions'])]
    public function benchSumAxis0_1kx128(): void
    {
        $r = self::$mat1kx128->sumAxis(0);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['reductions'])]
    public function benchSumAxis1_1kx128(): void
    {
        $r = self::$mat1kx128->sumAxis(1);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['reductions'])]
    public function benchMeanAxis1_1kx128(): void
    {
        $r = self::$mat1kx128->meanAxis(1);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['reductions'])]
    public function benchMaxAxis1_1kx128(): void
    {
        $r = self::$mat1kx128->maxAxis(1);
        unset($r);
    }
}
