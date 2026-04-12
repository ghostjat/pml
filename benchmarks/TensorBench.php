<?php
declare(strict_types=1);

namespace Pml\Benchmarks;

use PhpBench\Attributes as Bench;
use Pml\Tensor;

/**
 * Comprehensive Tensor performance benchmarks.
 *
 * Measures: FLOPS throughput, memory bandwidth, linalg latency,
 * aggregation speed, and SIMD path effectiveness.
 *
 * Run with:
 *   vendor/bin/phpbench run benchmarks/TensorBench.php --report=aggregate
 */
#[Bench\Groups(['tensor', 'ffi', 'simd', 'linalg'])]
final class TensorBench
{
    // Pre-allocated shared tensors (avoid constructor cost inside hot loops)
    private Tensor $vec1M;
    private Tensor $mat512;
    private Tensor $matA;
    private Tensor $matB;
    private Tensor $mat100;
    private Tensor $batch;
    private Tensor $vec1k;

    public function __construct()
    {
        $this->vec1M  = Tensor::randomNormal([1_000_000]);
        $this->mat512 = Tensor::randomNormal([512, 512]);
        $this->matA   = Tensor::randomNormal([256, 256]);
        $this->matB   = Tensor::randomNormal([256, 256]);
        $this->mat100 = Tensor::randomNormal([100, 100]);
        $this->batch  = Tensor::randomNormal([1000, 128]);
        $this->vec1k  = Tensor::randomNormal([1000]);
    }

    // =========================================================================
    // CREATION
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchFromArray1M(): void
    {
        $data = \array_fill(0, 1_000_000, 1.5);
        $t = Tensor::fromArray($data);
        unset($t);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchRandomNormal1M(): void
    {
        $t = Tensor::randomNormal([1_000_000]);
        unset($t);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchZeros512x512(): void
    {
        $t = Tensor::zeros(512, 512);
        unset($t);
    }

    // =========================================================================
    // ELEMENT-WISE MATH (SIMD / AVX2)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchAdd1M(): void
    {
        $r = $this->vec1M->add($this->vec1M);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchMul1M(): void
    {
        $r = $this->vec1M->mul($this->vec1M);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchAddScalar1M(): void
    {
        $r = $this->vec1M->addScalar(1.0);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchMulScalarInplace1M(): void
    {
        // in-place avoids allocation — measures pure AVX2 throughput
        $t = $this->vec1M->copy();
        $t->mulScalarInplace(2.0);
        unset($t);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchDiv1M(): void
    {
        $r = $this->vec1M->div($this->vec1M);
        unset($r);
    }

    // =========================================================================
    // UNARY MATH (SIMD)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchRelu1M(): void
    {
        $r = $this->vec1M->relu();
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchSigmoid1M(): void
    {
        $r = $this->vec1M->sigmoid();
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchExp1M(): void
    {
        $r = $this->vec1M->exp();
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchSqrt1M(): void
    {
        $r = $this->vec1M->sqrt();
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchLog1M(): void
    {
        $t = Tensor::randomUniform([1_000_000], 0.01, 10.0);
        $r = $t->log();
        unset($t, $r);
    }

    // =========================================================================
    // AGGREGATIONS
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(50)]
    public function benchSum1M(): void
    {
        $this->vec1M->sum();
    }

    #[Bench\Iterations(5), Bench\Revs(50)]
    public function benchMean1M(): void
    {
        $this->vec1M->mean();
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchVarianceStd1M(): void
    {
        $this->vec1M->variance();
        $this->vec1M->std();
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchSumAxis512(): void
    {
        $r = $this->mat512->sumAxis(0);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchMeanAxis512(): void
    {
        $r = $this->mat512->meanAxis(1);
        unset($r);
    }

    // =========================================================================
    // SHAPE OPS
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(50)]
    public function benchReshape1M(): void
    {
        $r = $this->vec1M->reshape(1000, 1000);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(50)]
    public function benchTranspose512(): void
    {
        $r = $this->mat512->transpose();
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(50)]
    public function benchFlatten512(): void
    {
        $r = $this->mat512->flatten();
        unset($r);
    }

    // =========================================================================
    // LINEAR ALGEBRA (OpenBLAS)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchMatmul256(): void
    {
        $r = $this->matA->matmul($this->matB);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(5)]
    public function benchMatmul512(): void
    {
        $r = $this->mat512->matmul($this->mat512);
        unset($r);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchInverse100(): void
    {
        $r = $this->mat100->inverse();
        unset($r);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchSvd100(): void
    {
        $svd = $this->mat100->svd();
        unset($svd);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchPinv100(): void
    {
        $r = $this->mat100->pinv();
        unset($r);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchBmm32x64x64(): void
    {
        $a = Tensor::randomNormal([32, 64, 64]);
        $b = Tensor::randomNormal([32, 64, 64]);
        $r = $a->bmm($b);
        unset($a, $b, $r);
    }

    // =========================================================================
    // SORTING & TOPK
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchSort1k(): void
    {
        $r = $this->vec1k->sort(0);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchArgsort1k(): void
    {
        $r = $this->vec1k->argsort(0);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchTopk100of1k(): void
    {
        $r = $this->vec1k->topk(100, 0);
        unset($r);
    }

    // =========================================================================
    // FANCY INDEXING
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchBooleanIndex(): void
    {
        $mask = $this->vec1k->greaterScalar(0.0);
        $r    = $this->vec1k->booleanIndex($mask);
        unset($mask, $r);
    }

    // =========================================================================
    // CONCAT / PAD
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchConcatBatch(): void
    {
        $parts = [];
        for ($i = 0; $i < 10; $i++) {
            $parts[] = Tensor::randomNormal([100, 128]);
        }
        $r = Tensor::concat($parts, 0);
        unset($parts, $r);
    }

    // =========================================================================
    // MEMORY BANDWIDTH: copy vs view
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchCopy512(): void
    {
        $r = $this->mat512->copy();
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(50)]
    public function benchView512(): void
    {
        $r = $this->mat512->view();
        unset($r);
    }

    // =========================================================================
    // I/O SERIALIZATION
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchSaveLoad512(): void
    {
        $path = \sys_get_temp_dir() . '/bench_tensor.bin';
        $this->mat512->save($path);
        $loaded = Tensor::load($path);
        unset($loaded);
        @\unlink($path);
    }

    // =========================================================================
    // FUSED KERNELS
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchFusedBceLossAndGrad(): void
    {
        $preds   = Tensor::randomUniform([1000], 0.01, 0.99);
        $targets = Tensor::randomUniform([1000], 0.0,  1.0)->round();
        $grads   = Tensor::zeros(1000);
        Tensor::fusedBceLossAndGrad($preds, $targets, $grads);
        unset($preds, $targets, $grads);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchFusedAdamStep(): void
    {
        $param = Tensor::randomNormal([1000]);
        $grad  = Tensor::randomNormal([1000]);
        $m     = Tensor::zeros(1000);
        $v     = Tensor::zeros(1000);
        Tensor::fusedAdamStep($param, $grad, $m, $v, 0.001, 0.9, 0.999, 1e-8, 1);
        unset($param, $grad, $m, $v);
    }
}
