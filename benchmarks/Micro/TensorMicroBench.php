<?php
declare(strict_types=1);

namespace Pml\Benchmarks\Micro;

use PhpBench\Attributes as Bench;
use Pml\Tensor;

#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['micro', 'tensor', 'ffi'])]
final class TensorMicroBench
{
    private static Tensor $vec1M;
    private static Tensor $mat512;
    private static Tensor $matA;
    private static Tensor $matB;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }

        self::$vec1M = Tensor::randomNormal([1_000_000]);
        self::$mat512 = Tensor::randomNormal([512, 512]);
        self::$matA = Tensor::randomNormal([256, 256]);
        self::$matB = Tensor::randomNormal([256, 256]);
        self::$initialized = true;
    }

    #[Bench\Iterations(3), Bench\Revs(20)]
    public function benchAdd1M(): void
    {
        $result = self::$vec1M->add(self::$vec1M);
        unset($result);
    }

    #[Bench\Iterations(3), Bench\Revs(20)]
    public function benchMul1M(): void
    {
        $result = self::$vec1M->mul(self::$vec1M);
        unset($result);
    }

    #[Bench\Iterations(3), Bench\Revs(20)]
    public function benchMatmul256(): void
    {
        $result = self::$matA->matmul(self::$matB);
        unset($result);
    }

    #[Bench\Iterations(3), Bench\Revs(20)]
    public function benchRelu1M(): void
    {
        $result = self::$vec1M->relu();
        unset($result);
    }
}