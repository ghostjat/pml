<?php
declare(strict_types=1);

namespace Pml\Benchmarks\JIT;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\Dataset;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Layers\Softmax;
use Pml\Losses\CategoricalCrossEntropy;
use Pml\NeuralNetwork\Optimizers\Adam;

/**
 * PHP JIT-sensitive benchmarks.
 *
 * These benchmarks stress the PHP interpreter paths rather than the C engine:
 *   - PHP loops that dispatch many FFI calls
 *   - Dataset iteration (Generator + PHP loop over batches)
 *   - Tensor-to-PHP array conversions (toFlatArray is pure PHP iteration)
 *
 * Run without JIT: php -d opcache.jit=0 vendor/bin/phpbench run benchmarks/JIT ...
 * Run with JIT:    php -d opcache.jit=1255 vendor/bin/phpbench run benchmarks/JIT ...
 *
 * Groups:
 *   jit         — all JIT-sensitive benchmarks
 *   loop        — PHP generator / batch iteration loops
 *   dispatch    — many small FFI calls per iteration
 *   conversion  — PHP ↔ C data marshaling
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['jit'])]
final class JITBench
{
    private static Dataset $ds10k;
    private static Dataset $ds100k;
    private static Tensor $mat1kx1k;
    private static Tensor $vec100k;
    private static Sequential $model;
    private static Tensor $batchInput;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }
        self::$ds10k    = new Dataset(Tensor::randomNormal([10_000, 32]), Tensor::randomNormal([10_000]));
        self::$ds100k   = new Dataset(Tensor::randomNormal([100_000, 16]));
        self::$mat1kx1k = Tensor::randomNormal([1000, 1000]);
        self::$vec100k  = Tensor::randomNormal([100_000]);
        self::$batchInput = Tensor::randomNormal([64, 64]);

        self::$model = new Sequential([
            new Dense(64, 128),
            new ReLU(),
            new Dense(128, 32),
            new ReLU(),
            new Dense(32, 4),
            new Softmax(),
        ], new CategoricalCrossEntropy(), new Adam(0.001));

        self::$initialized = true;
    }

    // =========================================================================
    // BATCH ITERATION — PHP Generator loop is JIT-accelerated
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    #[Bench\Groups(['jit', 'loop'])]
    public function benchIterateBatches32Over10k(): void
    {
        $total = 0;
        foreach (self::$ds10k->batches(32) as $batch) {
            $total += $batch->numRows();
        }
    }

    #[Bench\Iterations(5), Bench\Revs(5)]
    #[Bench\Groups(['jit', 'loop'])]
    public function benchIterateBatches32Over100k(): void
    {
        $total = 0;
        foreach (self::$ds100k->batches(32) as $batch) {
            $total += $batch->numRows();
        }
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    #[Bench\Groups(['jit', 'loop'])]
    public function benchIterateBatches256Over10k(): void
    {
        $total = 0;
        foreach (self::$ds10k->batches(256) as $batch) {
            $total += $batch->numRows();
        }
    }

    // =========================================================================
    // MULTIPLE FFI DISPATCH IN PHP LOOP — JIT removes per-iteration overhead
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    #[Bench\Groups(['jit', 'dispatch'])]
    public function benchPhpLoopMatrixRowAccess(): void
    {
        $n = self::$mat1kx1k->shape()[0];
        $sum = 0.0;
        for ($i = 0; $i < $n; $i += 50) {
            $row = self::$mat1kx1k->row($i);
            $sum += $row->sum();
        }
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['jit', 'dispatch'])]
    public function benchPhpLoopTensorChainOps(): void
    {
        // Chain of small FFI calls in a PHP loop — JIT benefits
        $t = self::$vec100k->copy();
        $t->addScalarInplace(0.1);
        $t->mulScalarInplace(0.99);
        $t->sigmoidInplace();
        $_ = $t->mean();
        unset($t);
    }

    // =========================================================================
    // DATA MARSHALING — toFlatArray traverses C memory into PHP arrays
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['jit', 'conversion'])]
    public function benchToFlatArray100k(): void
    {
        $arr = self::$vec100k->toFlatArray();
        unset($arr);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['jit', 'conversion'])]
    public function benchToFlatArray1kx1k(): void
    {
        $arr = self::$mat1kx1k->toFlatArray();
        unset($arr);
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['jit', 'conversion'])]
    public function benchFromArray10kx32(): void
    {
        $rows = \array_fill(0, 10_000, \array_fill(0, 32, 1.5));
        $t = Tensor::fromArray($rows);
        unset($t, $rows);
    }

    // =========================================================================
    // NEURAL NETWORK LOOP — epoch over batches (JIT speeds up PHP orchestration)
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['jit', 'loop'])]
    public function benchNNForwardBatchLoop100(): void
    {
        for ($i = 0; $i < 100; $i++) {
            $out = self::$model->forward(self::$batchInput);
            unset($out);
        }
    }
}
