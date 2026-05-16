<?php
declare(strict_types=1);

namespace Pml\Benchmarks\Memory;

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
 * Memory-focused benchmarks.
 *
 * Measures PHP peak memory, C pool allocator reuse, and lifecycle overhead.
 * PHPBench reports mem_peak per iteration — compare across bench subjects to
 * spot unexpected allocations.
 *
 * Groups:
 *   memory       — all memory benchmarks
 *   lifecycle    — create/use/free cycles (checks for leaks over iterations)
 *   pool         — pool allocator reuse (inplace ops should stay flat)
 *   dataset      — Dataset memory footprint
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['memory'])]
final class MemoryBench
{
    private static bool $initialized = false;
    private static Tensor $mat1kx1k;
    private static Dataset $ds100k;
    private static Sequential $model;
    private static Tensor $batchInput;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }
        self::$mat1kx1k  = Tensor::randomNormal([1000, 1000]);
        self::$ds100k    = new Dataset(Tensor::randomNormal([100_000, 32]));
        self::$batchInput = Tensor::randomNormal([128, 256]);

        self::$model = new Sequential([
            new Dense(256, 512),
            new ReLU(),
            new Dense(512, 512),
            new ReLU(),
            new Dense(512, 10),
            new Softmax(),
        ], new CategoricalCrossEntropy(), new Adam(0.001));

        self::$initialized = true;
    }

    // =========================================================================
    // LIFECYCLE — create and destroy tensors (leak detector: mem_peak should
    // not grow across iterations since we unset() each tensor)
    // =========================================================================

    #[Bench\Iterations(10), Bench\Revs(50)]
    #[Bench\Groups(['memory', 'lifecycle'])]
    public function benchTensorCreateDestroy1M(): void
    {
        $t = Tensor::randomNormal([1_000_000]);
        unset($t);
    }

    #[Bench\Iterations(10), Bench\Revs(20)]
    #[Bench\Groups(['memory', 'lifecycle'])]
    public function benchTensorCreateDestroy512x512(): void
    {
        $t = Tensor::randomNormal([512, 512]);
        unset($t);
    }

    #[Bench\Iterations(10), Bench\Revs(20)]
    #[Bench\Groups(['memory', 'lifecycle'])]
    public function benchTensorCopyDestroy1M(): void
    {
        $t = self::$mat1kx1k->copy();
        unset($t);
    }

    // =========================================================================
    // POOL ALLOCATOR — inplace ops reuse existing C buffers; mem_peak stays flat
    // =========================================================================

    #[Bench\Iterations(10), Bench\Revs(50)]
    #[Bench\Groups(['memory', 'pool'])]
    public function benchInplaceSigmoidNoAlloc(): void
    {
        $t = self::$mat1kx1k->copy();
        $t->sigmoidInplace();
        unset($t);
    }

    #[Bench\Iterations(10), Bench\Revs(50)]
    #[Bench\Groups(['memory', 'pool'])]
    public function benchInplaceReluNoAlloc(): void
    {
        $t = self::$mat1kx1k->copy();
        $t->reluInplace();
        unset($t);
    }

    #[Bench\Iterations(10), Bench\Revs(50)]
    #[Bench\Groups(['memory', 'pool'])]
    public function benchInplaceMulScalarNoAlloc(): void
    {
        $t = self::$mat1kx1k->copy();
        $t->mulScalarInplace(1.001);
        unset($t);
    }

    // =========================================================================
    // ALLOCATING OPS — each call allocates a new buffer; mem_peak grows per rev
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['memory', 'pool'])]
    public function benchAllocatingAdd1M(): void
    {
        $r = self::$mat1kx1k->add(self::$mat1kx1k);
        unset($r);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    #[Bench\Groups(['memory', 'pool'])]
    public function benchAllocatingMatmul1k(): void
    {
        $r = self::$mat1kx1k->matmul(self::$mat1kx1k);
        unset($r);
    }

    // =========================================================================
    // DATASET MEMORY — batch slicing is zero-copy (no duplication of C buffer)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['memory', 'dataset'])]
    public function benchBatchSliceZeroCopy100k(): void
    {
        foreach (self::$ds100k->batches(256) as $batch) {
            // access samples to trigger the slice — should not copy C memory
            $_ = $batch->numRows();
        }
    }

    #[Bench\Iterations(5), Bench\Revs(5)]
    #[Bench\Groups(['memory', 'dataset'])]
    public function benchSplitZeroCopy100k(): void
    {
        [$train, $test] = self::$ds100k->split(0.8);
        unset($train, $test);
    }

    // =========================================================================
    // NEURAL NETWORK — forward pass memory overhead (activations + temporaries)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    #[Bench\Groups(['memory', 'nn'])]
    public function benchNNForwardPassMemory(): void
    {
        $out = self::$model->forward(self::$batchInput);
        unset($out);
    }

    #[Bench\Iterations(5), Bench\Revs(5)]
    #[Bench\Groups(['memory', 'nn'])]
    public function benchNNForwardBackwardMemory(): void
    {
        $labels = Tensor::zeros(128, 10);
        $buf = $labels->buffer();
        for ($i = 0; $i < 128; $i++) {
            $buf[$i * 10 + ($i % 10)] = 1.0;
        }
        $cce = new CategoricalCrossEntropy();
        $out  = self::$model->forward(self::$batchInput);
        $dY   = $cce->differentiate($out, $labels);
        self::$model->backward($dY);
        unset($out, $dY, $labels);
    }
}
