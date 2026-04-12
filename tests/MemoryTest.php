<?php
declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Dataset;
use Pml\Tensor;
use Pml\Lib\TensorEngine;
use Pml\Estimators\Classifiers\DecisionTreeClassifier;
use Pml\Estimators\Regression\LinearRegression;
use Pml\Transformers\StandardScaler;

/**
 * Memory-safety test suite.
 *
 * Strategy: measure peak RSS or PHP memory_get_usage() before and after
 * repeated allocation/free cycles.  Each test verifies:
 * 1. No PHP-side memory leak (memory_get_usage returns close to baseline)
 * 2. No segfault / double-free (reaching the assertion is the proof)
 * 3. C-side objects are freed properly via __destruct()
 *
 * Note: PHP's GC is non-deterministic; we force gc_collect_cycles() before
 * measuring.  The tolerance is generous (2 MB) to account for PHP runtime
 * overhead that is non-tensor-related.
 */
final class MemoryTest extends TestCase
{
    private const LEAK_TOLERANCE_BYTES = 2 * 1024 * 1024;  // 2 MB

    // =========================================================================
    // HELPERS
    // =========================================================================

    private function baseline(): int
    {
        \gc_collect_cycles();
        return \memory_get_usage(true);
    }

    private function assertNoLeak(int $before, string $context): void
    {
        \gc_collect_cycles();
        $after = \memory_get_usage(true);
        $delta = $after - $before;
        $this->assertLessThan(
            self::LEAK_TOLERANCE_BYTES,
            $delta,
            "{$context}: memory grew by " . number_format($delta) . " bytes — possible leak"
        );
    }

    // =========================================================================
    // 1. TENSOR ALLOC / FREE
    // =========================================================================

    public function testTensorAllocFreeNoLeak(): void
    {
        $before = $this->baseline();

        for ($i = 0; $i < 200; $i++) {
            $t = Tensor::randomNormal([100, 100]);
            unset($t);
        }

        $this->assertNoLeak($before, 'Tensor alloc/free x200');
    }

    public function testTensorCopyFreeNoLeak(): void
    {
        $before = $this->baseline();

        $orig = Tensor::randomNormal([200, 200]);
        for ($i = 0; $i < 100; $i++) {
            $copy = $orig->copy();
            unset($copy);
        }
        unset($orig);

        $this->assertNoLeak($before, 'Tensor copy/free x100');
    }

    public function testTensorMathChainNoLeak(): void
    {
        $before = $this->baseline();

        for ($i = 0; $i < 50; $i++) {
            $a = Tensor::randomNormal([100, 50]);
            $b = Tensor::randomNormal([50, 100]);
            $c = $a->matmul($b);
            $d = $c->relu();
            $e = $d->sigmoid();
            unset($a, $b, $c, $d, $e);
        }

        $this->assertNoLeak($before, 'Tensor math chain x50');
    }

    public function testTensorLinalgNoLeak(): void
    {
        $before = $this->baseline();

        for ($i = 0; $i < 30; $i++) {
            $m   = Tensor::randomNormal([20, 20]);
            $inv = $m->inverse();
            $svd = $m->svd();
            unset($m, $inv, $svd);
        }

        $this->assertNoLeak($before, 'Tensor linalg (inverse+SVD) x30');
    }

    public function testTensorViewDoesNotDoubleFreeParen(): void
    {
        // Views share the parent's data pointer; both should destruct cleanly
        $before = $this->baseline();

        for ($i = 0; $i < 100; $i++) {
            $parent = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
            $row    = $parent->row(0);
            $col    = $parent->col(1);
            $view   = $parent->view();
            // Destroy children first, then parent — no double-free
            unset($row, $col, $view, $parent);
        }

        $this->assertNoLeak($before, 'Tensor view destructor x100');
    }

    public function testTensorSliceNoLeak(): void
    {
        $before = $this->baseline();
        $base   = Tensor::range(0.0, 1000.0, 1.0);

        for ($i = 0; $i < 200; $i++) {
            $s = $base->slice(0, 0, 100);
            unset($s);
        }
        unset($base);

        $this->assertNoLeak($before, 'Tensor slice x200');
    }

    // =========================================================================
    // 2. DATASET ALLOC / FREE
    // =========================================================================

    public function testDatasetAllocFreeNoLeak(): void
    {
        $before = $this->baseline();

        for ($i = 0; $i < 50; $i++) {
            $ds = new Dataset(
                Tensor::randomNormal([500, 20]),
                Tensor::randomUniform([500], 0.0, 1.0)
            );
            unset($ds);
        }

        $this->assertNoLeak($before, 'Dataset alloc/free x50');
    }

    public function testDatasetSplitNoLeak(): void
    {
        $before = $this->baseline();

        $ds = new Dataset(
            Tensor::randomNormal([1000, 10]),
            Tensor::randomUniform([1000], 0.0, 1.0)
        );

        for ($i = 0; $i < 20; $i++) {
            [$train, $test] = $ds->split(0.8);
            unset($train, $test);
        }
        unset($ds);

        $this->assertNoLeak($before, 'Dataset split x20');
    }

    public function testDatasetBatchIterationNoLeak(): void
    {
        $before = $this->baseline();

        $ds = new Dataset(
            Tensor::randomNormal([500, 10]),
            Tensor::randomUniform([500], 0.0, 1.0)
        );

        for ($i = 0; $i < 10; $i++) {
            foreach ($ds->batches(32) as $batch) {
                unset($batch);
            }
        }
        unset($ds);

        $this->assertNoLeak($before, 'Dataset batch iteration x10');
    }

    // =========================================================================
    // 3. TRANSFORMER ALLOC / FREE
    // =========================================================================

    public function testTransformerFitTransformNoLeak(): void
    {
        $before = $this->baseline();

        for ($i = 0; $i < 50; $i++) {
            $ds     = new Dataset(Tensor::randomNormal([200, 10]));
            $scaler = new StandardScaler();
            $scaler->fit($ds);
            $out = $scaler->transform($ds);
            unset($ds, $scaler, $out);
        }

        $this->assertNoLeak($before, 'StandardScaler fit/transform x50');
    }

    // =========================================================================
    // 4. ESTIMATOR TRAIN / PREDICT
    // =========================================================================

    public function testDecisionTreeTrainPredictNoLeak(): void
    {
        $before = $this->baseline();

        for ($i = 0; $i < 10; $i++) {
            $rows = []; $labels = [];
            for ($j = 0; $j < 200; $j++) {
                $rows[]   = [$j % 2 === 0 ? 1.0 : -1.0, (float)$j / 200.0];
                $labels[] = (float)($j % 2);
            }
            $ds   = new Dataset(Tensor::fromArray($rows), Tensor::fromArray($labels));
            $clf  = new DecisionTreeClassifier(maxDepth: 5);
            $clf->train($ds);
            $preds = $clf->predict($ds);
            unset($ds, $clf, $preds);
        }

        $this->assertNoLeak($before, 'DecisionTree train/predict x10');
    }

    public function testLinearRegressionTrainPredictNoLeak(): void
    {
        $before = $this->baseline();

        for ($i = 0; $i < 20; $i++) {
            $rows = []; $y = [];
            for ($j = 0; $j < 200; $j++) {
                $x = $j / 200.0;
                $rows[] = [$x, 1.0 - $x];
                $y[]    = 2.0 * $x;
            }
            $ds    = new Dataset(Tensor::fromArray($rows), Tensor::fromArray($y));
            $reg   = new LinearRegression();
            $reg->train($ds);
            $preds = $reg->predict($ds);
            unset($ds, $reg, $preds);
        }

        $this->assertNoLeak($before, 'LinearRegression train/predict x20');
    }

    // =========================================================================
    // 5. ARENA ALLOC / RESET / DESTROY
    // =========================================================================

    public function testArenaResetAndDestroyNoLeak(): void
    {
        $before = $this->baseline();
        $ffi    = TensorEngine::get();

        for ($i = 0; $i < 20; $i++) {
            $arena = $ffi->arena_create(4 * 1024 * 1024);  // 4 MB arena

            // Allocate several tensors inside the arena.
            // owned=false on all of them so __destruct skips tensor_free().
            $tensors = [];
            for ($j = 0; $j < 10; $j++) {
                $tensors[] = new Tensor([100, 100], Tensor::DTYPE_FLOAT32, $arena);
            }
            // Null PHP references before bulk-freeing the arena
            $tensors = [];

            // Single bulk-free: arena_destroy releases struct + data in one shot
            $ffi->arena_reset($arena);
            $ffi->arena_destroy($arena);
        }

        $this->assertNoLeak($before, 'Arena create/alloc/destroy x20');
    }

    // =========================================================================
    // 6. LARGE TEMPORARY TENSOR FREED PROMPTLY
    // =========================================================================

    public function testLargeTensorFreedAfterScope(): void
    {
        $before = $this->baseline();

        // Allocate a ~40 MB tensor ([1000×10000]), verify it is released after scope
        $t = Tensor::randomNormal([1000, 10000]);
        $this->assertSame([1000, 10000], $t->shape());
        unset($t);

        \gc_collect_cycles();
        $after = \memory_get_usage(true);
        // After freeing, PHP memory should drop back close to baseline.
        // Allow 8 MB headroom for PHP runtime overhead.
        $this->assertLessThan(
            $before + 8 * 1024 * 1024,
            $after,
            "Large tensor not freed: memory stayed high after unset()"
        );
    }

    // =========================================================================
    // 7. CONCAT / PAD CHAIN
    // =========================================================================

    public function testConcatChainNoLeak(): void
    {
        $before = $this->baseline();

        for ($i = 0; $i < 30; $i++) {
            $parts = [];
            for ($j = 0; $j < 5; $j++) {
                $parts[] = Tensor::randomNormal([20, 10]);
            }
            $merged = Tensor::concat($parts, 0);
            unset($parts, $merged);
        }

        $this->assertNoLeak($before, 'Tensor::concat chain x30');
    }
}
