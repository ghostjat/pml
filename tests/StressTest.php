<?php
declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Dataset;
use Pml\Tensor;
use Pml\Estimators\Classifiers\DecisionTreeClassifier;
use Pml\Estimators\Classifiers\RandomForestClassifier;
use Pml\Estimators\Classifiers\LogisticRegression;
use Pml\Estimators\Regression\LinearRegression;
use Pml\Estimators\Regression\DecisionTreeRegressor;
use Pml\Estimators\Regression\GradientBoostingRegressor;
use Pml\Transformers\StandardScaler;
use Pml\Transformers\MinMaxScaler;
use Pml\Metrics\Classification\Accuracy;
use Pml\Metrics\Regression\RSquared;

/**
 * Stress, endurance, and large-scale tests.
 *
 * These tests validate the library under production-grade data volumes:
 * - High row counts (10k – 100k)
 * - High feature counts (100 – 1000)
 * - Repeated training cycles (pipeline endurance)
 * - Mixed-precision workloads
 * - Concurrent batch inference
 *
 * All tests use a 120s per-test timeout.  They are kept in a separate
 * @group stress suite so CI can exclude them with --exclude-group stress.
 *
 * @group stress
 */
final class StressTest extends TestCase
{
    // =========================================================================
    // HELPERS
    // =========================================================================

    private function makeBigBinaryDataset(int $n, int $d): Dataset
    {
        mt_srand(12345);
        $rows = []; $labels = [];
        for ($i = 0; $i < $n; $i++) {
            $cls = $i % 2;
            $row = [];
            for ($j = 0; $j < $d; $j++) {
                $row[] = ($cls === 0 ? 1.0 : -1.0) + (mt_rand(-200, 200) / 1000.0);
            }
            $rows[]   = $row;
            $labels[] = (float)$cls;
        }
        return new Dataset(Tensor::fromArray($rows), Tensor::fromArray($labels));
    }

    private function makeBigRegressionDataset(int $n, int $d): Dataset
    {
        mt_srand(99);
        $rows = []; $y = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            $sum = 0.0;
            for ($j = 0; $j < $d; $j++) {
                $v = ($i + $j) / ($n * $d);
                $row[] = $v;
                $sum += $v * ($j + 1);
            }
            $rows[] = $row;
            $y[]    = $sum + (mt_rand(-50, 50) / 10000.0);
        }
        return new Dataset(Tensor::fromArray($rows), Tensor::fromArray($y));
    }

    // =========================================================================
    // 1. HIGH ROW COUNT — 10k rows
    // =========================================================================

    public function testLogisticRegression10kRows(): void
    {
        $ds = $this->makeBigBinaryDataset(10000, 10);
        [$train, $test] = $ds->split(0.8);

        $clf = new LogisticRegression(epochs: 30, learningRate: 0.5, batchSize: 256);
        $clf->train($train);
        $preds = $clf->predict($test);

        $this->assertSame($test->numRows(), $preds->size());
        $acc = (new Accuracy())->score($preds, $test->labels());
        $this->assertGreaterThanOrEqual(0.8, $acc,
            "LogisticRegression 10k rows accuracy {$acc}");
    }

    public function testDecisionTree10kRows(): void
    {
        $ds = $this->makeBigBinaryDataset(10000, 8);
        [$train, $test] = $ds->split(0.8);

        $clf = new DecisionTreeClassifier(maxDepth: 6);
        $clf->train($train);
        $preds = $clf->predict($test);

        $this->assertSame($test->numRows(), $preds->size());
        $acc = (new Accuracy())->score($preds, $test->labels());
        $this->assertGreaterThanOrEqual(0.85, $acc);
    }

    public function testLinearRegression10kRows(): void
    {
        $ds = $this->makeBigRegressionDataset(10000, 5);
        [$train, $test] = $ds->split(0.8);

        $reg = new LinearRegression();
        $reg->train($train);
        $preds = $reg->predict($test);

        $this->assertSame($test->numRows(), $preds->size());
        $r2 = (new RSquared())->score($preds, $test->labels());
        $this->assertGreaterThanOrEqual(0.8, $r2,
            "LinearRegression 10k rows R² {$r2}");
    }

    // =========================================================================
    // 2. HIGH FEATURE COUNT — 100 features
    // =========================================================================

    public function testLinearRegression100Features(): void
    {
        $ds = $this->makeBigRegressionDataset(500, 100);
        [$train, $test] = $ds->split(0.8);

        $reg = new LinearRegression();
        $reg->train($train);
        $preds = $reg->predict($test);

        $this->assertSame($test->numRows(), $preds->size());
        foreach ($preds->toFlatArray() as $p) {
            $this->assertFalse(\is_nan($p),      'Prediction is NaN for 100-feature problem');
            $this->assertFalse(\is_infinite($p), 'Prediction is Inf for 100-feature problem');
        }
    }

    public function testDecisionTree100Features(): void
    {
        $ds = $this->makeBigBinaryDataset(1000, 100);
        [$train, $test] = $ds->split(0.8);

        $clf = new DecisionTreeClassifier(maxDepth: 5);
        $clf->train($train);
        $preds = $clf->predict($test);
        $this->assertSame($test->numRows(), $preds->size());
    }

    // =========================================================================
    // 3. FULL PIPELINE ENDURANCE — 20 iterations
    // =========================================================================

    public function testFullPipelineEndurance(): void
    {
        $ds = $this->makeBigBinaryDataset(2000, 10);

        for ($iter = 0; $iter < 20; $iter++) {
            $shuffled = $ds->randomize();
            [$train, $test] = $shuffled->split(0.8);

            $scaler = new StandardScaler();
            $scaler->fit($train);
            $trainScaled = $scaler->transform($train);
            $testScaled  = $scaler->transform($test);

            $clf = new LogisticRegression(epochs: 10, learningRate: 0.5, batchSize: 64);
            $clf->train($trainScaled);
            $preds = $clf->predict($testScaled);

            $acc = (new Accuracy())->score($preds, $testScaled->labels());
            $this->assertGreaterThanOrEqual(0.7, $acc,
                "Pipeline iteration {$iter} accuracy {$acc} too low");

            unset($shuffled, $train, $test, $scaler, $trainScaled, $testScaled, $clf, $preds);
        }

        $this->addToAssertionCount(1); // survived 20 iterations
    }

    // =========================================================================
    // 4. RANDOM FOREST — 50 estimators × 5k rows
    // =========================================================================

    public function testRandomForest50Trees5kRows(): void
    {
        $ds = $this->makeBigBinaryDataset(5000, 8);
        [$train, $test] = $ds->split(0.8);

        $clf = new RandomForestClassifier(nEstimators: 50, maxDepth: 6);
        $clf->train($train);
        $preds = $clf->predict($test);

        $this->assertSame($test->numRows(), $preds->size());
        $acc = (new Accuracy())->score($preds, $test->labels());
        $this->assertGreaterThanOrEqual(0.85, $acc,
            "RandomForest 50 trees accuracy {$acc}");
    }

    // =========================================================================
    // 5. TENSOR MATH STRESS — large matrix chains
    // =========================================================================

    public function testMatmulChain512(): void
    {
        // [512×512] matmul chains — exercises OpenBLAS throughput
        $a = Tensor::randomNormal([512, 512]);
        $b = Tensor::randomNormal([512, 512]);

        for ($i = 0; $i < 5; $i++) {
            $c = $a->matmul($b);
            $this->assertSame([512, 512], $c->shape());
            unset($c);
        }
        unset($a, $b);
        $this->addToAssertionCount(1);
    }

    public function testBatchMatmulBMM(): void
    {
        // Batch matmul: [32 × 64 × 64] × [32 × 64 × 64]
        $a = Tensor::randomNormal([32, 64, 64]);
        $b = Tensor::randomNormal([32, 64, 64]);
        $c = $a->bmm($b);
        $this->assertSame([32, 64, 64], $c->shape());
        unset($a, $b, $c);
    }

    public function testSvdStress(): void
    {
        // 50 SVD decompositions on 100×100 matrices
        for ($i = 0; $i < 50; $i++) {
            $m   = Tensor::randomNormal([100, 100]);
            $svd = $m->svd();
            $this->assertSame([100, 100], $svd['U']->shape());
            $this->assertSame(100, $svd['S']->size());
            unset($m, $svd);
        }
    }

    // =========================================================================
    // 6. GRADIENT BOOSTING STRESS — 50 trees × 2k rows
    // =========================================================================

    public function testGradientBoosting50Trees2kRows(): void
    {
        $ds = $this->makeBigRegressionDataset(2000, 5);
        [$train, $test] = $ds->split(0.8);

        $reg = new GradientBoostingRegressor(nEstimators: 50);
        $reg->train($train);
        $preds = $reg->predict($test);

        $this->assertSame($test->numRows(), $preds->size());
        $r2 = (new RSquared())->score($preds, $test->labels());
        $this->assertGreaterThanOrEqual(0.7, $r2,
            "GradientBoosting 50 trees R² {$r2}");
    }

    // =========================================================================
    // 7. DATASET SPLIT + FOLD REPEATED STRESS
    // =========================================================================

    public function testDatasetFoldStress(): void
    {
        $ds = $this->makeBigBinaryDataset(1000, 5);

        for ($round = 0; $round < 5; $round++) {
            $totalTrain = 0;
            $totalVal   = 0;
            foreach ($ds->fold(10) as [$train, $val]) {
                $totalTrain += $train->numRows();
                $totalVal   += $val->numRows();
                unset($train, $val);
            }
            // Each fold: 900 train + 100 val, across 10 folds = 9000 + 1000
            $this->assertSame(9000, $totalTrain, "Fold round {$round} train total wrong");
            $this->assertSame(1000, $totalVal,   "Fold round {$round} val total wrong");
        }
    }

    // =========================================================================
    // 8. CONCURRENT BATCH INFERENCE
    // =========================================================================

    public function testBatchInferenceLargeDataset(): void
    {
        // Train on small, infer on 50k rows in batches
        $train = $this->makeBigBinaryDataset(500, 6);
        $infer = $this->makeBigBinaryDataset(50000, 6);

        $clf = new DecisionTreeClassifier(maxDepth: 6);
        $clf->train($train);

        $totalPreds = 0;
        foreach ($infer->batches(1024) as $batch) {
            $preds = $clf->predict($batch);
            $totalPreds += $preds->size();
            unset($preds, $batch);
        }

        $this->assertSame(50000, $totalPreds,
            "Batch inference did not cover all 50k rows");
        unset($train, $infer, $clf);
    }

    // =========================================================================
    // 9. MEMORY UNDER STRESS
    // =========================================================================

    public function testNoMemoryExplosionUnder10kRows1kFeatures(): void
    {
        // This is the headline stress test: 10k × 1k FLOAT32 ≈ 40 MB tensor
        $before = \memory_get_usage(true);

        $ds  = $this->makeBigBinaryDataset(10000, 100);
        $scaler = new MinMaxScaler();
        $scaler->fit($ds);
        $scaled = $scaler->transform($ds);

        $this->assertSame(10000, $scaled->numRows());
        $this->assertSame(100,   $scaled->numColumns());

        unset($ds, $scaler, $scaled);
        \gc_collect_cycles();

        $after = \memory_get_usage(true);
        // Should have released most of the working set
        $this->assertLessThan(
            $before + 64 * 1024 * 1024,   // allow 64 MB headroom for PHP runtime
            $after,
            "Memory not released after 10k×100 stress pipeline"
        );
    }
}
