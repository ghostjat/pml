<?php
declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Pipeline;
use Pml\Dataset;
use Pml\Tensor;
use Pml\Estimators\Classifiers\DecisionTreeClassifier;
use Pml\Estimators\Regression\LinearRegression;

/**
 * Comprehensive test suite for Pipeline.
 */
final class PipelineTest extends TestCase
{
    private const DELTA = 1e-4;

    // =========================================================================
    // HELPERS
    // =========================================================================

    private function makeDataset(int $n = 100, int $d = 4): Dataset
    {
        mt_srand(42);
        $rows = [];
        $labels = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $d; $j++) {
                $row[] = ($i % 2 === 0 ? 1.0 : -1.0) + (mt_rand(-100, 100) / 1000.0);
            }
            $rows[] = $row;
            $labels[] = (float)($i % 2);
        }
        return new Dataset(Tensor::fromArray($rows), Tensor::fromArray($labels));
    }

    // =========================================================================
    // 1. BASIC PIPELINE FUNCTIONALITY
    // =========================================================================

    public function testPipelineCanTrainAndPredict(): void
    {
        $ds = $this->makeDataset(100, 4);
        $pipeline = new Pipeline([], new DecisionTreeClassifier(maxDepth: 5));
        
        $pipeline->train($ds);
        $predictions = $pipeline->predict($ds);
        
        $this->assertInstanceOf(Tensor::class, $predictions);
        $this->assertSame(100, $predictions->size());
    }

    public function testPipelineWithEmptyTransformers(): void
    {
        $ds = $this->makeDataset(100, 4);
        $pipeline = new Pipeline([], new DecisionTreeClassifier(maxDepth: 5));
        
        $pipeline->train($ds);
        $predictions = $pipeline->predict($ds);
        
        $this->assertInstanceOf(Tensor::class, $predictions);
    }

    public function testPipelineTrainedStatus(): void
    {
        $ds = $this->makeDataset(100, 4);
        $pipeline = new Pipeline([], new DecisionTreeClassifier(maxDepth: 5));
        
        $this->assertFalse($pipeline->trained());
        
        $pipeline->train($ds);
        
        $this->assertTrue($pipeline->trained());
    }

    // =========================================================================
    // 2. PIPELINE PREDICTION ACCURACY
    // =========================================================================

    public function testPipelinePredictionsMatchDirectEstimator(): void
    {
        $ds = $this->makeDataset(100, 4);
        
        // Train directly
        $directEstimator = new DecisionTreeClassifier(maxDepth: 5);
        $directEstimator->train($ds);
        $directPredictions = $directEstimator->predict($ds);
        
        // Train via pipeline
        $pipeline = new Pipeline([], new DecisionTreeClassifier(maxDepth: 5));
        $pipeline->train($ds);
        $pipelinePredictions = $pipeline->predict($ds);
        
        // Results should be identical
        $direct = $directPredictions->toFlatArray();
        $piped = $pipelinePredictions->toFlatArray();
        
        foreach ($direct as $i => $val) {
            $this->assertEqualsWithDelta($val, $piped[$i], self::DELTA);
        }
    }

    // =========================================================================
    // 3. PIPELINE ERROR HANDLING
    // =========================================================================

    public function testPipelinePredictBeforeTrainingThrows(): void
    {
        $ds = $this->makeDataset(100, 4);
        $pipeline = new Pipeline([], new DecisionTreeClassifier(maxDepth: 5));
        
        $this->expectException(\RuntimeException::class);
        $pipeline->predict($ds);
    }

    // =========================================================================
    // 4. PIPELINE PERSISTENCE
    // =========================================================================

    public function testPipelineSaveCreatesDirectory(): void
    {
        $ds = $this->makeDataset(100, 4);
        $pipeline = new Pipeline([], new DecisionTreeClassifier(maxDepth: 5));
        $pipeline->train($ds);
        
        $tempDir = sys_get_temp_dir() . '/pipeline_test_dir_' . uniqid();
        if (is_dir($tempDir)) {
            $this->cleanupDir($tempDir);
        }
        
        $pipeline->save($tempDir);
        
        $this->assertDirectoryExists($tempDir);
        $this->assertFileExists($tempDir . '/pipeline.json');
        
        // Cleanup
        $this->cleanupDir($tempDir);
    }

    // Note: Pipeline persistence requires estimator to implement Persistable
    // DecisionTreeClassifier doesn't implement Persistable, so we test directory creation only

    // =========================================================================
    // 5. EDGE CASES
    // =========================================================================

    public function testPipelineWithSmallDataset(): void
    {
        $ds = new Dataset(
            Tensor::fromArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            Tensor::fromArray([0.0, 1.0, 0.0, 1.0])
        );
        $pipeline = new Pipeline([], new DecisionTreeClassifier(maxDepth: 2));
        
        $pipeline->train($ds);
        $predictions = $pipeline->predict($ds);
        
        $this->assertInstanceOf(Tensor::class, $predictions);
        $this->assertSame(4, $predictions->size());
    }

    public function testPipelineWithLargeDataset(): void
    {
        $ds = $this->makeDataset(1000, 10);
        $pipeline = new Pipeline([], new DecisionTreeClassifier(maxDepth: 5));
        
        $pipeline->train($ds);
        $predictions = $pipeline->predict($ds);
        
        $this->assertInstanceOf(Tensor::class, $predictions);
        $this->assertSame(1000, $predictions->size());
    }

    // =========================================================================
    // 6. REGRESSION PIPELINE
    // =========================================================================

    public function testPipelineWithRegressionEstimator(): void
    {
        mt_srand(42);
        $rows = [];
        $labels = [];
        for ($i = 0; $i < 100; $i++) {
            $x = $i / 10.0;
            $rows[] = [$x];
            $labels[] = [2.0 * $x + 1.0 + (mt_rand(-10, 10) / 10.0)];
        }
        $ds = new Dataset(Tensor::fromArray($rows), Tensor::fromArray($labels));
        
        $pipeline = new Pipeline([], new LinearRegression());
        $pipeline->train($ds);
        $predictions = $pipeline->predict($ds);
        
        $this->assertInstanceOf(Tensor::class, $predictions);
        $this->assertSame(100, $predictions->size());
    }

    // =========================================================================
    // HELPERS
    // =========================================================================

    private function cleanupDir(string $dir): void
    {
        if (!is_dir($dir)) {
            return;
        }
        $files = array_diff(scandir($dir), ['.', '..']);
        foreach ($files as $file) {
            $path = $dir . '/' . $file;
            if (is_dir($path)) {
                $this->cleanupDir($path);
            } else {
                unlink($path);
            }
        }
        rmdir($dir);
    }
}