<?php
declare(strict_types=1);

namespace Pml\Benchmarks;

use PhpBench\Attributes as Bench;
use Pml\Dataset;
use Pml\Tensor;
use Pml\Estimators\Classifiers\DecisionTreeClassifier;
use Pml\Estimators\Classifiers\RandomForestClassifier;
use Pml\Estimators\Classifiers\GaussianNB;
use Pml\Estimators\Classifiers\KNNClassifier;
use Pml\Estimators\Classifiers\LogisticRegression;
use Pml\Estimators\Classifiers\AdaBoostClassifier;
use Pml\Estimators\Regression\LinearRegression;
use Pml\Estimators\Regression\DecisionTreeRegressor;
use Pml\Estimators\Regression\GradientBoostingRegressor;
use Pml\Transformers\StandardScaler;

/**
 * Estimator throughput benchmarks: samples/sec for train and predict.
 *
 * Methodology:
 * - Shared pre-trained model per class (constructor trains once)
 * - Benchmark methods measure only the hot path (predict or one epoch)
 * - Data sizes chosen to surface real-world latency differences
 *
 * Run with:
 *   vendor/bin/phpbench run benchmarks/ClassifierBench.php --report=aggregate
 */
#[Bench\Groups(['estimators', 'classifiers', 'regressors', 'pipeline'])]
final class ClassifierBench
{
    // Binary classification datasets at different scales
    private Dataset $train200;
    private Dataset $train2k;
    private Dataset $infer1k;
    private Dataset $infer10k;

    // Regression dataset
    private Dataset $regTrain;
    private Dataset $regInfer;

    // Pre-trained models (train cost paid in constructor, not in bench)
    private DecisionTreeClassifier $dtTrained;
    private RandomForestClassifier $rfTrained;
    private GaussianNB             $gnbTrained;
    private KNNClassifier          $knnTrained;
    private LogisticRegression     $lrTrained;
    private DecisionTreeRegressor  $dtrTrained;
    private LinearRegression       $linTrained;

    public function __construct()
    {
        // Build datasets
        $this->train200  = $this->makeBinary(200, 10);
        $this->train2k   = $this->makeBinary(2000, 10);
        $this->infer1k   = $this->makeBinary(1000, 10);
        $this->infer10k  = $this->makeBinary(10000, 10);
        $this->regTrain  = $this->makeRegression(500, 5);
        $this->regInfer  = $this->makeRegression(1000, 5);

        // Pre-train all models
        $this->dtTrained  = new DecisionTreeClassifier(maxDepth: 8);
        $this->dtTrained->train($this->train2k);

        $this->rfTrained  = new RandomForestClassifier(nEstimators: 20, maxDepth: 6);
        $this->rfTrained->train($this->train2k);

        $this->gnbTrained = new GaussianNB();
        $this->gnbTrained->train($this->train2k);

        $this->knnTrained = new KNNClassifier(k: 5);
        $this->knnTrained->train($this->train200);

        $this->lrTrained  = new LogisticRegression(epochs: 50, learningRate: 0.5, batchSize: 64);
        $this->lrTrained->train($this->train2k);

        $this->dtrTrained = new DecisionTreeRegressor(maxDepth: 8);
        $this->dtrTrained->train($this->regTrain);

        $this->linTrained = new LinearRegression();
        $this->linTrained->train($this->regTrain);
    }

    // =========================================================================
    // HELPERS
    // =========================================================================

    private function makeBinary(int $n, int $d): Dataset
    {
        \mt_srand(42);
        $rows = []; $labels = [];
        for ($i = 0; $i < $n; $i++) {
            $cls = $i % 2;
            $row = [];
            for ($j = 0; $j < $d; $j++) {
                $row[] = ($cls === 0 ? 1.0 : -1.0) + (\mt_rand(-100, 100) / 1000.0);
            }
            $rows[]   = $row;
            $labels[] = (float)$cls;
        }
        return new Dataset(Tensor::fromArray($rows), Tensor::fromArray($labels));
    }

    private function makeRegression(int $n, int $d): Dataset
    {
        \mt_srand(99);
        $rows = []; $y = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            $sum = 0.0;
            for ($j = 0; $j < $d; $j++) {
                $v = ($i + $j) / ($n * $d);
                $row[] = $v;
                $sum  += $v;
            }
            $rows[] = $row;
            $y[]    = $sum;
        }
        return new Dataset(Tensor::fromArray($rows), Tensor::fromArray($y));
    }

    // =========================================================================
    // TRAINING THROUGHPUT
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchDecisionTreeTrain2k(): void
    {
        $clf = new DecisionTreeClassifier(maxDepth: 8);
        $clf->train($this->train2k);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchRandomForest20Trees2k(): void
    {
        $clf = new RandomForestClassifier(nEstimators: 20, maxDepth: 6);
        $clf->train($this->train2k);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchGaussianNBTrain2k(): void
    {
        $clf = new GaussianNB();
        $clf->train($this->train2k);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchLogisticRegressionTrain2k(): void
    {
        $clf = new LogisticRegression(epochs: 20, learningRate: 0.5, batchSize: 128);
        $clf->train($this->train2k);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchAdaBoostTrain2k(): void
    {
        $clf = new AdaBoostClassifier(nEstimators: 10);
        $clf->train($this->train2k);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchLinearRegressionTrain(): void
    {
        $reg = new LinearRegression();
        $reg->train($this->regTrain);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchGradientBoosting20TreesTrain(): void
    {
        $reg = new GradientBoostingRegressor(nEstimators: 20);
        $reg->train($this->regTrain);
    }

    // =========================================================================
    // INFERENCE THROUGHPUT (pre-trained)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchDecisionTreePredict1k(): void
    {
        $this->dtTrained->predict($this->infer1k);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchDecisionTreePredict10k(): void
    {
        $this->dtTrained->predict($this->infer10k);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchRandomForestPredict1k(): void
    {
        $this->rfTrained->predict($this->infer1k);
    }

    #[Bench\Iterations(5), Bench\Revs(5)]
    public function benchRandomForestPredict10k(): void
    {
        $this->rfTrained->predict($this->infer10k);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchGaussianNBPredict1k(): void
    {
        $this->gnbTrained->predict($this->infer1k);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchLogisticRegressionPredict1k(): void
    {
        $this->lrTrained->predict($this->infer1k);
    }

    #[Bench\Iterations(5), Bench\Revs(5)]
    public function benchLogisticRegressionPredict10k(): void
    {
        $this->lrTrained->predict($this->infer10k);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchLinearRegressionPredict1k(): void
    {
        $this->linTrained->predict($this->regInfer);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchDecisionTreeRegressorPredict1k(): void
    {
        $this->dtrTrained->predict($this->regInfer);
    }

    // =========================================================================
    // PIPELINE: scaler + classifier
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchScalerPlusLogRegPipeline2k(): void
    {
        $scaler = new StandardScaler();
        $scaler->fit($this->train2k);
        $trainScaled = $scaler->transform($this->train2k);
        $testScaled  = $scaler->transform($this->infer1k);

        $clf = new LogisticRegression(epochs: 20, learningRate: 0.5, batchSize: 64);
        $clf->train($trainScaled);
        $clf->predict($testScaled);

        unset($scaler, $trainScaled, $testScaled, $clf);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchScalerPlusDecisionTreePipeline(): void
    {
        $scaler = new StandardScaler();
        $scaler->fit($this->train2k);
        $trainScaled = $scaler->transform($this->train2k);
        $testScaled  = $scaler->transform($this->infer1k);

        $clf = new DecisionTreeClassifier(maxDepth: 6);
        $clf->train($trainScaled);
        $clf->predict($testScaled);

        unset($scaler, $trainScaled, $testScaled, $clf);
    }

    // =========================================================================
    // BATCH INFERENCE (simulates real-time serving)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchBatchInference32(): void
    {
        foreach ($this->infer1k->batches(32) as $batch) {
            $this->dtTrained->predict($batch);
        }
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchBatchInference256(): void
    {
        foreach ($this->infer1k->batches(256) as $batch) {
            $this->dtTrained->predict($batch);
        }
    }

    // =========================================================================
    // CROSS-VALIDATION COST
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchKFold5DecisionTree(): void
    {
        foreach ($this->train2k->fold(5) as [$train, $val]) {
            $clf = new DecisionTreeClassifier(maxDepth: 5);
            $clf->train($train);
            $clf->predict($val);
            unset($clf, $train, $val);
        }
    }
}
