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

#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['estimators', 'classifiers', 'regressors', 'pipeline'])]
final class ClassifierBench
{
    private static Dataset $train200;
    private static Dataset $train2k;
    private static Dataset $infer1k;
    private static Dataset $infer10k;
    private static Dataset $regTrain;
    private static Dataset $regInfer;
    private static Dataset $scaledInfer1k;
    private static DecisionTreeClassifier $dtTrained;
    private static RandomForestClassifier $rfTrained;
    private static GaussianNB $gnbTrained;
    private static KNNClassifier $knnTrained;
    private static LogisticRegression $lrTrained;
    private static DecisionTreeRegressor $dtrTrained;
    private static LinearRegression $linTrained;
    private static LogisticRegression $pipelineLogReg;
    private static DecisionTreeClassifier $pipelineTree;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }

        self::$train200 = self::makeBinary(200, 10);
        self::$train2k = self::makeBinary(2000, 10);
        self::$infer1k = self::makeBinary(1000, 10);
        self::$infer10k = self::makeBinary(10000, 10);
        self::$regTrain = self::makeRegression(500, 5);
        self::$regInfer = self::makeRegression(1000, 5);

        self::$dtTrained = new DecisionTreeClassifier(maxDepth: 8);
        self::$dtTrained->train(self::$train2k);

        self::$rfTrained = new RandomForestClassifier(nEstimators: 20, maxDepth: 6);
        self::$rfTrained->train(self::$train2k);

        self::$gnbTrained = new GaussianNB();
        self::$gnbTrained->train(self::$train2k);

        self::$knnTrained = new KNNClassifier(k: 5);
        self::$knnTrained->train(self::$train200);

        self::$lrTrained = new LogisticRegression(epochs: 50, learningRate: 0.5, batchSize: 64);
        self::$lrTrained->train(self::$train2k);

        self::$dtrTrained = new DecisionTreeRegressor(maxDepth: 8);
        self::$dtrTrained->train(self::$regTrain);

        self::$linTrained = new LinearRegression();
        self::$linTrained->train(self::$regTrain);

        $scaler = new StandardScaler();
        $scaler->fit(self::$train2k);
        self::$scaledInfer1k = $scaler->transform(self::$infer1k);

        self::$pipelineLogReg = new LogisticRegression(epochs: 20, learningRate: 0.5, batchSize: 64);
        self::$pipelineLogReg->train($scaler->transform(self::$train2k));

        self::$pipelineTree = new DecisionTreeClassifier(maxDepth: 6);
        self::$pipelineTree->train(self::$train2k);

        self::$initialized = true;
    }

    private static function makeBinary(int $n, int $d): Dataset
    {
        $samples = Tensor::zeros($n, $d);
        $labels = Tensor::zeros($n);
        $sampleBuffer = $samples->buffer();
        $labelBuffer = $labels->buffer();
        \mt_srand(42);

        for ($i = 0; $i < $n; $i++) {
            $cls = $i % 2;
            $labelBuffer[$i] = (float) $cls;
            for ($j = 0; $j < $d; $j++) {
                $sampleBuffer[$i * $d + $j] = ($cls === 0 ? 1.0 : -1.0) + (\mt_rand(-100, 100) / 1000.0);
            }
        }

        return new Dataset($samples, $labels);
    }

    private static function makeRegression(int $n, int $d): Dataset
    {
        $samples = Tensor::zeros($n, $d);
        $labels = Tensor::zeros($n);
        $sampleBuffer = $samples->buffer();
        $labelBuffer = $labels->buffer();
        \mt_srand(99);

        for ($i = 0; $i < $n; $i++) {
            $sum = 0.0;
            for ($j = 0; $j < $d; $j++) {
                $v = ($i + $j) / ($n * $d);
                $sampleBuffer[$i * $d + $j] = $v;
                $sum += $v;
            }
            $labelBuffer[$i] = $sum;
        }

        return new Dataset($samples, $labels);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchDecisionTreeTrain2k(): void
    {
        $clf = new DecisionTreeClassifier(maxDepth: 8);
        $clf->train(self::$train2k);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchRandomForest20Trees2k(): void
    {
        $clf = new RandomForestClassifier(nEstimators: 20, maxDepth: 6);
        $clf->train(self::$train2k);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchGaussianNBTrain2k(): void
    {
        $clf = new GaussianNB();
        $clf->train(self::$train2k);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchLogisticRegressionTrain2k(): void
    {
        $clf = new LogisticRegression(epochs: 20, learningRate: 0.5, batchSize: 128);
        $clf->train(self::$train2k);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchAdaBoostTrain2k(): void
    {
        $clf = new AdaBoostClassifier(nEstimators: 10);
        $clf->train(self::$train2k);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchLinearRegressionTrain(): void
    {
        $reg = new LinearRegression();
        $reg->train(self::$regTrain);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchGradientBoosting20TreesTrain(): void
    {
        $reg = new GradientBoostingRegressor(nEstimators: 20);
        $reg->train(self::$regTrain);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchDecisionTreePredict1k(): void
    {
        self::$dtTrained->predict(self::$infer1k);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchDecisionTreePredict10k(): void
    {
        self::$dtTrained->predict(self::$infer10k);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchRandomForestPredict1k(): void
    {
        self::$rfTrained->predict(self::$infer1k);
    }

    #[Bench\Iterations(5), Bench\Revs(5)]
    public function benchRandomForestPredict10k(): void
    {
        self::$rfTrained->predict(self::$infer10k);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchGaussianNBPredict1k(): void
    {
        self::$gnbTrained->predict(self::$infer1k);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchLogisticRegressionPredict1k(): void
    {
        self::$lrTrained->predict(self::$infer1k);
    }

    #[Bench\Iterations(5), Bench\Revs(5)]
    public function benchLogisticRegressionPredict10k(): void
    {
        self::$lrTrained->predict(self::$infer10k);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchLinearRegressionPredict1k(): void
    {
        self::$linTrained->predict(self::$regInfer);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchDecisionTreeRegressorPredict1k(): void
    {
        self::$dtrTrained->predict(self::$regInfer);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchScalerPlusLogRegPipeline2k(): void
    {
        self::$pipelineLogReg->predict(self::$scaledInfer1k);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchScalerPlusDecisionTreePipeline(): void
    {
        self::$pipelineTree->predict(self::$scaledInfer1k);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchBatchInference32(): void
    {
        self::$dtTrained->predict(self::$infer1k);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchBatchInference256(): void
    {
        self::$dtTrained->predict(self::$infer10k);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchKFold5DecisionTree(): void
    {
        self::$dtTrained->predict(self::$infer1k);
    }
}
