<?php
declare(strict_types=1);

namespace Pml\Benchmarks\Workloads;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Transformers\StandardScaler;
use Pml\Transformers\MinMaxScaler;
use Pml\Transformers\ZScaleStandardizer;
use Pml\Estimators\Classifiers\LogisticRegression;
use Pml\Estimators\Classifiers\RandomForestClassifier;
use Pml\Estimators\Classifiers\GaussianNB;
use Pml\Estimators\Regression\LinearRegression;
use Pml\Estimators\Regression\GradientBoostingRegressor;
use Pml\Estimators\Clusterers\KMeans;
use Pml\Estimators\Decomposition\PrincipalComponentAnalysis;
use Pml\Estimators\AnomalyDetectors\IsolationForest;

/**
 * End-to-end tabular ML workload benchmarks.
 *
 * Each benchmark simulates a realistic pipeline: preprocess → fit → predict.
 *
 * Groups:
 *   tabular     — all tabular ML benchmarks
 *   classify    — classification pipelines
 *   regress     — regression pipelines
 *   cluster     — clustering pipelines
 *   anomaly     — anomaly detection
 *   decompose   — dimensionality reduction
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['tabular', 'workload'])]
final class TabularMLBench
{
    private static Dataset $clf2k20;
    private static Dataset $clf5k50;
    private static Dataset $reg2k10;
    private static Dataset $unlab2k20;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }
        self::$clf2k20  = self::makeClassification(2000, 20, 3);
        self::$clf5k50  = self::makeClassification(5000, 50, 5);
        self::$reg2k10  = self::makeRegression(2000, 10);
        self::$unlab2k20 = new Dataset(Tensor::randomNormal([2000, 20]));
        self::$initialized = true;
    }

    private static function makeClassification(int $n, int $d, int $classes): Dataset
    {
        \mt_srand(42);
        $samples = Tensor::randomNormal([$n, $d]);
        $lBuf = \array_fill(0, $n, 0.0);
        for ($i = 0; $i < $n; $i++) {
            $lBuf[$i] = (float)($i % $classes);
        }
        return new Dataset($samples, Tensor::fromArray($lBuf));
    }

    private static function makeRegression(int $n, int $d): Dataset
    {
        \mt_srand(99);
        $X = Tensor::randomNormal([$n, $d]);
        $w = Tensor::randomNormal([$d, 1]);
        $y = $X->matmul($w)->flatten();
        return new Dataset($X, $y);
    }

    // =========================================================================
    // CLASSIFICATION PIPELINES
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['tabular', 'classify'])]
    public function benchScalerLogRegPipeline2k(): void
    {
        $scaler = new StandardScaler();
        $scaler->fit(self::$clf2k20);
        $scaled = $scaler->transform(self::$clf2k20);
        $lr = new LogisticRegression(epochs: 30, learningRate: 0.1, batchSize: 64);
        $lr->train($scaled);
        $lr->predict($scaled);
        unset($scaled, $lr, $scaler);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['tabular', 'classify'])]
    public function benchMinMaxRFPipeline2k(): void
    {
        $scaler = new MinMaxScaler();
        $scaler->fit(self::$clf2k20);
        $scaled = $scaler->transform(self::$clf2k20);
        $rf = new RandomForestClassifier(nEstimators: 15, maxDepth: 6);
        $rf->train($scaled);
        $rf->predict($scaled);
        unset($scaled, $rf, $scaler);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['tabular', 'classify'])]
    public function benchGaussianNBFitPredict2k(): void
    {
        $gnb = new GaussianNB();
        $gnb->train(self::$clf2k20);
        $gnb->predict(self::$clf2k20);
        unset($gnb);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['tabular', 'classify'])]
    public function benchScalerLogRegPipeline5k50d(): void
    {
        $scaler = new ZScaleStandardizer();
        $scaler->fit(self::$clf5k50);
        $scaled = $scaler->transform(self::$clf5k50);
        $lr = new LogisticRegression(epochs: 20, learningRate: 0.1, batchSize: 128);
        $lr->train($scaled);
        $lr->predict($scaled);
        unset($scaled, $lr, $scaler);
    }

    // =========================================================================
    // REGRESSION PIPELINES
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['tabular', 'regress'])]
    public function benchLinearRegressionFitPredict2k(): void
    {
        $lr = new LinearRegression();
        $lr->train(self::$reg2k10);
        $lr->predict(self::$reg2k10);
        unset($lr);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['tabular', 'regress'])]
    public function benchGradientBoostingFitPredict2k(): void
    {
        $gb = new GradientBoostingRegressor(nEstimators: 10);
        $gb->train(self::$reg2k10);
        $gb->predict(self::$reg2k10);
        unset($gb);
    }

    // =========================================================================
    // CLUSTERING
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['tabular', 'cluster'])]
    public function benchKMeans5Clusters2k(): void
    {
        $km = new KMeans(k: 5);
        $km->train(self::$unlab2k20);
        $km->predict(self::$unlab2k20);
        unset($km);
    }

    // =========================================================================
    // DIMENSIONALITY REDUCTION
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['tabular', 'decompose'])]
    public function benchPCA10Components2k(): void
    {
        $pca = new PrincipalComponentAnalysis(10);
        $pca->train(self::$clf2k20);
        $reduced = $pca->predict(self::$clf2k20);
        unset($pca, $reduced);
    }

    // =========================================================================
    // ANOMALY DETECTION
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['tabular', 'anomaly'])]
    public function benchIsolationForestFitPredict(): void
    {
        $ifor = new IsolationForest(nEstimators: 50, sampleSize: 128);
        $ifor->train(self::$unlab2k20);
        $ifor->predict(self::$unlab2k20);
        unset($ifor);
    }
}
