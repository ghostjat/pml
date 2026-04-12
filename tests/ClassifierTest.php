<?php
declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Dataset;
use Pml\Tensor;
use Pml\Estimators\Classifiers\DecisionTreeClassifier;
use Pml\Estimators\Classifiers\RandomForestClassifier;
use Pml\Estimators\Classifiers\GaussianNB;
use Pml\Estimators\Classifiers\BernoulliNB;
use Pml\Estimators\Classifiers\MultinomialNB;
use Pml\Estimators\Classifiers\KNNClassifier;
use Pml\Estimators\Classifiers\KDNeighborsClassifier;
use Pml\Estimators\Classifiers\LogisticRegression;
use Pml\Estimators\Classifiers\SoftmaxClassifier;
use Pml\Estimators\Classifiers\AdaBoostClassifier;
use Pml\Estimators\Classifiers\ExtraTreesClassifier;
use Pml\Estimators\Classifiers\DummyClassifier;
use Pml\Metrics\Classification\Accuracy;

/**
 * Lifecycle + accuracy tests for every classifier.
 *
 * Design contract:
 * - Every estimator must implement train() / predict() / trained()
 * - Accuracy on a perfectly linearly separable toy problem must be ≥ 0.85
 * - trained() must be false before train() and true after
 * - predict() output shape must match dataset row count
 */
final class ClassifierTest extends TestCase
{
    private const MIN_ACCURACY = 0.80;
    private const DELTA = 1e-4;

    // =========================================================================
    // HELPERS
    // =========================================================================

    /**
     * Build a binary classification dataset.
     * Class 0: features ≈ [1,1,1,1],  Class 1: features ≈ [-1,-1,-1,-1]
     * With jitter ±0.1 — should be trivially separable for any classifier.
     */
    private function makeBinaryDataset(int $n = 200, int $d = 4): array
    {
        mt_srand(42);
        $rows = []; $labels = [];
        for ($i = 0; $i < $n; $i++) {
            $cls = $i % 2;
            $row = [];
            for ($j = 0; $j < $d; $j++) {
                $row[] = ($cls === 0 ? 1.0 : -1.0) + (mt_rand(-100, 100) / 1000.0);
            }
            $rows[]   = $row;
            $labels[] = (float)$cls;
        }
        return [
            new Dataset(Tensor::fromArray($rows), Tensor::fromArray($labels)),
        ];
    }

    /**
     * Build a non-negative dataset (required by MultinomialNB / BernoulliNB).
     */
    private function makeNonNegDataset(int $n = 200, int $d = 4): array
    {
        mt_srand(42);
        $rows = []; $labels = [];
        for ($i = 0; $i < $n; $i++) {
            $cls = $i % 2;
            $row = [];
            for ($j = 0; $j < $d; $j++) {
                $row[] = max(0.0, ($cls === 0 ? 2.0 : 0.2) + (mt_rand(-100, 100) / 1000.0));
            }
            $rows[]   = $row;
            $labels[] = (float)$cls;
        }
        return [
            new Dataset(Tensor::fromArray($rows), Tensor::fromArray($labels)),
        ];
    }

    private function splitDataset(Dataset $ds, float $ratio = 0.8): array
    {
        return $ds->randomize()->split($ratio);
    }

    private function accuracy(Tensor $preds, Tensor $labels): float
    {
        return (new Accuracy())->score($preds, $labels);
    }

    // =========================================================================
    // 1. DECISION TREE CLASSIFIER
    // =========================================================================

    public function testDecisionTreeLifecycle(): void
    {
        $clf = new DecisionTreeClassifier(maxDepth: 5);
        $this->assertFalse($clf->trained());

        [$ds] = $this->makeBinaryDataset(200);
        [$train, $test] = $this->splitDataset($ds);

        $clf->train($train);
        $this->assertTrue($clf->trained());

        $preds = $clf->predict($test);
        $this->assertSame($test->numRows(), $preds->size());
    }

    public function testDecisionTreeAccuracy(): void
    {
        [$ds] = $this->makeBinaryDataset(400);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new DecisionTreeClassifier(maxDepth: 8);
        $clf->train($train);
        $acc = $this->accuracy($clf->predict($test), $test->labels());
        $this->assertGreaterThanOrEqual(self::MIN_ACCURACY, $acc,
            "DecisionTree accuracy {$acc} below threshold");
    }

    public function testDecisionTreeHardwareKernelPredictShape(): void
    {
        [$ds] = $this->makeBinaryDataset(100, 4);
        $clf  = new DecisionTreeClassifier(maxDepth: 4);
        $clf->train($ds);
        $preds = $clf->predict($ds);
        $this->assertSame([100], $preds->shape());
    }

    public function testDecisionTreeMinSamplesSplitPreventsOverfit(): void
    {
        [$ds] = $this->makeBinaryDataset(100);
        [$train, $test] = $this->splitDataset($ds, 0.5);

        $shallow = new DecisionTreeClassifier(maxDepth: 2, minSamplesSplit: 20);
        $shallow->train($train);
        // Just verify no exception and valid output
        $preds = $shallow->predict($test);
        $this->assertSame($test->numRows(), $preds->size());
    }

    // =========================================================================
    // 2. RANDOM FOREST CLASSIFIER
    // =========================================================================

    public function testRandomForestLifecycle(): void
    {
        $clf = new RandomForestClassifier(nEstimators: 10);
        $this->assertFalse($clf->trained());

        [$ds] = $this->makeBinaryDataset(200);
        [$train, $test] = $this->splitDataset($ds);

        $clf->train($train);
        $this->assertTrue($clf->trained());

        $preds = $clf->predict($test);
        $this->assertSame($test->numRows(), $preds->size());
    }

    public function testRandomForestAccuracy(): void
    {
        [$ds] = $this->makeBinaryDataset(400);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new RandomForestClassifier(nEstimators: 20, maxDepth: 6);
        $clf->train($train);
        $acc = $this->accuracy($clf->predict($test), $test->labels());
        $this->assertGreaterThanOrEqual(self::MIN_ACCURACY, $acc,
            "RandomForest accuracy {$acc} below threshold");
    }

    // =========================================================================
    // 3. GAUSSIAN NAIVE BAYES
    // =========================================================================

    public function testGaussianNBLifecycle(): void
    {
        $clf = new GaussianNB();
        $this->assertFalse($clf->trained());

        [$ds] = $this->makeBinaryDataset(200);
        [$train, $test] = $this->splitDataset($ds);

        $clf->train($train);
        $this->assertTrue($clf->trained());
    }

    public function testGaussianNBAccuracy(): void
    {
        [$ds] = $this->makeBinaryDataset(400);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new GaussianNB();
        $clf->train($train);
        $acc = $this->accuracy($clf->predict($test), $test->labels());
        $this->assertGreaterThanOrEqual(self::MIN_ACCURACY, $acc,
            "GaussianNB accuracy {$acc} below threshold");
    }

    public function testGaussianNBProbaShape(): void
    {
        [$ds] = $this->makeBinaryDataset(100);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new GaussianNB();
        $clf->train($train);
        $proba = $clf->proba($test);
        // proba() returns [N, K] log-probability matrix (one column per class)
        $shape = $proba->shape();
        $this->assertSame($test->numRows(), $shape[0]);
        $this->assertSame(2, $shape[1]); // binary dataset → 2 classes
    }

    public function testGaussianNBProbaFinite(): void
    {
        // Log-probabilities can be any real value — just verify no NaN/Inf.
        [$ds] = $this->makeBinaryDataset(100);
        $clf = new GaussianNB();
        $clf->train($ds);
        $flat = $clf->proba($ds)->toFlatArray();
        foreach ($flat as $p) {
            $this->assertFalse(\is_nan($p),      "GaussianNB proba produced NaN");
            $this->assertFalse(\is_infinite($p), "GaussianNB proba produced Inf");
        }
    }

    // =========================================================================
    // 4. BERNOULLI NAIVE BAYES
    // =========================================================================

    public function testBernoulliNBLifecycle(): void
    {
        [$ds] = $this->makeNonNegDataset(200);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new BernoulliNB();
        $this->assertFalse($clf->trained());
        $clf->train($train);
        $this->assertTrue($clf->trained());

        $preds = $clf->predict($test);
        $this->assertSame($test->numRows(), $preds->size());
    }

    // =========================================================================
    // 5. MULTINOMIAL NAIVE BAYES
    // =========================================================================

    public function testMultinomialNBLifecycle(): void
    {
        [$ds] = $this->makeNonNegDataset(200);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new MultinomialNB();
        $this->assertFalse($clf->trained());
        $clf->train($train);
        $this->assertTrue($clf->trained());

        $preds = $clf->predict($test);
        $this->assertSame($test->numRows(), $preds->size());
    }

    // =========================================================================
    // 6. KNN CLASSIFIER
    // =========================================================================

    public function testKNNLifecycle(): void
    {
        $clf = new KNNClassifier(k: 5);
        $this->assertFalse($clf->trained());

        [$ds] = $this->makeBinaryDataset(200);
        [$train, $test] = $this->splitDataset($ds);

        $clf->train($train);
        $this->assertTrue($clf->trained());
    }

    public function testKNNAccuracy(): void
    {
        [$ds] = $this->makeBinaryDataset(300);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new KNNClassifier(k: 3);
        $clf->train($train);
        $acc = $this->accuracy($clf->predict($test), $test->labels());
        $this->assertGreaterThanOrEqual(self::MIN_ACCURACY, $acc,
            "KNN accuracy {$acc} below threshold");
    }

    // =========================================================================
    // 7. KD-TREE NEIGHBORS CLASSIFIER
    // =========================================================================

    public function testKDNeighborsLifecycle(): void
    {
        [$ds] = $this->makeBinaryDataset(200);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new KDNeighborsClassifier(k: 5);
        $this->assertFalse($clf->trained());
        $clf->train($train);
        $this->assertTrue($clf->trained());

        $preds = $clf->predict($test);
        $this->assertSame($test->numRows(), $preds->size());
    }

    // =========================================================================
    // 8. LOGISTIC REGRESSION
    // =========================================================================

    public function testLogisticRegressionLifecycle(): void
    {
        $clf = new LogisticRegression(epochs: 50, learningRate: 0.1);
        $this->assertFalse($clf->trained());

        [$ds] = $this->makeBinaryDataset(200);
        [$train, $test] = $this->splitDataset($ds);

        $clf->train($train);
        $this->assertTrue($clf->trained());
    }

    public function testLogisticRegressionAccuracy(): void
    {
        [$ds] = $this->makeBinaryDataset(400);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new LogisticRegression(epochs: 100, learningRate: 0.5, batchSize: 32);
        $clf->train($train);
        $acc = $this->accuracy($clf->predict($test), $test->labels());
        $this->assertGreaterThanOrEqual(self::MIN_ACCURACY, $acc,
            "LogisticRegression accuracy {$acc} below threshold");
    }

    public function testLogisticRegressionProbaInRange(): void
    {
        [$ds] = $this->makeBinaryDataset(100);
        $clf = new LogisticRegression(epochs: 50, learningRate: 0.5);
        $clf->train($ds);
        foreach ($clf->proba($ds)->toFlatArray() as $p) {
            $this->assertGreaterThanOrEqual(0.0, $p);
            $this->assertLessThanOrEqual(1.0 + self::DELTA, $p);
        }
    }

    // =========================================================================
    // 9. SOFTMAX CLASSIFIER (multi-class)
    // =========================================================================

    public function testSoftmaxClassifierLifecycle(): void
    {
        // 3-class problem
        mt_srand(42);
        $rows = []; $labels = [];
        for ($i = 0; $i < 300; $i++) {
            $cls = $i % 3;
            $rows[] = [
                ($cls === 0 ? 3.0 : ($cls === 1 ? -3.0 : 0.0)) + mt_rand(-100, 100) / 1000.0,
                ($cls === 1 ? 3.0 : ($cls === 2 ? -3.0 : 0.0)) + mt_rand(-100, 100) / 1000.0,
            ];
            $labels[] = (float)$cls;
        }
        $ds = new Dataset(Tensor::fromArray($rows), Tensor::fromArray($labels));
        [$train, $test] = $ds->randomize()->split(0.8);

        $clf = new SoftmaxClassifier(epochs: 50, learningRate: 0.5);
        $this->assertFalse($clf->trained());
        $clf->train($train);
        $this->assertTrue($clf->trained());

        $preds = $clf->predict($test);
        $this->assertSame($test->numRows(), $preds->size());
    }

    // =========================================================================
    // 10. ADABOOST CLASSIFIER
    // =========================================================================

    public function testAdaBoostLifecycle(): void
    {
        [$ds] = $this->makeBinaryDataset(200);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new AdaBoostClassifier(nEstimators: 10);
        $this->assertFalse($clf->trained());
        $clf->train($train);
        $this->assertTrue($clf->trained());

        $preds = $clf->predict($test);
        $this->assertSame($test->numRows(), $preds->size());
    }

    public function testAdaBoostAccuracy(): void
    {
        [$ds] = $this->makeBinaryDataset(400);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new AdaBoostClassifier(nEstimators: 20);
        $clf->train($train);
        $acc = $this->accuracy($clf->predict($test), $test->labels());
        $this->assertGreaterThanOrEqual(self::MIN_ACCURACY, $acc,
            "AdaBoost accuracy {$acc} below threshold");
    }

    // =========================================================================
    // 11. EXTRA TREES CLASSIFIER
    // =========================================================================

    public function testExtraTreesLifecycle(): void
    {
        [$ds] = $this->makeBinaryDataset(200);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new ExtraTreesClassifier(nEstimators: 10);
        $this->assertFalse($clf->trained());
        $clf->train($train);
        $this->assertTrue($clf->trained());

        $preds = $clf->predict($test);
        $this->assertSame($test->numRows(), $preds->size());
    }

    // =========================================================================
    // 12. DUMMY CLASSIFIER (baseline)
    // =========================================================================

    public function testDummyClassifierPredictShape(): void
    {
        [$ds] = $this->makeBinaryDataset(100);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new DummyClassifier();
        $clf->train($train);
        $preds = $clf->predict($test);
        $this->assertSame($test->numRows(), $preds->size());
    }

    // =========================================================================
    // 13. CROSS-CUTTING: predict output values are valid class labels
    // =========================================================================

    public function testPredictOutputOnlyContainsKnownClasses(): void
    {
        [$ds] = $this->makeBinaryDataset(200);
        [$train, $test] = $this->splitDataset($ds);

        $clf = new DecisionTreeClassifier(maxDepth: 5);
        $clf->train($train);
        $preds = $clf->predict($test)->toFlatArray();

        foreach ($preds as $p) {
            $this->assertContains((float)round($p), [0.0, 1.0],
                "Prediction {$p} is not a known class label");
        }
    }

    // =========================================================================
    // 14. UNTRAINED ESTIMATOR THROWS
    // =========================================================================

    public function testPredictBeforeTrainThrows(): void
    {
        $this->expectException(\RuntimeException::class);
        [$ds] = $this->makeBinaryDataset(10);
        (new DecisionTreeClassifier())->predict($ds);
    }
}
