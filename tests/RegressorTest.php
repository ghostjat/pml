<?php
declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Dataset;
use Pml\Tensor;
use Pml\Estimators\Regression\LinearRegression;
use Pml\Estimators\Regression\Ridge;
use Pml\Estimators\Regression\Lasso;
use Pml\Estimators\Regression\ElasticNet;
use Pml\Estimators\Regression\DecisionTreeRegressor;
use Pml\Estimators\Regression\GradientBoostingRegressor;
use Pml\Estimators\Regression\KNNRegressor;
use Pml\Estimators\Regression\KDNeighborsRegressor;
use Pml\Estimators\Regression\DummyRegressor;
use Pml\Estimators\Regression\SVR;
use Pml\Estimators\Regression\ExtraTreeRegressor;
use Pml\Metrics\Regression\RSquared;
use Pml\Metrics\Regression\MeanSquaredError;

/**
 * Lifecycle + R² / shape tests for every regressor.
 *
 * Design contract:
 * - Every estimator must implement train() / predict() / trained()
 * - R² on a linear toy problem must be ≥ 0.85 for linear models, ≥ 0.70 for tree/knn
 * - predict() output size must match dataset row count
 */
final class RegressorTest extends TestCase
{
    private const MIN_R2_LINEAR = 0.85;
    private const MIN_R2_TREE   = 0.70;

    // =========================================================================
    // HELPERS
    // =========================================================================

    /**
     * y = 2*x0 + 3*x1 + noise.
     * Well-conditioned, low-noise linear regression problem.
     */
    private function makeLinearDataset(int $n = 300, int $d = 3, float $noise = 0.05): array
    {
        mt_srand(42);
        $rows = []; $y = [];
        for ($i = 0; $i < $n; $i++) {
            $x = [];
            for ($j = 0; $j < $d; $j++) {
                $x[] = ($i + $j) / $n;
            }
            $rows[] = $x;
            $y[]    = 2.0 * $x[0] + 3.0 * $x[1] + (mt_rand(-100, 100) / 100.0) * $noise;
        }
        return [
            new Dataset(Tensor::fromArray($rows), Tensor::fromArray($y)),
        ];
    }

    private function r2(Tensor $preds, Tensor $labels): float
    {
        return (new RSquared())->score($preds, $labels);
    }

    private function mse(Tensor $preds, Tensor $labels): float
    {
        return (new MeanSquaredError())->score($preds, $labels);
    }

    private function split(Dataset $ds, float $ratio = 0.8): array
    {
        return $ds->randomize()->split($ratio);
    }

    // =========================================================================
    // 1. LINEAR REGRESSION (OLS / Pseudo-Inverse)
    // =========================================================================

    public function testLinearRegressionLifecycle(): void
    {
        $reg = new LinearRegression();
        $this->assertFalse($reg->trained());

        [$ds] = $this->makeLinearDataset();
        [$train, $test] = $this->split($ds);

        $reg->train($train);
        $this->assertTrue($reg->trained());
    }

    public function testLinearRegressionPredictShape(): void
    {
        [$ds] = $this->makeLinearDataset(100);
        [$train, $test] = $this->split($ds);

        $reg = new LinearRegression();
        $reg->train($train);
        $preds = $reg->predict($test);
        $this->assertSame($test->numRows(), $preds->size());
    }

    public function testLinearRegressionR2(): void
    {
        [$ds] = $this->makeLinearDataset(400);
        [$train, $test] = $this->split($ds);

        $reg = new LinearRegression();
        $reg->train($train);
        $r2 = $this->r2($reg->predict($test), $test->labels());
        $this->assertGreaterThanOrEqual(self::MIN_R2_LINEAR, $r2,
            "LinearRegression R² {$r2} below threshold");
    }

    public function testLinearRegressionMseLow(): void
    {
        [$ds] = $this->makeLinearDataset(400);
        [$train, $test] = $this->split($ds);

        $reg = new LinearRegression();
        $reg->train($train);
        $mse = $this->mse($reg->predict($test), $test->labels());
        $this->assertLessThan(0.1, $mse, "LinearRegression MSE {$mse} unexpectedly high");
    }

    // =========================================================================
    // 2. RIDGE REGRESSION (L2)
    // =========================================================================

    public function testRidgeLifecycle(): void
    {
        $reg = new Ridge();
        $this->assertFalse($reg->trained());

        [$ds] = $this->makeLinearDataset(200);
        [$train, $test] = $this->split($ds);

        $reg->train($train);
        $this->assertTrue($reg->trained());
        $this->assertSame($test->numRows(), $reg->predict($test)->size());
    }

    public function testRidgeR2(): void
    {
        [$ds] = $this->makeLinearDataset(400);
        [$train, $test] = $this->split($ds);

        $reg = new Ridge(alpha: 0.01);
        $reg->train($train);
        $r2 = $this->r2($reg->predict($test), $test->labels());
        $this->assertGreaterThanOrEqual(self::MIN_R2_LINEAR, $r2,
            "Ridge R² {$r2} below threshold");
    }

    // =========================================================================
    // 3. LASSO (L1)
    // =========================================================================

    public function testLassoLifecycle(): void
    {
        [$ds] = $this->makeLinearDataset(200);
        [$train, $test] = $this->split($ds);

        $reg = new Lasso();
        $this->assertFalse($reg->trained());
        $reg->train($train);
        $this->assertTrue($reg->trained());
        $this->assertSame($test->numRows(), $reg->predict($test)->size());
    }

    // =========================================================================
    // 4. ELASTIC NET (L1+L2)
    // =========================================================================

    public function testElasticNetLifecycle(): void
    {
        [$ds] = $this->makeLinearDataset(200);
        [$train, $test] = $this->split($ds);

        $reg = new ElasticNet();
        $this->assertFalse($reg->trained());
        $reg->train($train);
        $this->assertTrue($reg->trained());
        $this->assertSame($test->numRows(), $reg->predict($test)->size());
    }

    // =========================================================================
    // 5. DECISION TREE REGRESSOR
    // =========================================================================

    public function testDecisionTreeRegressorLifecycle(): void
    {
        $reg = new DecisionTreeRegressor(maxDepth: 5);
        $this->assertFalse($reg->trained());

        [$ds] = $this->makeLinearDataset(300);
        [$train, $test] = $this->split($ds);

        $reg->train($train);
        $this->assertTrue($reg->trained());
    }

    public function testDecisionTreeRegressorPredictShape(): void
    {
        [$ds] = $this->makeLinearDataset(100);
        $reg = new DecisionTreeRegressor(maxDepth: 4);
        $reg->train($ds);
        $this->assertSame([$ds->numRows()], $reg->predict($ds)->shape());
    }

    public function testDecisionTreeRegressorR2(): void
    {
        [$ds] = $this->makeLinearDataset(400);
        [$train, $test] = $this->split($ds);

        $reg = new DecisionTreeRegressor(maxDepth: 8);
        $reg->train($train);
        $r2 = $this->r2($reg->predict($test), $test->labels());
        $this->assertGreaterThanOrEqual(self::MIN_R2_TREE, $r2,
            "DecisionTreeRegressor R² {$r2} below threshold");
    }

    public function testDecisionTreeRegressorHardwareKernel(): void
    {
        // Verify the hardware-node path runs without error and produces valid floats
        [$ds] = $this->makeLinearDataset(200);
        $reg = new DecisionTreeRegressor(maxDepth: 6);
        $reg->train($ds);
        $preds = $reg->predict($ds)->toFlatArray();
        foreach ($preds as $p) {
            $this->assertIsFloat($p);
            $this->assertFalse(is_nan($p), "Prediction is NaN");
            $this->assertFalse(is_infinite($p), "Prediction is Inf");
        }
    }

    // =========================================================================
    // 6. GRADIENT BOOSTING REGRESSOR
    // =========================================================================

    public function testGradientBoostingLifecycle(): void
    {
        [$ds] = $this->makeLinearDataset(300);
        [$train, $test] = $this->split($ds);

        $reg = new GradientBoostingRegressor(nEstimators: 10);
        $this->assertFalse($reg->trained());
        $reg->train($train);
        $this->assertTrue($reg->trained());
        $this->assertSame($test->numRows(), $reg->predict($test)->size());
    }

    public function testGradientBoostingR2(): void
    {
        [$ds] = $this->makeLinearDataset(400);
        [$train, $test] = $this->split($ds);

        $reg = new GradientBoostingRegressor(nEstimators: 20);
        $reg->train($train);
        $r2 = $this->r2($reg->predict($test), $test->labels());
        $this->assertGreaterThanOrEqual(self::MIN_R2_TREE, $r2,
            "GradientBoosting R² {$r2} below threshold");
    }

    // =========================================================================
    // 7. KNN REGRESSOR
    // =========================================================================

    public function testKNNRegressorLifecycle(): void
    {
        [$ds] = $this->makeLinearDataset(200);
        [$train, $test] = $this->split($ds);

        $reg = new KNNRegressor(k: 5);
        $this->assertFalse($reg->trained());
        $reg->train($train);
        $this->assertTrue($reg->trained());
        $this->assertSame($test->numRows(), $reg->predict($test)->size());
    }

    // =========================================================================
    // 8. KD-TREE NEIGHBORS REGRESSOR
    // =========================================================================

    public function testKDNeighborsRegressorLifecycle(): void
    {
        [$ds] = $this->makeLinearDataset(200);
        [$train, $test] = $this->split($ds);

        $reg = new KDNeighborsRegressor(k: 5);
        $this->assertFalse($reg->trained());
        $reg->train($train);
        $this->assertTrue($reg->trained());
        $this->assertSame($test->numRows(), $reg->predict($test)->size());
    }

    // =========================================================================
    // 9. EXTRA TREE REGRESSOR
    // =========================================================================

    public function testExtraTreeRegressorLifecycle(): void
    {
        [$ds] = $this->makeLinearDataset(200);
        [$train, $test] = $this->split($ds);

        $reg = new ExtraTreeRegressor();
        $this->assertFalse($reg->trained());
        $reg->train($train);
        $this->assertTrue($reg->trained());
        $this->assertSame($test->numRows(), $reg->predict($test)->size());
    }

    // =========================================================================
    // 10. DUMMY REGRESSOR (baseline)
    // =========================================================================

    public function testDummyRegressorPredictShape(): void
    {
        [$ds] = $this->makeLinearDataset(100);
        [$train, $test] = $this->split($ds);

        $reg = new DummyRegressor();
        $reg->train($train);
        $this->assertSame($test->numRows(), $reg->predict($test)->size());
    }

    // =========================================================================
    // 11. CROSS-CUTTING: untrained predict throws
    // =========================================================================

    public function testPredictBeforeTrainThrows(): void
    {
        $this->expectException(\RuntimeException::class);
        [$ds] = $this->makeLinearDataset(10);
        (new LinearRegression())->predict($ds);
    }

    // =========================================================================
    // 12. CROSS-CUTTING: predictions are finite floats
    // =========================================================================

    /**
     * @dataProvider regressorProvider
     */
    public function testPredictionsAreFinite(object $reg): void
    {
        [$ds] = $this->makeLinearDataset(100);
        [$train, $test] = $this->split($ds);
        $reg->train($train);
        foreach ($reg->predict($test)->toFlatArray() as $p) {
            $this->assertFalse(\is_nan($p),       \get_class($reg) . ' produced NaN');
            $this->assertFalse(\is_infinite($p),  \get_class($reg) . ' produced Inf');
        }
    }

    public static function regressorProvider(): array
    {
        return [
            'LinearRegression'  => [new LinearRegression()],
            'Ridge'             => [new Ridge()],
            'DecisionTree'      => [new DecisionTreeRegressor(maxDepth: 4)],
            'KNN'               => [new KNNRegressor(k: 3)],
            'DummyRegressor'    => [new DummyRegressor()],
        ];
    }
}
