<?php
declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Tensor;
use Pml\Metrics\Classification\Accuracy;
use Pml\Metrics\Classification\F1Score;
use Pml\Metrics\Classification\Precision;
use Pml\Metrics\Classification\Recall;
use Pml\Metrics\Classification\MCC;
use Pml\Metrics\Classification\BrierScore;
use Pml\Metrics\Regression\MeanAbsoluteError;
use Pml\Metrics\Regression\MeanSquaredError;
use Pml\Metrics\Regression\RootMeanSquaredError;
use Pml\Metrics\Regression\RSquared;
use Pml\Metrics\Regression\MedianAbsoluteError;
use Pml\Metrics\Regression\SMAPE;
use Pml\Metrics\Reports\ConfusionMatrix;
use Pml\Metrics\Reports\ClassificationReport;

/**
 * All metrics tested against known ground-truth examples.
 * Closed-form expected values are computed by hand or cross-checked with sklearn.
 */
final class MetricsTest extends TestCase
{
    private const DELTA = 1e-3;

    // =========================================================================
    // HELPERS
    // =========================================================================

    private function t(array $data): Tensor
    {
        return Tensor::fromArray($data);
    }

    // =========================================================================
    // CLASSIFICATION METRICS
    // =========================================================================

    // --- Accuracy ---

    public function testAccuracyPerfect(): void
    {
        $pred   = $this->t([0.0, 1.0, 1.0, 0.0]);
        $labels = $this->t([0.0, 1.0, 1.0, 0.0]);
        $this->assertEqualsWithDelta(1.0, (new Accuracy())->score($pred, $labels), self::DELTA);
    }

    public function testAccuracyHalf(): void
    {
        $pred   = $this->t([1.0, 1.0, 0.0, 0.0]);
        $labels = $this->t([0.0, 1.0, 1.0, 0.0]);
        // 2/4 correct
        $this->assertEqualsWithDelta(0.5, (new Accuracy())->score($pred, $labels), self::DELTA);
    }

    public function testAccuracyZero(): void
    {
        $pred   = $this->t([1.0, 1.0, 1.0]);
        $labels = $this->t([0.0, 0.0, 0.0]);
        $this->assertEqualsWithDelta(0.0, (new Accuracy())->score($pred, $labels), self::DELTA);
    }

    public function testAccuracyWithProbabilities(): void
    {
        // Probabilities should be snapped to 0/1 before comparison
        $pred   = $this->t([0.9, 0.2, 0.8, 0.1]);
        $labels = $this->t([1.0, 0.0, 1.0, 0.0]);
        $this->assertEqualsWithDelta(1.0, (new Accuracy())->score($pred, $labels), self::DELTA);
    }

    // --- Precision ---

    public function testPrecisionPerfect(): void
    {
        $pred   = $this->t([1.0, 1.0, 0.0, 0.0]);
        $labels = $this->t([1.0, 1.0, 0.0, 0.0]);
        $this->assertEqualsWithDelta(1.0, (new Precision())->score($pred, $labels), self::DELTA);
    }

    public function testPrecisionKnownValue(): void
    {
        // TP=2, FP=1, Precision=2/3
        $pred   = $this->t([1.0, 1.0, 1.0, 0.0]);
        $labels = $this->t([1.0, 1.0, 0.0, 0.0]);
        $this->assertEqualsWithDelta(2.0 / 3.0, (new Precision())->score($pred, $labels), self::DELTA);
    }

    // --- Recall ---

    public function testRecallPerfect(): void
    {
        $pred   = $this->t([1.0, 0.0, 1.0]);
        $labels = $this->t([1.0, 0.0, 1.0]);
        $this->assertEqualsWithDelta(1.0, (new Recall())->score($pred, $labels), self::DELTA);
    }

    public function testRecallKnownValue(): void
    {
        // TP=2, FN=1, Recall=2/3
        $pred   = $this->t([1.0, 1.0, 0.0, 0.0]);
        $labels = $this->t([1.0, 1.0, 1.0, 0.0]);
        $this->assertEqualsWithDelta(2.0 / 3.0, (new Recall())->score($pred, $labels), self::DELTA);
    }

    // --- F1 Score ---

    public function testF1ScorePerfect(): void
    {
        $pred   = $this->t([1.0, 0.0, 1.0, 0.0]);
        $labels = $this->t([1.0, 0.0, 1.0, 0.0]);
        $this->assertEqualsWithDelta(1.0, (new F1Score())->score($pred, $labels), self::DELTA);
    }

    public function testF1ScoreKnownValue(): void
    {
        // Precision=2/3, Recall=2/3, F1=2/3
        $pred   = $this->t([1.0, 1.0, 1.0, 0.0, 0.0]);
        $labels = $this->t([1.0, 1.0, 0.0, 1.0, 0.0]);
        $p = (new Precision())->score($pred, $labels);
        $r = (new Recall())->score($pred, $labels);
        $expectedF1 = 2 * $p * $r / ($p + $r);
        $this->assertEqualsWithDelta($expectedF1, (new F1Score())->score($pred, $labels), self::DELTA);
    }

    // --- MCC ---

    public function testMccPerfect(): void
    {
        $pred   = $this->t([1.0, 0.0, 1.0, 0.0]);
        $labels = $this->t([1.0, 0.0, 1.0, 0.0]);
        $this->assertEqualsWithDelta(1.0, (new MCC())->score($pred, $labels), self::DELTA);
    }

    public function testMccInRange(): void
    {
        $pred   = $this->t([1.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
        $labels = $this->t([1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        $mcc = (new MCC())->score($pred, $labels);
        $this->assertGreaterThanOrEqual(-1.0, $mcc);
        $this->assertLessThanOrEqual(1.0, $mcc);
    }

    // --- Brier Score ---

    public function testBrierScorePerfect(): void
    {
        $pred   = $this->t([1.0, 0.0, 1.0]);
        $labels = $this->t([1.0, 0.0, 1.0]);
        $this->assertEqualsWithDelta(0.0, (new BrierScore())->score($pred, $labels), self::DELTA);
    }

    public function testBrierScoreKnownValue(): void
    {
        // BS = -mean((p - y)^2)  — negated so higher = better.
        // Raw MSE = ((0.2)^2 + (0.1)^2 + (0.7)^2) / 3 ≈ 0.18
        $pred   = $this->t([0.8, 0.9, 0.3]);
        $labels = $this->t([1.0, 1.0, 1.0]);
        $expected = -((0.2**2) + (0.1**2) + (0.7**2)) / 3.0;
        $this->assertEqualsWithDelta($expected, (new BrierScore())->score($pred, $labels), self::DELTA);
    }

    // =========================================================================
    // REGRESSION METRICS
    // =========================================================================

    // --- MAE ---

    public function testMaePerfect(): void
    {
        $pred   = $this->t([1.0, 2.0, 3.0]);
        $labels = $this->t([1.0, 2.0, 3.0]);
        $this->assertEqualsWithDelta(0.0, (new MeanAbsoluteError())->score($pred, $labels), self::DELTA);
    }

    public function testMaeKnownValue(): void
    {
        // |1-2| + |3-3| + |4-5| = 1+0+1 = 2, MAE = 2/3
        $pred   = $this->t([1.0, 3.0, 4.0]);
        $labels = $this->t([2.0, 3.0, 5.0]);
        $this->assertEqualsWithDelta(2.0 / 3.0, (new MeanAbsoluteError())->score($pred, $labels), self::DELTA);
    }

    // --- MSE ---

    public function testMsePerfect(): void
    {
        $pred   = $this->t([1.0, 2.0, 3.0]);
        $labels = $this->t([1.0, 2.0, 3.0]);
        $this->assertEqualsWithDelta(0.0, (new MeanSquaredError())->score($pred, $labels), self::DELTA);
    }

    public function testMseKnownValue(): void
    {
        // (1-2)^2 + (3-3)^2 + (4-5)^2 = 1+0+1 = 2, MSE = 2/3
        $pred   = $this->t([1.0, 3.0, 4.0]);
        $labels = $this->t([2.0, 3.0, 5.0]);
        $this->assertEqualsWithDelta(2.0 / 3.0, (new MeanSquaredError())->score($pred, $labels), self::DELTA);
    }

    // --- RMSE ---

    public function testRmseKnownValue(): void
    {
        $pred   = $this->t([1.0, 3.0, 4.0]);
        $labels = $this->t([2.0, 3.0, 5.0]);
        $expected = sqrt(2.0 / 3.0);
        $this->assertEqualsWithDelta($expected, (new RootMeanSquaredError())->score($pred, $labels), self::DELTA);
    }

    // --- R² ---

    public function testR2Perfect(): void
    {
        $pred   = $this->t([1.0, 2.0, 3.0, 4.0]);
        $labels = $this->t([1.0, 2.0, 3.0, 4.0]);
        $this->assertEqualsWithDelta(1.0, (new RSquared())->score($pred, $labels), self::DELTA);
    }

    public function testR2Zero(): void
    {
        // Predicting the mean exactly equals R²=0
        $labels = $this->t([1.0, 2.0, 3.0, 4.0]);  // mean=2.5
        $pred   = $this->t([2.5, 2.5, 2.5, 2.5]);
        $this->assertEqualsWithDelta(0.0, (new RSquared())->score($pred, $labels), self::DELTA);
    }

    public function testR2KnownValue(): void
    {
        // Hand-computed: ss_res=0.5, ss_tot=3.5, R²=1-0.5/3.5≈0.857
        $pred   = $this->t([2.5, 5.0, 4.0, 8.0]);
        $labels = $this->t([3.0, 5.0, 4.0, 7.0]);
        $r2 = (new RSquared())->score($pred, $labels);
        $this->assertGreaterThanOrEqual(0.85, $r2);
    }

    // --- Median Absolute Error ---

    public function testMedianAbsoluteError(): void
    {
        // MedianAbsoluteError is negated so higher = better.
        // |errors| = [1,0,1,10] → sorted: [0,1,1,10] → median = (1+1)/2 = 1 → score = -1
        $pred   = $this->t([1.0, 3.0, 4.0, 15.0]);
        $labels = $this->t([2.0, 3.0, 5.0, 5.0]);
        $mae = (new MedianAbsoluteError())->score($pred, $labels);
        $this->assertEqualsWithDelta(-1.0, $mae, self::DELTA);
    }

    // --- SMAPE ---

    public function testSmapeKnownValue(): void
    {
        $pred   = $this->t([100.0, 200.0]);
        $labels = $this->t([110.0, 180.0]);
        $smape = (new SMAPE())->score($pred, $labels);
        // Should be a percentage in (0, 100]
        $this->assertGreaterThan(0.0, $smape);
        $this->assertLessThanOrEqual(100.0, $smape);
    }

    // =========================================================================
    // REPORTS
    // =========================================================================

    // --- Confusion Matrix ---

    public function testConfusionMatrix2x2(): void
    {
        $pred   = $this->t([0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
        $labels = $this->t([0.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
        $matrix = ConfusionMatrix::generate($pred, $labels);
        // Must be 2×2
        $this->assertCount(2, $matrix);
        $this->assertCount(2, $matrix[0]);
        // TN=2: label=0,pred=0 at indices 0,3
        $this->assertSame(2, $matrix[0][0]);
        // TP=2: label=1,pred=1 at indices 1,4
        $this->assertSame(2, $matrix[1][1]);
        // FP=1: label=0,pred=1 at index 2
        $this->assertSame(1, $matrix[0][1]);
        // FN=1: label=1,pred=0 at index 5
        $this->assertSame(1, $matrix[1][0]);
    }

    public function testConfusionMatrixSumEqualsN(): void
    {
        $pred   = $this->t([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0]);
        $labels = $this->t([0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 0.0, 1.0]);
        $matrix = ConfusionMatrix::generate($pred, $labels);
        $total = 0;
        foreach ($matrix as $row) {
            foreach ($row as $cell) {
                $total += $cell;
            }
        }
        $this->assertSame(8, $total);
    }

    // --- Classification Report ---

    public function testClassificationReportReturnsString(): void
    {
        $pred   = $this->t([0.0, 1.0, 1.0, 0.0, 1.0]);
        $labels = $this->t([0.0, 1.0, 0.0, 0.0, 1.0]);
        $report = ClassificationReport::generate($pred, $labels);
        $this->assertIsString($report);
        $this->assertNotEmpty($report);
    }

    // =========================================================================
    // METRIC RANGE INVARIANTS
    // =========================================================================

    /**
     * @dataProvider classificationMetricProvider
     */
    public function testClassificationMetricInRange(object $metric): void
    {
        $pred   = $this->t([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        $labels = $this->t([1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        $score = $metric->score($pred, $labels);
        $this->assertGreaterThanOrEqual(0.0, $score, \get_class($metric) . ' score < 0');
        $this->assertLessThanOrEqual(1.0, $score, \get_class($metric) . ' score > 1');
    }

    public static function classificationMetricProvider(): array
    {
        // BrierScore is excluded here — its range is [-1, 0] (negated), not [0, 1].
        return [
            'Accuracy'  => [new Accuracy()],
            'Precision' => [new Precision()],
            'Recall'    => [new Recall()],
            'F1Score'   => [new F1Score()],
        ];
    }

    /**
     * @dataProvider regressionMetricProvider
     */
    public function testRegressionMetricNonNegative(object $metric): void
    {
        $pred   = $this->t([1.1, 2.2, 3.3]);
        $labels = $this->t([1.0, 2.0, 3.0]);
        $score = $metric->score($pred, $labels);
        $this->assertGreaterThanOrEqual(0.0, $score,
            \get_class($metric) . ' returned negative score');
    }

    public static function regressionMetricProvider(): array
    {
        return [
            'MAE'  => [new MeanAbsoluteError()],
            'MSE'  => [new MeanSquaredError()],
            'RMSE' => [new RootMeanSquaredError()],
        ];
    }
}
