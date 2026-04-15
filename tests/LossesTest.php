<?php
declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Tensor;
use Pml\Losses\MeanSquaredError;
use Pml\Losses\BinaryCrossEntropy;
use Pml\Losses\CategoricalCrossEntropy;
use Pml\Losses\Huber;
use Pml\Losses\Hinge;

/**
 * Comprehensive test suite for Loss functions.
 */
final class LossesTest extends TestCase
{
    private const DELTA = 1e-4;

    // =========================================================================
    // 1. MEAN SQUARED ERROR
    // =========================================================================

    public function testMeanSquaredErrorPerfect(): void
    {
        $preds = Tensor::fromArray([1.0, 2.0, 3.0]);
        $targets = Tensor::fromArray([1.0, 2.0, 3.0]);
        
        $loss = new MeanSquaredError();
        $result = $loss->compute($preds, $targets);
        
        $this->assertEqualsWithDelta(0.0, $result, self::DELTA);
    }

    public function testMeanSquaredErrorKnownValue(): void
    {
        $preds = Tensor::fromArray([1.0, 2.0, 3.0]);
        $targets = Tensor::fromArray([2.0, 2.0, 4.0]);
        
        $loss = new MeanSquaredError();
        $result = $loss->compute($preds, $targets);
        
        // MSE = ((1-2)^2 + (2-2)^2 + (3-4)^2) / 3 = (1 + 0 + 1) / 3 = 2/3
        $this->assertEqualsWithDelta(2.0 / 3.0, $result, self::DELTA);
    }

    public function testMeanSquaredErrorAlwaysNonNegative(): void
    {
        $preds = Tensor::fromArray([1.0, 2.0, 3.0]);
        $targets = Tensor::fromArray([5.0, 6.0, 7.0]);
        
        $loss = new MeanSquaredError();
        $result = $loss->compute($preds, $targets);
        
        $this->assertGreaterThanOrEqual(0.0, $result);
    }

    // =========================================================================
    // 2. BINARY CROSS ENTROPY
    // =========================================================================

    public function testBinaryCrossEntropyPerfect(): void
    {
        $preds = Tensor::fromArray([1.0, 0.0, 1.0, 0.0]);
        $targets = Tensor::fromArray([1.0, 0.0, 1.0, 0.0]);
        
        $loss = new BinaryCrossEntropy();
        $result = $loss->compute($preds, $targets);
        
        $this->assertEqualsWithDelta(0.0, $result, self::DELTA);
    }

    public function testBinaryCrossEntropyWithProbabilities(): void
    {
        $preds = Tensor::fromArray([0.9, 0.1, 0.8, 0.2]);
        $targets = Tensor::fromArray([1.0, 0.0, 1.0, 0.0]);
        
        $loss = new BinaryCrossEntropy();
        $result = $loss->compute($preds, $targets);
        
        // BCE = -mean(y*log(p) + (1-y)*log(1-p))
        // Should be small positive value
        $this->assertGreaterThan(0.0, $result);
        $this->assertLessThan(1.0, $result);
    }

    public function testBinaryCrossEntropyWorstCase(): void
    {
        // When predictions are completely wrong (predict 0 when target is 1)
        $preds = Tensor::fromArray([0.001, 0.001, 0.001]);
        $targets = Tensor::fromArray([1.0, 1.0, 1.0]);
        
        $loss = new BinaryCrossEntropy();
        $result = $loss->compute($preds, $targets);
        
        // Should be large positive value (high loss)
        $this->assertGreaterThan(5.0, $result);
    }

    // =========================================================================
    // 3. CATEGORICAL CROSS ENTROPY
    // =========================================================================

    public function testCategoricalCrossEntropyPerfect(): void
    {
        // One-hot predictions matching targets exactly
        $preds = Tensor::fromArray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        $targets = Tensor::fromArray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        
        $loss = new CategoricalCrossEntropy();
        $result = $loss->compute($preds, $targets);
        
        $this->assertEqualsWithDelta(0.0, $result, self::DELTA);
    }

    public function testCategoricalCrossEntropyWithSoftmax(): void
    {
        // Simulate softmax outputs
        $preds = Tensor::fromArray([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]);
        $targets = Tensor::fromArray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        
        $loss = new CategoricalCrossEntropy();
        $result = $loss->compute($preds, $targets);
        
        // CCE = -mean(sum(targets * log(preds)))
        $this->assertGreaterThan(0.0, $result);
        $this->assertLessThan(2.0, $result);
    }

    // =========================================================================
    // 4. HUBER LOSS
    // =========================================================================

    public function testHuberLossPerfect(): void
    {
        $preds = Tensor::fromArray([1.0, 2.0, 3.0]);
        $targets = Tensor::fromArray([1.0, 2.0, 3.0]);
        
        $loss = new Huber(delta: 1.0);
        $result = $loss->compute($preds, $targets);
        
        $this->assertEqualsWithDelta(0.0, $result, self::DELTA);
    }

    public function testHuberLossSmallErrorsQuadratic(): void
    {
        // For errors < delta, Huber loss is quadratic (like MSE)
        $preds = Tensor::fromArray([1.0, 2.0, 3.0]);
        $targets = Tensor::fromArray([1.5, 2.5, 3.5]);  // errors = 0.5
        
        $loss = new Huber(delta: 1.0);
        $result = $loss->compute($preds, $targets);
        
        // For small errors: 0.5 * error^2 = 0.5 * 0.25 = 0.125
        $this->assertEqualsWithDelta(0.125, $result, self::DELTA);
    }

    public function testHuberLossLargeErrorsLinear(): void
    {
        // For errors > delta, Huber loss is linear (like MAE)
        $preds = Tensor::fromArray([1.0]);
        $targets = Tensor::fromArray([5.0]);  // error = 4.0
        
        $loss = new Huber(delta: 1.0);
        $result = $loss->compute($preds, $targets);
        
        // For large errors: delta * (|error| - 0.5 * delta) = 1 * (4 - 0.5) = 3.5
        $this->assertEqualsWithDelta(3.5, $result, self::DELTA);
    }

    // =========================================================================
    // 5. HINGE LOSS
    // =========================================================================

    public function testHingeLossPerfect(): void
    {
        // Perfect classification with margin > 1
        $preds = Tensor::fromArray([2.0, -2.0, 3.0]);
        $targets = Tensor::fromArray([1.0, -1.0, 1.0]);
        
        $loss = new Hinge();
        $result = $loss->compute($preds, $targets);
        
        // Hinge loss = max(0, 1 - y * pred)
        // All y * pred > 1, so loss = 0
        $this->assertEqualsWithDelta(0.0, $result, self::DELTA);
    }

    public function testHingeLossWithMisclassification(): void
    {
        $preds = Tensor::fromArray([1.0, -1.0]);
        $targets = Tensor::fromArray([1.0, 1.0]);  // Second sample misclassified
        
        $loss = new Hinge();
        $result = $loss->compute($preds, $targets);
        
        // Sample 1: max(0, 1 - 1*1) = 0
        // Sample 2: max(0, 1 - 1*(-1)) = max(0, 2) = 2
        // Mean = (0 + 2) / 2 = 1
        $this->assertEqualsWithDelta(1.0, $result, self::DELTA);
    }

    public function testHingeLossAlwaysNonNegative(): void
    {
        $preds = Tensor::fromArray([-5.0, -3.0, 2.0]);
        $targets = Tensor::fromArray([1.0, 1.0, -1.0]);
        
        $loss = new Hinge();
        $result = $loss->compute($preds, $targets);
        
        $this->assertGreaterThanOrEqual(0.0, $result);
    }

    // =========================================================================
    // 6. EDGE CASES
    // =========================================================================

    public function testMeanSquaredErrorSingleValue(): void
    {
        $preds = Tensor::fromArray([3.0]);
        $targets = Tensor::fromArray([5.0]);
        
        $loss = new MeanSquaredError();
        $result = $loss->compute($preds, $targets);
        
        $this->assertEqualsWithDelta(4.0, $result, self::DELTA);
    }

    public function testBinaryCrossEntropyHandlesEdgeProbabilities(): void
    {
        // Test with probabilities very close to 0 and 1
        $preds = Tensor::fromArray([0.999, 0.001]);
        $targets = Tensor::fromArray([1.0, 0.0]);
        
        $loss = new BinaryCrossEntropy();
        $result = $loss->compute($preds, $targets);
        
        $this->assertIsFloat($result);
        $this->assertGreaterThan(0.0, $result);
    }

    public function testCategoricalCrossEntropySingleClass(): void
    {
        $preds = Tensor::fromArray([[0.8], [0.2]]);
        $targets = Tensor::fromArray([[1.0], [0.0]]);
        
        $loss = new CategoricalCrossEntropy();
        $result = $loss->compute($preds, $targets);
        
        $this->assertIsFloat($result);
    }

    // =========================================================================
    // 7. DIFFERENTIATE METHODS
    // =========================================================================

    public function testMeanSquaredErrorDifferentiate(): void
    {
        $preds = Tensor::fromArray([1.0, 2.0, 3.0]);
        $targets = Tensor::fromArray([2.0, 2.0, 4.0]);
        
        $loss = new MeanSquaredError();
        $grad = $loss->differentiate($preds, $targets);
        
        // d(MSE)/d(pred) = 2 * (pred - target) / n
        // = 2 * ([-1, 0, -1]) / 3 = [-2/3, 0, -2/3]
        $expected = [-2.0/3.0, 0.0, -2.0/3.0];
        $actual = $grad->toFlatArray();
        
        foreach ($expected as $i => $exp) {
            $this->assertEqualsWithDelta($exp, $actual[$i], self::DELTA);
        }
    }

    public function testBinaryCrossEntropyDifferentiate(): void
    {
        $preds = Tensor::fromArray([0.8, 0.2]);
        $targets = Tensor::fromArray([1.0, 0.0]);
        
        $loss = new BinaryCrossEntropy();
        $grad = $loss->differentiate($preds, $targets);
        
        // d(BCE)/d(pred) = (pred - target) / (pred * (1 - pred) * N)
        // For pred=0.8, target=1: (0.8 - 1) / (0.8 * 0.2 * 2) = -0.2 / 0.32 = -0.625
        // For pred=0.2, target=0: (0.2 - 0) / (0.2 * 0.8 * 2) = 0.2 / 0.32 = 0.625
        $actual = $grad->toFlatArray();
        
        $this->assertEqualsWithDelta(-0.625, $actual[0], 0.01);
        $this->assertEqualsWithDelta(0.625, $actual[1], 0.01);
    }

    public function testHingeLossDifferentiate(): void
    {
        $preds = Tensor::fromArray([0.5, -0.5]);
        $targets = Tensor::fromArray([1.0, 1.0]);
        
        $loss = new Hinge();
        $grad = $loss->differentiate($preds, $targets);
        
        // d(hinge)/d(pred) = -y if y*pred < 1, else 0, then divided by N
        // For pred=0.5, y=1: -1 / 2 = -0.5 (since 0.5 < 1)
        // For pred=-0.5, y=1: -1 / 2 = -0.5 (since -0.5 < 1)
        $actual = $grad->toFlatArray();
        
        $this->assertEqualsWithDelta(-0.5, $actual[0], 0.01);
        $this->assertEqualsWithDelta(-0.5, $actual[1], 0.01);
    }
}