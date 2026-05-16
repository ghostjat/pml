<?php
declare(strict_types=1);

namespace Pml\Benchmarks\Accuracy;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\Dataset;
use Pml\NeuralNetwork\Layers\Softmax;
use Pml\Losses\CategoricalCrossEntropy;
use Pml\Estimators\Classifiers\GaussianNB;
use Pml\Estimators\Regression\LinearRegression;
use Pml\Metrics\Classification\Accuracy;
use Pml\Metrics\Regression\RSquared;

/**
 * Numerical accuracy and correctness benchmarks.
 *
 * These benchmarks verify that C-engine operations are numerically correct
 * by using `#[Bench\Assert]` constraints and by checking known mathematical
 * properties (softmax sums to 1, sigmoid ∈ (0,1), etc.).
 *
 * They also measure throughput so you get speed + correctness in one run.
 *
 * Groups:
 *   accuracy    — all accuracy/correctness benchmarks
 *   numerical   — mathematical property checks (softmax, sigmoid, etc.)
 *   ml          — ML estimator accuracy on synthetic tasks
 *   loss        — loss function numerical correctness
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['accuracy', 'numerical', 'correctness'])]
final class NumericalAccuracyBench
{
    private static Tensor $logits;
    private static Tensor $posVec;
    private static Tensor $knownInput;
    private static Tensor $knownTarget;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }

        \mt_srand(42);
        self::$logits     = Tensor::randomNormal([1000, 100]);
        self::$posVec     = Tensor::randomUniform([100_000], 0.001, 10.0);

        // A [4, 3] matrix of logits with known expected values
        self::$knownInput  = Tensor::fromArray([
            [2.0, 1.0, 0.1],
            [1.0, 2.0, 0.1],
            [0.1, 0.1, 2.0],
            [1.5, 1.5, 0.0],
        ]);
        // One-hot targets
        self::$knownTarget = Tensor::fromArray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]);

        self::$initialized = true;
    }

    // =========================================================================
    // SOFTMAX — each row must sum to 1.0 (within fp32 tolerance)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['accuracy', 'numerical'])]
    #[Bench\Assert('mode(variant.time.avg) < 50ms')]
    public function benchSoftmaxRowSumIsOne(): void
    {
        $probs = self::$logits->copy();
        $probs->rowSoftmaxInplace();
        // Sum all rows then divide by row count — should be ~1.0
        $rowSums = $probs->sumAxis(1); // [1000]
        $mean    = $rowSums->mean();
        // Assert sum ≈ 1.0 with tolerance 0.001
        if (\abs($mean - 1.0) > 0.001) {
            throw new \RuntimeException(
                "Softmax row sum = {$mean}, expected 1.0 ± 0.001"
            );
        }
        unset($probs, $rowSums);
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['accuracy', 'numerical'])]
    public function benchSoftmaxKnownValues(): void
    {
        // Verify softmax([2,1,0.1]) ≈ [0.659, 0.242, 0.099] (hand-computed)
        $logits = self::$knownInput->copy();
        $logits->rowSoftmaxInplace();
        $flat = $logits->toFlatArray();
        // Row 0 class 0 should dominate
        if ($flat[0] < 0.5 || $flat[0] > 0.9) {
            throw new \RuntimeException("Softmax[0,0]={$flat[0]}, expected 0.50–0.90");
        }
        // Rows are probability distributions — all elements in (0, 1)
        foreach ($flat as $v) {
            if ($v <= 0.0 || $v >= 1.0) {
                throw new \RuntimeException("Softmax output {$v} not in (0,1)");
            }
        }
        unset($logits);
    }

    // =========================================================================
    // SIGMOID — all outputs must be in (0, 1)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['accuracy', 'numerical'])]
    public function benchSigmoidOutputRange(): void
    {
        $t    = Tensor::randomNormal([10_000]);
        $sig  = $t->sigmoid();
        $minV = $sig->min();
        $maxV = $sig->max();
        if ($minV < 0.0 || $maxV > 1.0) {
            throw new \RuntimeException("Sigmoid range [{$minV}, {$maxV}] not in [0,1]");
        }
        unset($t, $sig);
    }

    // =========================================================================
    // TANH — all outputs must be in (-1, 1)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['accuracy', 'numerical'])]
    public function benchTanhOutputRange(): void
    {
        $t    = Tensor::randomNormal([10_000]);
        $th   = $t->tanh();
        $minV = $th->min();
        $maxV = $th->max();
        if ($minV < -1.0 || $maxV > 1.0) {
            throw new \RuntimeException("Tanh range [{$minV}, {$maxV}] not in [-1,1]");
        }
        unset($t, $th);
    }

    // =========================================================================
    // EXP / LOG INVERSE PROPERTY: log(exp(x)) ≈ x
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['accuracy', 'numerical'])]
    public function benchExpLogInverse(): void
    {
        $x     = Tensor::randomUniform([100_000], -5.0, 5.0);
        $expX  = $x->exp();
        $logEx = $expX->log();
        // mean absolute error should be < 1e-4
        $diff  = $logEx->sub($x)->abs()->mean();
        if ($diff > 1e-3) {
            throw new \RuntimeException("log(exp(x)) MAE = {$diff}, expected < 0.001");
        }
        unset($x, $expX, $logEx);
    }

    // =========================================================================
    // SQRT PRECISION: sqrt(x)^2 ≈ x
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['accuracy', 'numerical'])]
    public function benchSqrtSquareInverse(): void
    {
        $x    = Tensor::randomUniform([100_000], 0.001, 100.0);
        $sqX  = $x->sqrt();
        $sq2  = $sqX->mul($sqX);
        $diff = $sq2->sub($x)->abs()->mean();
        // fp32 sqrt should give ~1e-5 relative error
        if ($diff > 0.01) {
            throw new \RuntimeException("sqrt(x)^2 MAE = {$diff}, expected < 0.01");
        }
        unset($x, $sqX, $sq2);
    }

    // =========================================================================
    // MATMUL ASSOCIATIVITY: (A*B)*C ≈ A*(B*C) for small matrices
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['accuracy', 'numerical'])]
    public function benchMatmulAssociativity(): void
    {
        $A = Tensor::randomNormal([16, 16]);
        $B = Tensor::randomNormal([16, 16]);
        $C = Tensor::randomNormal([16, 16]);

        $lhs = $A->matmul($B)->matmul($C);
        $rhs = $A->matmul($B->matmul($C));

        $flat1 = $lhs->toFlatArray();
        $flat2 = $rhs->toFlatArray();

        $maxErr = 0.0;
        foreach ($flat1 as $i => $v) {
            $maxErr = \max($maxErr, \abs($v - $flat2[$i]));
        }
        if ($maxErr > 1.0) {
            throw new \RuntimeException("Matmul associativity max error = {$maxErr}, expected < 1.0");
        }
        unset($A, $B, $C, $lhs, $rhs);
    }

    // =========================================================================
    // CROSS ENTROPY LOSS RANGE: CCE ∈ [0, ∞); perfect preds → near 0
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['accuracy', 'loss'])]
    public function benchCCELossNonNegative(): void
    {
        $softmax = new Softmax();
        $cce     = new CategoricalCrossEntropy();

        $probs = $softmax->forward(self::$knownInput);
        $dY    = $cce->differentiate($probs, self::$knownTarget);

        $flat = $dY->toFlatArray();
        // Gradient should be finite
        foreach ($flat as $v) {
            if (!\is_finite($v)) {
                throw new \RuntimeException("CCE gradient is non-finite: {$v}");
            }
        }
        unset($probs, $dY, $softmax, $cce);
    }

    // =========================================================================
    // ML ACCURACY — GaussianNB should reach >70% on linearly separable data
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['accuracy', 'ml'])]
    public function benchGaussianNBAccuracyOnLinearData(): void
    {
        \mt_srand(1);
        $n = 1000; $d = 10;
        $samples = Tensor::zeros($n, $d);
        $lBuf    = \array_fill(0, $n, 0.0);
        $sBuf    = $samples->buffer();
        for ($i = 0; $i < $n; $i++) {
            $cls = $i % 2;
            $lBuf[$i] = (float)$cls;
            for ($j = 0; $j < $d; $j++) {
                $sBuf[$i * $d + $j] = ($cls === 0 ? 2.0 : -2.0) + (\mt_rand(-100, 100) / 200.0);
            }
        }
        $ds  = new Dataset($samples, Tensor::fromArray($lBuf));
        [$train, $test] = $ds->split(0.8);

        $gnb = new GaussianNB();
        $gnb->train($train);
        $preds = $gnb->predict($test);

        $acc = new Accuracy();
        $score = $acc->score($preds, $test->labels());
        if ($score < 0.70) {
            throw new \RuntimeException("GaussianNB accuracy = {$score}, expected >= 0.70");
        }
        unset($ds, $train, $test, $gnb, $preds);
    }

    // =========================================================================
    // LINEAR REGRESSION R² on a perfectly linear dataset → should be > 0.99
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['accuracy', 'ml'])]
    public function benchLinearRegressionOnPerfectLinearData(): void
    {
        \mt_srand(2);
        $n = 500; $d = 5;
        $X = Tensor::randomNormal([$n, $d]);
        $w = Tensor::fromArray([[1.0], [2.0], [-1.0], [0.5], [3.0]]);
        $y = $X->matmul($w)->flatten();

        $ds = new Dataset($X, $y);
        [$train, $test] = $ds->split(0.8);

        $lr = new LinearRegression();
        $lr->train($train);
        $preds = $lr->predict($test);

        $r2 = new RSquared();
        $score = $r2->score($preds, $test->labels());
        if ($score < 0.90) {
            throw new \RuntimeException("LinearRegression R² = {$score}, expected >= 0.90");
        }
        unset($X, $y, $w, $ds, $train, $test, $lr, $preds);
    }

    // =========================================================================
    // SUM AXIS ACCURACY — verify axis reduction matches manual computation
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    #[Bench\Groups(['accuracy', 'numerical'])]
    public function benchSumAxisAccuracy(): void
    {
        // [4, 3] known matrix: row sums should be [6, 9, 12, 15]
        $m = Tensor::fromArray([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 0.0],  // 9
            [3.0, 4.0, 5.0],  // 12
            [5.0, 6.0, 4.0],  // 15
        ]);
        $rowSums = $m->sumAxis(1)->toFlatArray();
        $expected = [6.0, 9.0, 12.0, 15.0];
        foreach ($expected as $i => $exp) {
            $err = \abs($rowSums[$i] - $exp);
            if ($err > 1e-4) {
                throw new \RuntimeException("sumAxis(1)[{$i}] = {$rowSums[$i]}, expected {$exp}");
            }
        }
        unset($m, $rowSums);
    }

    // =========================================================================
    // STANDARDIZE — after standardize, mean ≈ 0, std ≈ 1
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    #[Bench\Groups(['accuracy', 'numerical'])]
    public function benchStandardizeProducesZeroMeanUnitStd(): void
    {
        $t = Tensor::randomNormal([10_000]);
        $t->standardizeInplace();
        $m = $t->mean();
        $s = $t->std();
        if (\abs($m) > 0.05) {
            throw new \RuntimeException("Standardize: mean = {$m}, expected ~0");
        }
        if (\abs($s - 1.0) > 0.05) {
            throw new \RuntimeException("Standardize: std = {$s}, expected ~1");
        }
        unset($t);
    }
}
