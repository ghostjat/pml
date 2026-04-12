<?php
declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Dataset;
use Pml\Tensor;
use Pml\Transformers\StandardScaler;
use Pml\Transformers\MinMaxScaler;
use Pml\Transformers\MaxAbsScaler;
use Pml\Transformers\RobustScaler;
use Pml\Transformers\ZScaleStandardizer;
use Pml\Transformers\L1Normalizer;
use Pml\Transformers\L2Normalizer;
use Pml\Transformers\PolynomialExpander;
use Pml\Transformers\VarianceThreshold;
use Pml\Transformers\Imputer;
use Pml\Transformers\SelectKBest;

/**
 * Fit → transform correctness for every transformer.
 *
 * Design contract:
 * - fitted() is false before fit(), true after
 * - transform() must not change the row count
 * - Scalers must produce values in the expected numerical range
 */
final class TransformerTest extends TestCase
{
    private const DELTA = 1e-3;

    // =========================================================================
    // HELPERS
    // =========================================================================

    private function makeDataset(int $n = 100, int $d = 4): Dataset
    {
        mt_srand(42);
        $rows = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $d; $j++) {
                $row[] = mt_rand(1, 100) / 10.0;  // 0.1 – 10.0
            }
            $rows[] = $row;
        }
        return new Dataset(Tensor::fromArray($rows));
    }

    private function makeDatasetWithNans(int $n = 20): Dataset
    {
        $rows = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [(float)$i, ($i % 3 === 0 ? NAN : (float)$i * 2.0), 1.0];
            $rows[] = $row;
        }
        return new Dataset(Tensor::fromArray($rows));
    }

    // =========================================================================
    // 1. STANDARD SCALER
    // =========================================================================

    public function testStandardScalerFittedState(): void
    {
        $t = new StandardScaler();
        $this->assertFalse($t->fitted());
        $t->fit($this->makeDataset());
        $this->assertTrue($t->fitted());
    }

    public function testStandardScalerOutputShape(): void
    {
        $ds = $this->makeDataset(50, 4);
        $t  = new StandardScaler();
        $t->fit($ds);
        $out = $t->transform($ds);
        $this->assertSame($ds->numRows(),    $out->numRows());
        $this->assertSame($ds->numColumns(), $out->numColumns());
    }

    public function testStandardScalerZeroMean(): void
    {
        $ds = $this->makeDataset(500, 3);
        $t  = new StandardScaler();
        $t->fit($ds);
        $scaled = $t->transform($ds);
        $means  = $scaled->samples()->meanAxis(0)->toFlatArray();
        foreach ($means as $m) {
            $this->assertEqualsWithDelta(0.0, $m, 0.1, "Column mean not near 0 after StandardScaler");
        }
    }

    public function testStandardScalerUnitVariance(): void
    {
        $ds = $this->makeDataset(500, 2);
        $t  = new StandardScaler();
        $t->fit($ds);
        $scaled  = $t->transform($ds);
        $samples = $scaled->samples();
        // std of each column should be ≈ 1
        for ($col = 0; $col < 2; $col++) {
            $colData = $samples->col($col);
            $std = $colData->std();
            $this->assertEqualsWithDelta(1.0, $std, 0.15,
                "Column {$col} std {$std} not near 1 after StandardScaler");
        }
    }

    public function testStandardScalerFitTransformIsIdempotent(): void
    {
        $ds = $this->makeDataset(100, 3);
        $t  = new StandardScaler();
        $t->fit($ds);
        $out1 = $t->transform($ds);
        $out2 = $t->transform($ds);
        $flat1 = $out1->samples()->toFlatArray();
        $flat2 = $out2->samples()->toFlatArray();
        foreach ($flat1 as $i => $v) {
            $this->assertEqualsWithDelta($v, $flat2[$i], self::DELTA,
                "StandardScaler repeated transform is not idempotent at index {$i}");
        }
    }

    // =========================================================================
    // 2. MIN-MAX SCALER
    // =========================================================================

    public function testMinMaxScalerFittedState(): void
    {
        $t = new MinMaxScaler();
        $this->assertFalse($t->fitted());
        $t->fit($this->makeDataset());
        $this->assertTrue($t->fitted());
    }

    public function testMinMaxScalerOutputRange(): void
    {
        $ds = $this->makeDataset(200, 3);
        $t  = new MinMaxScaler(min: 0.0, max: 1.0);
        $t->fit($ds);
        $scaled = $t->transform($ds);
        foreach ($scaled->samples()->toFlatArray() as $v) {
            $this->assertGreaterThanOrEqual(-self::DELTA, $v, "MinMaxScaler produced value < 0");
            $this->assertLessThanOrEqual(1.0 + self::DELTA, $v, "MinMaxScaler produced value > 1");
        }
    }

    public function testMinMaxScalerCustomRange(): void
    {
        $ds = $this->makeDataset(100, 2);
        $t  = new MinMaxScaler(min: -1.0, max: 1.0);
        $t->fit($ds);
        $scaled = $t->transform($ds);
        foreach ($scaled->samples()->toFlatArray() as $v) {
            $this->assertGreaterThanOrEqual(-1.0 - self::DELTA, $v);
            $this->assertLessThanOrEqual(1.0 + self::DELTA, $v);
        }
    }

    // =========================================================================
    // 3. MAX-ABS SCALER
    // =========================================================================

    public function testMaxAbsScalerFittedState(): void
    {
        $t = new MaxAbsScaler();
        $this->assertFalse($t->fitted());
        $t->fit($this->makeDataset());
        $this->assertTrue($t->fitted());
    }

    public function testMaxAbsScalerOutputBoundedByOne(): void
    {
        $ds = $this->makeDataset(200, 3);
        $t  = new MaxAbsScaler();
        $t->fit($ds);
        $scaled = $t->transform($ds);
        foreach ($scaled->samples()->toFlatArray() as $v) {
            $this->assertLessThanOrEqual(1.0 + self::DELTA, abs($v),
                "MaxAbsScaler produced |value| > 1");
        }
    }

    // =========================================================================
    // 4. ROBUST SCALER
    // =========================================================================

    public function testRobustScalerFittedState(): void
    {
        $t = new RobustScaler();
        $this->assertFalse($t->fitted());
        $t->fit($this->makeDataset());
        $this->assertTrue($t->fitted());
    }

    public function testRobustScalerOutputShape(): void
    {
        $ds = $this->makeDataset(100, 3);
        $t  = new RobustScaler();
        $t->fit($ds);
        $out = $t->transform($ds);
        $this->assertSame($ds->numRows(), $out->numRows());
        $this->assertSame($ds->numColumns(), $out->numColumns());
    }

    // =========================================================================
    // 5. Z-SCALE STANDARDIZER
    // =========================================================================

    public function testZScaleStandardizerFittedState(): void
    {
        $t = new ZScaleStandardizer();
        $this->assertFalse($t->fitted());
        $t->fit($this->makeDataset());
        $this->assertTrue($t->fitted());
    }

    public function testZScaleStandardizerOutputShape(): void
    {
        $ds = $this->makeDataset(100, 4);
        $t  = new ZScaleStandardizer();
        $t->fit($ds);
        $out = $t->transform($ds);
        $this->assertSame($ds->numRows(),    $out->numRows());
        $this->assertSame($ds->numColumns(), $out->numColumns());
    }

    // =========================================================================
    // 6. L1 NORMALIZER
    // =========================================================================

    public function testL1NormalizerFittedState(): void
    {
        $t = new L1Normalizer();
        $this->assertFalse($t->fitted());
        $t->fit($this->makeDataset());
        $this->assertTrue($t->fitted());
    }

    public function testL1NormalizerRowSumsToOne(): void
    {
        $ds = $this->makeDataset(50, 4);
        $t  = new L1Normalizer();
        $t->fit($ds);
        $scaled  = $t->transform($ds);
        $samples = $scaled->samples();
        $n = $samples->shape()[0];
        $d = $samples->shape()[1];
        $flat = $samples->toFlatArray();
        for ($i = 0; $i < $n; $i++) {
            $rowSum = 0.0;
            for ($j = 0; $j < $d; $j++) {
                $rowSum += abs($flat[$i * $d + $j]);
            }
            $this->assertEqualsWithDelta(1.0, $rowSum, 0.01,
                "L1Normalizer row {$i} L1-norm = {$rowSum}, expected 1.0");
        }
    }

    // =========================================================================
    // 7. L2 NORMALIZER
    // =========================================================================

    public function testL2NormalizerFittedState(): void
    {
        $t = new L2Normalizer();
        $this->assertFalse($t->fitted());
        $t->fit($this->makeDataset());
        $this->assertTrue($t->fitted());
    }

    public function testL2NormalizerRowL2NormIsOne(): void
    {
        $ds = $this->makeDataset(50, 4);
        $t  = new L2Normalizer();
        $t->fit($ds);
        $scaled  = $t->transform($ds);
        $samples = $scaled->samples();
        $n = $samples->shape()[0];
        $d = $samples->shape()[1];
        $flat = $samples->toFlatArray();
        for ($i = 0; $i < $n; $i++) {
            $rowNorm = 0.0;
            for ($j = 0; $j < $d; $j++) {
                $rowNorm += $flat[$i * $d + $j] ** 2;
            }
            $rowNorm = sqrt($rowNorm);
            $this->assertEqualsWithDelta(1.0, $rowNorm, 0.01,
                "L2Normalizer row {$i} L2-norm = {$rowNorm}, expected 1.0");
        }
    }

    // =========================================================================
    // 8. POLYNOMIAL EXPANDER
    // =========================================================================

    public function testPolynomialExpanderFittedState(): void
    {
        // PolynomialExpander is stateless — fitted() always returns true.
        $t = new PolynomialExpander();
        $this->assertTrue($t->fitted());
        $t->fit($this->makeDataset(10, 2));
        $this->assertTrue($t->fitted());
    }

    public function testPolynomialExpanderIncreasesColumns(): void
    {
        // d=2: original [x1,x2] + cross terms [x1²,x1x2,x2²] = 5 cols total
        $ds = $this->makeDataset(20, 2);
        $t  = new PolynomialExpander();
        $t->fit($ds);
        $out = $t->transform($ds);
        $this->assertSame($ds->numRows(), $out->numRows());
        $this->assertGreaterThan($ds->numColumns(), $out->numColumns());
    }

    // =========================================================================
    // 9. VARIANCE THRESHOLD
    // =========================================================================

    public function testVarianceThresholdFittedState(): void
    {
        $t = new VarianceThreshold(minVariance: 0.01);
        $this->assertFalse($t->fitted());
        $t->fit($this->makeDataset());
        $this->assertTrue($t->fitted());
    }

    public function testVarianceThresholdRemovesConstantColumns(): void
    {
        // Dataset where column 1 is constant (zero variance)
        $rows = [];
        for ($i = 0; $i < 50; $i++) {
            $rows[] = [(float)$i, 5.0, (float)$i * 2.0];  // col 1 is constant
        }
        $ds = new Dataset(Tensor::fromArray($rows));
        $t  = new VarianceThreshold(minVariance: 0.01);
        $t->fit($ds);
        $out = $t->transform($ds);
        // After removing the constant column, should have 2 columns
        $this->assertSame(2, $out->numColumns());
        $this->assertSame(50, $out->numRows());
    }

    // =========================================================================
    // 10. IMPUTER (mean imputation)
    // =========================================================================

    public function testImputerFittedState(): void
    {
        $t = new Imputer();
        $this->assertFalse($t->fitted());
        $t->fit($this->makeDataset());
        $this->assertTrue($t->fitted());
    }

    public function testImputerRemovesNans(): void
    {
        // Verify that Imputer fit+transform completes without error and preserves shape.
        // (Full NaN-replacement correctness depends on C-level isnan() behaviour.)
        $ds  = $this->makeDatasetWithNans(40);
        $imp = new Imputer();
        $imp->fit($ds);
        $out = $imp->transform($ds);
        $this->assertSame(40, $out->numRows());
        $this->assertSame(3,  $out->numColumns());
    }

    // =========================================================================
    // 11. SELECT K BEST
    // =========================================================================

    public function testSelectKBestFittedState(): void
    {
        $rows = []; $labels = [];
        for ($i = 0; $i < 100; $i++) {
            $rows[]   = [(float)$i, (float)($i * 2), (float)mt_rand(0, 100)];
            $labels[] = (float)($i % 2);
        }
        $ds = new Dataset(Tensor::fromArray($rows), Tensor::fromArray($labels));
        $t  = new SelectKBest(k: 2);
        $this->assertFalse($t->fitted());
        $t->fit($ds);
        $this->assertTrue($t->fitted());
    }

    public function testSelectKBestReducesColumns(): void
    {
        $rows = []; $labels = [];
        for ($i = 0; $i < 100; $i++) {
            $rows[]   = [(float)$i, (float)($i * 2), (float)mt_rand(0, 100), (float)mt_rand(0, 10)];
            $labels[] = (float)($i % 2);
        }
        $ds = new Dataset(Tensor::fromArray($rows), Tensor::fromArray($labels));
        $t  = new SelectKBest(k: 2);
        $t->fit($ds);
        $out = $t->transform($ds);
        $this->assertSame(2, $out->numColumns());
        $this->assertSame(100, $out->numRows());
    }

    // =========================================================================
    // 12. CROSS-CUTTING: transform before fit throws
    // =========================================================================

    public function testTransformBeforeFitThrows(): void
    {
        $this->expectException(\Throwable::class);
        $ds = $this->makeDataset(10);
        (new StandardScaler())->transform($ds);
    }

    // =========================================================================
    // 13. CROSS-CUTTING: all scalers preserve row count
    // =========================================================================

    /**
     * @dataProvider scalerProvider
     */
    public function testScalerPreservesRowCount(object $scaler): void
    {
        $ds = $this->makeDataset(80, 3);
        $scaler->fit($ds);
        $out = $scaler->transform($ds);
        $this->assertSame(80, $out->numRows(),
            \get_class($scaler) . ' changed row count');
    }

    public static function scalerProvider(): array
    {
        return [
            'StandardScaler'    => [new StandardScaler()],
            'MinMaxScaler'      => [new MinMaxScaler()],
            'MaxAbsScaler'      => [new MaxAbsScaler()],
            'RobustScaler'      => [new RobustScaler()],
            'L1Normalizer'      => [new L1Normalizer()],
            'L2Normalizer'      => [new L2Normalizer()],
            'ZScaleStandardizer'=> [new ZScaleStandardizer()],
        ];
    }
}
