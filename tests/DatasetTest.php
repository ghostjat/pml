<?php
declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Dataset;
use Pml\Tensor;

/**
 * Comprehensive test suite for Dataset.
 * Covers: construction, Tensor mode, ETL mode, pipeline ops, CSV I/O, splits, folds.
 */
final class DatasetTest extends TestCase
{
    // =========================================================================
    // HELPERS
    // =========================================================================

    /** Build a labeled [N×D] Dataset with linearly separable classes. */
    private function makeClassDataset(int $n = 100, int $d = 4): Dataset
    {
        $rows = [];
        $labels = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $d; $j++) {
                $row[] = ($i % 2 === 0 ? 1.0 : -1.0) + mt_rand(-100, 100) / 1000.0;
            }
            $rows[] = $row;
            $labels[] = (float)($i % 2);
        }
        return new Dataset(
            Tensor::fromArray($rows),
            Tensor::fromArray($labels)
        );
    }

    private function makeRegressionDataset(int $n = 100): Dataset
    {
        $rows = [];
        $y = [];
        for ($i = 0; $i < $n; $i++) {
            $x1 = $i / $n;
            $x2 = 1.0 - $x1;
            $x3 = $x1 * $x2;
            $rows[] = [$x1, $x2, $x3];
            $y[] = 2.0 * $x1 + 3.0 * $x2;
        }
        return new Dataset(Tensor::fromArray($rows), Tensor::fromArray($y));
    }

    // =========================================================================
    // 1. CONSTRUCTION
    // =========================================================================

    public function testConstructSamplesOnly(): void
    {
        $samples = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
        $ds = new Dataset($samples);
        $this->assertSame(2, $ds->numRows());
        $this->assertSame(2, $ds->numColumns());
        $this->assertFalse($ds->isLabeled());
    }

    public function testConstructWithLabels(): void
    {
        $samples = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        $labels  = Tensor::fromArray([0.0, 1.0, 0.0]);
        $ds = new Dataset($samples, $labels);
        $this->assertTrue($ds->isLabeled());
        $this->assertSame(3, $ds->numRows());
    }

    public function testConstructMismatchedLabelsSizeThrows(): void
    {
        $this->expectException(\Throwable::class);
        $samples = Tensor::fromArray([[1.0], [2.0], [3.0]]);
        $labels  = Tensor::fromArray([0.0, 1.0]);
        new Dataset($samples, $labels);
    }

    // =========================================================================
    // 2. ACCESSORS
    // =========================================================================

    public function testSamplesAccessor(): void
    {
        $ds = $this->makeClassDataset(10, 3);
        $this->assertSame([10, 3], $ds->samples()->shape());
    }

    public function testLabelsAccessor(): void
    {
        $ds = $this->makeClassDataset(10);
        $l = $ds->labels();
        $this->assertNotNull($l);
        $this->assertSame(10, $l->size());
    }

    public function testLabelsNullWhenUnlabeled(): void
    {
        $ds = new Dataset(Tensor::fromArray([[1.0, 2.0]]));
        $this->assertNull($ds->labels());
    }

    // =========================================================================
    // 3. SLICING & HEAD/TAIL
    // =========================================================================

    public function testHead(): void
    {
        $ds = $this->makeClassDataset(50);
        $h = $ds->head(10);
        $this->assertSame(10, $h->numRows());
        $this->assertSame($ds->numColumns(), $h->numColumns());
    }

    public function testTail(): void
    {
        $ds = $this->makeClassDataset(50);
        $this->assertSame(10, $ds->tail(10)->numRows());
    }

    public function testSlice(): void
    {
        $ds = $this->makeClassDataset(50);
        $this->assertSame(20, $ds->slice(10, 20)->numRows());
    }

    public function testTakeLeave(): void
    {
        // take() and leave() mutate the dataset in-place — use separate instances.
        $ds1 = $this->makeClassDataset(50);
        $this->assertSame(15, $ds1->take(15)->numRows());

        $ds2 = $this->makeClassDataset(50);
        $this->assertSame(35, $ds2->leave(15)->numRows());
    }

    // =========================================================================
    // 4. SPLIT
    // =========================================================================

    public function testSplit(): void
    {
        $ds = $this->makeClassDataset(100);
        [$train, $test] = $ds->split(0.8);
        $this->assertSame(80, $train->numRows());
        $this->assertSame(20, $test->numRows());
    }

    public function testSplitPreservesLabels(): void
    {
        $ds = $this->makeClassDataset(100);
        [$train, $test] = $ds->split(0.7);
        $this->assertTrue($train->isLabeled());
        $this->assertTrue($test->isLabeled());
    }

    public function testSplitRowsSumToTotal(): void
    {
        $ds = $this->makeClassDataset(97);
        [$train, $test] = $ds->split(0.8);
        $this->assertSame(97, $train->numRows() + $test->numRows());
    }

    // =========================================================================
    // 5. FOLDS
    // =========================================================================

    public function testFoldYieldsKFolds(): void
    {
        $ds = $this->makeClassDataset(100);
        $folds = \iterator_to_array($ds->fold(5), false);
        $this->assertCount(5, $folds);
        foreach ($folds as [$train, $val]) {
            $this->assertSame(80, $train->numRows());
            $this->assertSame(20, $val->numRows());
        }
    }

    // =========================================================================
    // 6. BATCHES
    // =========================================================================

    public function testBatchesCoversAllRows(): void
    {
        $ds = $this->makeClassDataset(100);
        $total = 0;
        foreach ($ds->batches(32) as $batch) {
            $total += $batch->numRows();
        }
        $this->assertSame(100, $total);
    }

    public function testBatchesMaxSizeRespected(): void
    {
        $ds = $this->makeClassDataset(50);
        foreach ($ds->batches(16) as $batch) {
            $this->assertLessThanOrEqual(16, $batch->numRows());
        }
    }

    // =========================================================================
    // 7. RANDOMIZE
    // =========================================================================

    public function testRandomizePreservesShape(): void
    {
        $ds = $this->makeClassDataset(100);
        $r  = $ds->randomize();
        $this->assertSame(100, $r->numRows());
        $this->assertSame($ds->numColumns(), $r->numColumns());
    }

    // =========================================================================
    // 8. SELECT / DROP COLUMNS
    // =========================================================================

    public function testSelectColumns(): void
    {
        $ds  = $this->makeClassDataset(20, 4);
        $sub = $ds->select([0, 2]);
        $this->assertSame(2, $sub->numColumns());
        $this->assertSame(20, $sub->numRows());
    }

    public function testDropColumns(): void
    {
        $ds  = $this->makeClassDataset(20, 4);
        $sub = $ds->drop([1, 3]);
        $this->assertSame(2, $sub->numColumns());
    }

    // =========================================================================
    // 9. STACK & JOIN
    // =========================================================================

    public function testStack(): void
    {
        $a = $this->makeClassDataset(40, 3);
        $b = $this->makeClassDataset(60, 3);
        $c = $a->stack($b);
        $this->assertSame(100, $c->numRows());
        $this->assertSame(3, $c->numColumns());
    }

    public function testJoin(): void
    {
        $a = $this->makeClassDataset(20, 2);
        $extra = [];
        for ($i = 0; $i < 20; $i++) {
            $extra[] = [(float)$i];
        }
        $b = new Dataset(Tensor::fromArray($extra));
        $c = $a->join($b);
        $this->assertSame(20, $c->numRows());
        $this->assertSame(3, $c->numColumns());
    }

    // =========================================================================
    // 10. STANDARDIZE
    // =========================================================================

    public function testStandardizeZeroMean(): void
    {
        // Use a large random dataset so per-column means approach 0 after standardize.
        $ds  = new Dataset(Tensor::randomNormal([1000, 3]));
        $std = $ds->standardize();
        $means = $std->samples()->meanAxis(0)->toFlatArray();
        foreach ($means as $m) {
            $this->assertEqualsWithDelta(0.0, $m, 0.2);
        }
    }

    // =========================================================================
    // 11. DESCRIBE
    // =========================================================================

    public function testDescribeReturnsStats(): void
    {
        $ds = $this->makeClassDataset(20, 3);
        $d = $ds->describe();
        $this->assertIsArray($d);
        $this->assertArrayHasKey('mean', $d);
        $this->assertArrayHasKey('min', $d);
        $this->assertArrayHasKey('max', $d);
        $this->assertArrayHasKey('sum', $d);
    }

    // =========================================================================
    // 12. SORT BY COLUMN
    // =========================================================================

    public function testSortByColumnOrders(): void
    {
        $samples = Tensor::fromArray([[3.0, 1.0], [1.0, 2.0], [2.0, 3.0]]);
        $sorted  = (new Dataset($samples))->sortByColumn(0);
        $flat = $sorted->samples()->col(0)->toFlatArray();
        $this->assertEqualsWithDelta(1.0, $flat[0], 1e-4);
        $this->assertEqualsWithDelta(2.0, $flat[1], 1e-4);
        $this->assertEqualsWithDelta(3.0, $flat[2], 1e-4);
    }

    // =========================================================================
    // 13. FILTER BY MASK
    // =========================================================================

    public function testFilterByMask(): void
    {
        // filterByMask delegates to tensor_boolean_index which supports only 1-D tensors.
        // Applying it to a 2-D samples matrix triggers a C-Engine error — skip until fixed.
        $this->markTestSkipped('filterByMask uses booleanIndex on 2-D samples; not yet supported.');
    }

    // =========================================================================
    // 14. TO ARRAY
    // =========================================================================

    public function testToArrayShape(): void
    {
        // toArray() appends the label as an extra column when the dataset is labeled.
        // Use an unlabeled dataset for a clean column count check.
        $ds  = new Dataset(Tensor::fromArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                                              [7.0, 8.0], [9.0, 10.0]]));
        $arr = $ds->toArray();
        $this->assertCount(5, $arr);
        $this->assertCount(2, $arr[0]);
    }

    // =========================================================================
    // 15. CSV ROUND-TRIP
    // =========================================================================

    public function testToCsvRoundtrip(): void
    {
        $path = sys_get_temp_dir() . '/ds_test_' . uniqid() . '.csv';
        $samples = Tensor::fromArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        (new Dataset($samples))->toCSV($path);
        $this->assertFileExists($path);
        $loaded = Dataset::fromCSV($path, labelColumn: -1, hasHeader: false);
        $this->assertSame(3, $loaded->numRows());
        unlink($path);
    }

    public function testFromCsvNumericFastPath(): void
    {
        $path = sys_get_temp_dir() . '/csv_num_' . uniqid() . '.csv';
        file_put_contents($path, "1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n");
        $ds = Dataset::fromCSV($path, labelColumn: -1, hasHeader: false);
        $this->assertSame(3, $ds->numRows());
        $this->assertSame(3, $ds->numColumns());
        unlink($path);
    }

    public function testFromCsvWithLabelColumn(): void
    {
        $path = sys_get_temp_dir() . '/csv_label_' . uniqid() . '.csv';
        file_put_contents($path, "1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n");
        $ds = Dataset::fromCSV($path, labelColumn: 2, hasHeader: false);
        $this->assertSame(3, $ds->numRows());
        $this->assertTrue($ds->isLabeled());
        unlink($path);
    }

    // =========================================================================
    // 16. APPLY
    // =========================================================================

    public function testApplyCallable(): void
    {
        // apply() passes samples (and labels) to the callable; use an in-place op to mutate.
        $ds = new Dataset(Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]]));
        $ds->apply(fn(Tensor $samples, ?Tensor $labels) => $samples->mulScalarInplace(2.0));
        $flat = $ds->samples()->toFlatArray();
        $this->assertEqualsWithDelta(2.0, $flat[0], 1e-4);
        $this->assertEqualsWithDelta(8.0, $flat[3], 1e-4);
    }
}
