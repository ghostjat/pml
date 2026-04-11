<?php

declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Tensor;
use Pml\Dataset;

final class DatasetTest extends TestCase
{
    private array $sampleData = [
        [1.0, 0.5, 0.2],
        [2.0, 1.5, 0.4],
        [3.0, 2.5, 0.6],
        [4.0, 3.5, 0.8],
        [5.0, 4.5, 1.0],
    ];

    private array $labelData = [0, 1, 0, 1, 0];

    public function testDatasetCreationAndProperties(): void
    {
        $dataset = Dataset::fromArray($this->sampleData, $this->labelData);

        $this->assertSame(5, $dataset->numRows());
        $this->assertSame(3, $dataset->numColumns());
        $this->assertTrue($dataset->isLabeled());
        
        $this->assertSame([5, 3], $dataset->samples()->shape());
        $this->assertSame([5], $dataset->labels()->shape());
    }

    public function testSelectAndDropColumns(): void
    {
        $dataset = Dataset::fromArray($this->sampleData);

        $selected = $dataset->select([0, 2]);
        $this->assertSame(2, $selected->numColumns());
        $this->assertEqualsWithDelta(0.2, $selected->samples()->toFlatArray()[1], 0.0001);

        $dropped = $dataset->drop([1]);
        $this->assertSame(2, $dropped->numColumns());
        
        $this->assertEquals($selected->toArray(), $dropped->toArray());
    }

    public function testZeroCopyHeadTailAndSlice(): void
    {
        $dataset = Dataset::fromArray($this->sampleData, $this->labelData);

        $head = $dataset->head(2);
        $this->assertSame(2, $head->numRows());
        $this->assertEquals(1.0, $head->samples()->toFlatArray()[0]);

        $tail = $dataset->tail(2);
        $this->assertSame(2, $tail->numRows());
        $this->assertEquals(4.0, $tail->samples()->toFlatArray()[0]); 
        
        $slice = $dataset->slice(1, 3);
        $this->assertSame(3, $slice->numRows());
        $this->assertEquals(2.0, $slice->samples()->toFlatArray()[0]);
    }

    public function testSplitAndFold(): void
    {
        $dataset = Dataset::fromArray($this->sampleData, $this->labelData);

        [$train, $test] = $dataset->split(0.8);
        $this->assertSame(4, $train->numRows());
        $this->assertSame(1, $test->numRows());

        $foldsCount = 0;
        foreach ($dataset->fold(5) as [$trainFold, $valFold]) {
            $this->assertSame(4, $trainFold->numRows());
            $this->assertSame(1, $valFold->numRows());
            $foldsCount++;
        }
        $this->assertSame(5, $foldsCount);
    }

    public function testBatchesGenerator(): void
    {
        $dataset = Dataset::fromArray($this->sampleData, $this->labelData);
        
        $batches = [];
        foreach ($dataset->batches(2) as $batch) {
            $batches[] = $batch;
        }

        $this->assertCount(3, $batches);
        $this->assertSame(2, $batches[0]->numRows());
        $this->assertSame(2, $batches[1]->numRows());
        $this->assertSame(1, $batches[2]->numRows());
    }
}