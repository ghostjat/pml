<?php

declare(strict_types=1);

namespace Pml\Benchmarks;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Performance profile for the Dataset API.
 * Focuses on Zero-Copy speed, FFI Memory ingestion, and JIT Loop Optimizations.
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Warmup(2)]
#[Bench\Revs(5)]
#[Bench\Iterations(3)]
final class DatasetBench
{
    private Dataset $massiveDataset;

    public function setUp(): void
    {
        // 50,000 Rows, 100 Feature Columns (5,000,000 Total Elements)
        // This size realistically represents a substantial tabular training set
        $samples = Tensor::randomNormal([50000, 100], 0.0, 1.0);
        $labels = Tensor::randomUniform([50000], 0.0, 1.0)->round();
        
        $this->massiveDataset = new Dataset($samples, $labels);
    }

    #[Bench\Groups(['dataset', 'zerocopy'])]
    #[Bench\Assert('mode(variant.time.avg) < 1ms')] // Split should be instant (just pointer math)
    public function benchDatasetSplit(): void
    {
        $this->massiveDataset->split(0.8);
    }

    #[Bench\Groups(['dataset', 'zerocopy', 'loops'])]
    #[Bench\Assert('mode(variant.time.avg) < 5ms')]
    public function benchDatasetBatchesIteration(): void
    {
        // Iterating 1,562 mini-batches of 32 rows
        // Evaluates the JIT caching of FFI dimensions and Zero-Copy Slice execution
        foreach ($this->massiveDataset->batches(32) as $batch) {
            // Simulating a fast-loop forward pass
            $rows = $batch->numRows(); 
        }
    }

    #[Bench\Groups(['dataset', 'zerocopy'])]
    public function benchDatasetFold(): void
    {
        // Generating 10 cross-validation folds and performing C-level Concatenation
        foreach ($this->massiveDataset->fold(10) as [$train, $val]) {
            // Simulated validation cycle
        }
    }

    #[Bench\Groups(['dataset', 'mutation'])]
    public function benchDatasetStandardize(): void
    {
        // In-Place Z-Score Normalization of 5,000,000 floats via C/OpenMP
        $this->massiveDataset->standardize();
    }

    #[Bench\Groups(['dataset', 'mutation'])]
    public function benchDatasetRandomize(): void
    {
        // Tests the speed of C-level argsort combined with memory swapping (fancy indexing)
        $this->massiveDataset->randomize();
    }

    #[Bench\Groups(['dataset', 'memory'])]
    public function benchDatasetSelectDropColumns(): void
    {
        // Dynamically drops 50 columns out of 100, forcing a massive memory subset copy
        $dropIndices = range(0, 49);
        $this->massiveDataset->drop($dropIndices);
    }
    
    #[Bench\Groups(['dataset', 'export'])]
    public function benchDatasetToArray(): void
    {
        // Measures the physical bottleneck of serializing 5 Million C-floats into a nested PHP Array
        $this->massiveDataset->toArray();
    }
}