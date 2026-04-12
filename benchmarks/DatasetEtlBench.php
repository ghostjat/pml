<?php
declare(strict_types=1);

namespace Pml\Benchmarks;

use PhpBench\Attributes as Bench;
use Pml\Dataset;
use Pml\Tensor;
use Pml\Transformers\StandardScaler;
use Pml\Transformers\MinMaxScaler;

/**
 * Dataset construction and ETL pipeline benchmarks.
 *
 * Measures: CSV ingest speed, tensor construction, split/fold throughput,
 * standardization, and pipeline chaining overhead.
 *
 * Run with:
 *   vendor/bin/phpbench run benchmarks/DatasetEtlBench.php --report=aggregate
 */
#[Bench\Groups(['dataset', 'etl', 'pipeline', 'io'])]
final class DatasetEtlBench
{
    // Paths to pre-generated CSV fixtures (created in constructor)
    private string $csv1k;
    private string $csv10k;
    private string $csv100k;

    // Pre-built in-memory datasets
    private Dataset $ds1k;
    private Dataset $ds10k;
    private Dataset $ds100k;

    // Pre-fitted scalers
    private StandardScaler $stdScaler;
    private MinMaxScaler   $mmScaler;

    public function __construct()
    {
        // Write CSV fixtures
        $this->csv1k   = $this->writeCsv(1_000,   10);
        $this->csv10k  = $this->writeCsv(10_000,  10);
        $this->csv100k = $this->writeCsv(100_000, 10);

        // Pre-build datasets
        $this->ds1k   = $this->makeDataset(1_000,   10);
        $this->ds10k  = $this->makeDataset(10_000,  10);
        $this->ds100k = $this->makeDataset(100_000, 10);

        // Pre-fit scalers on ds10k
        $this->stdScaler = new StandardScaler();
        $this->stdScaler->fit($this->ds10k);

        $this->mmScaler = new MinMaxScaler();
        $this->mmScaler->fit($this->ds10k);
    }

    public function __destruct()
    {
        foreach ([$this->csv1k, $this->csv10k, $this->csv100k] as $path) {
            if (\file_exists($path)) {
                @\unlink($path);
            }
        }
    }

    // =========================================================================
    // HELPERS
    // =========================================================================

    private function writeCsv(int $n, int $d): string
    {
        $path = \sys_get_temp_dir() . "/bench_ds_{$n}x{$d}_" . \getmypid() . ".csv";
        $fh   = \fopen($path, 'w');
        \mt_srand(42);
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $d; $j++) {
                $row[] = \number_format(\mt_rand(-1000, 1000) / 100.0, 4, '.', '');
            }
            $row[] = (string)($i % 2);
            \fputcsv($fh, $row);
        }
        \fclose($fh);
        return $path;
    }

    private function makeDataset(int $n, int $d): Dataset
    {
        \mt_srand(42);
        $rows = []; $labels = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $d; $j++) {
                $row[] = \mt_rand(-100, 100) / 10.0;
            }
            $rows[]   = $row;
            $labels[] = (float)($i % 2);
        }
        return new Dataset(Tensor::fromArray($rows), Tensor::fromArray($labels));
    }

    // =========================================================================
    // CSV INGEST (numeric fast path — tensor_dataset_from_csv)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(5)]
    public function benchFromCsv1kRows(): void
    {
        $ds = Dataset::fromCSV($this->csv1k, labelColumn: 10, hasHeader: false);
        unset($ds);
    }

    #[Bench\Iterations(5), Bench\Revs(3)]
    public function benchFromCsv10kRows(): void
    {
        $ds = Dataset::fromCSV($this->csv10k, labelColumn: 10, hasHeader: false);
        unset($ds);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchFromCsv100kRows(): void
    {
        $ds = Dataset::fromCSV($this->csv100k, labelColumn: 10, hasHeader: false);
        unset($ds);
    }

    // =========================================================================
    // IN-MEMORY CONSTRUCTION
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchFromArray1kx10(): void
    {
        $ds = $this->makeDataset(1_000, 10);
        unset($ds);
    }

    #[Bench\Iterations(5), Bench\Revs(5)]
    public function benchFromArray10kx10(): void
    {
        $ds = $this->makeDataset(10_000, 10);
        unset($ds);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchFromArray100kx10(): void
    {
        $ds = $this->makeDataset(100_000, 10);
        unset($ds);
    }

    // =========================================================================
    // SPLIT & FOLD
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchSplit10k(): void
    {
        [$train, $test] = $this->ds10k->split(0.8);
        unset($train, $test);
    }

    #[Bench\Iterations(5), Bench\Revs(5)]
    public function benchSplit100k(): void
    {
        [$train, $test] = $this->ds100k->split(0.8);
        unset($train, $test);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchFold10x10k(): void
    {
        foreach ($this->ds10k->fold(10) as [$train, $val]) {
            unset($train, $val);
        }
    }

    // =========================================================================
    // RANDOMIZE
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchRandomize10k(): void
    {
        $r = $this->ds10k->randomize();
        unset($r);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchRandomize100k(): void
    {
        $r = $this->ds100k->randomize();
        unset($r);
    }

    // =========================================================================
    // BATCH ITERATION
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchBatches32over10k(): void
    {
        foreach ($this->ds10k->batches(32) as $b) {
            unset($b);
        }
    }

    #[Bench\Iterations(5), Bench\Revs(5)]
    public function benchBatches256over10k(): void
    {
        foreach ($this->ds10k->batches(256) as $b) {
            unset($b);
        }
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    public function benchBatches1024over100k(): void
    {
        foreach ($this->ds100k->batches(1024) as $b) {
            unset($b);
        }
    }

    // =========================================================================
    // SCALER TRANSFORM THROUGHPUT
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchStandardScalerTransform10k(): void
    {
        $out = $this->stdScaler->transform($this->ds10k);
        unset($out);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchStandardScalerTransform100k(): void
    {
        $out = $this->stdScaler->transform($this->ds100k);
        unset($out);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchMinMaxScalerTransform10k(): void
    {
        $out = $this->mmScaler->transform($this->ds10k);
        unset($out);
    }

    // =========================================================================
    // COLUMN SELECT / DROP
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchSelectColumns10k(): void
    {
        $out = $this->ds10k->select([0, 1, 2, 3, 4]);
        unset($out);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchDropColumns10k(): void
    {
        $out = $this->ds10k->drop([5, 6, 7, 8, 9]);
        unset($out);
    }

    // =========================================================================
    // STACK (vertical concatenation)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchStack10k(): void
    {
        $stacked = $this->ds10k->stack($this->ds10k);
        unset($stacked);
    }

    // =========================================================================
    // STANDARDIZE (Dataset-level — computes column stats inline)
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    public function benchDatasetStandardize10k(): void
    {
        $out = $this->ds10k->standardize();
        unset($out);
    }

    // =========================================================================
    // HEAD / TAIL / SLICE
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(50)]
    public function benchHead100kTo100(): void
    {
        $h = $this->ds100k->head(100);
        unset($h);
    }

    #[Bench\Iterations(5), Bench\Revs(20)]
    public function benchSlice100kMidSection(): void
    {
        $s = $this->ds100k->slice(25000, 50000);
        unset($s);
    }
}
