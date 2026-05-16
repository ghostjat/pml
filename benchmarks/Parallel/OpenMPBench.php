<?php
declare(strict_types=1);

namespace Pml\Benchmarks\Parallel;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Transformers\WordCountVectorizer;

/**
 * OpenMP parallelism benchmarks.
 *
 * Benchmarks operations that use `#pragma omp parallel for` in the C engine:
 *   - tensor_sum_axis (axis=1: OpenMP over rows)
 *   - df_transform_bow (OpenMP over document rows)
 *   - tensor_add / tensor_mul (OpenMP over elements)
 *   - row-wise softmax (OpenMP over rows)
 *   - DataFrame load (OpenMP over rows in CSV parser)
 *
 * Scaling behavior: runtime should decrease as thread count increases
 * (set OMP_NUM_THREADS=1/2/4/8 externally to measure scaling).
 *
 * Groups:
 *   parallel    — all OpenMP benchmarks
 *   rowwise     — operations parallelized over rows
 *   elementwise — operations parallelized over elements
 *   nlp         — NLP ops with OpenMP
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['parallel', 'openmp'])]
final class OpenMPBench
{
    private static Tensor $bigMat;      // [10k, 512] — row-parallel ops
    private static Tensor $wideMat;    // [512, 10k] — axis=0 column parallel
    private static Tensor $vec10M;     // element-parallel
    private static Tensor $logits4kx1k; // row-wise softmax
    private static Dataset $rawNlpDs;
    private static WordCountVectorizer $wcv;
    private static string $csvPath;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }

        self::$bigMat      = Tensor::randomNormal([10_000, 512]);
        self::$wideMat     = Tensor::randomNormal([512, 10_000]);
        self::$vec10M      = Tensor::randomNormal([10_000_000]);
        self::$logits4kx1k = Tensor::randomNormal([4000, 1000]);

        self::$csvPath = \sys_get_temp_dir() . '/pml_omp_bench_' . \getmypid() . '.csv';
        self::writeCsv(self::$csvPath, 5000);
        self::$rawNlpDs = Dataset::load(self::$csvPath, hasHeader: true);

        self::$wcv = new WordCountVectorizer(1000, textColumn: 'text');
        self::$wcv->fit(self::$rawNlpDs);

        self::$initialized = true;
    }

    public function __destruct()
    {
        @\unlink(self::$csvPath);
    }

    private static function writeCsv(string $path, int $n): void
    {
        $words = ['good', 'bad', 'great', 'terrible', 'excellent', 'poor',
                  'awesome', 'awful', 'nice', 'horrible', 'love', 'hate'];
        $fh = \fopen($path, 'w');
        \fputcsv($fh, ['text', 'label']);
        \mt_srand(7);
        for ($i = 0; $i < $n; $i++) {
            $len  = \mt_rand(8, 25);
            $text = '';
            for ($j = 0; $j < $len; $j++) {
                $text .= ($j ? ' ' : '') . $words[\mt_rand(0, \count($words) - 1)];
            }
            \fputcsv($fh, [$text, (string)($i % 2)]);
        }
        \fclose($fh);
    }

    // =========================================================================
    // ROW-WISE PARALLEL OPS
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['parallel', 'rowwise'])]
    public function benchSumAxis1_10kx512(): void
    {
        // axis=1 = sum each row → [10k] result; OpenMP over rows
        $r = self::$bigMat->sumAxis(1);
        unset($r);
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['parallel', 'rowwise'])]
    public function benchMeanAxis1_10kx512(): void
    {
        $r = self::$bigMat->meanAxis(1);
        unset($r);
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['parallel', 'rowwise'])]
    public function benchRowSoftmax4kx1k(): void
    {
        $t = self::$logits4kx1k->copy();
        $t->rowSoftmaxInplace();
        unset($t);
    }

    // =========================================================================
    // COLUMN-WISE PARALLEL OPS
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['parallel', 'rowwise'])]
    public function benchSumAxis0_10kx512(): void
    {
        // axis=0 = sum each column → [512] result; tiled accumulation
        $r = self::$bigMat->sumAxis(0);
        unset($r);
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['parallel', 'rowwise'])]
    public function benchSumAxis0_512x10k(): void
    {
        $r = self::$wideMat->sumAxis(0);
        unset($r);
    }

    // =========================================================================
    // ELEMENT-WISE PARALLEL OPS (OpenMP over flat array)
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['parallel', 'elementwise'])]
    public function benchAdd10M(): void
    {
        $r = self::$vec10M->add(self::$vec10M);
        unset($r);
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['parallel', 'elementwise'])]
    public function benchSigmoid10M(): void
    {
        $r = self::$vec10M->sigmoid();
        unset($r);
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['parallel', 'elementwise'])]
    public function benchExp10M(): void
    {
        $r = self::$vec10M->exp();
        unset($r);
    }

    // =========================================================================
    // NLP PARALLEL (df_transform_bow OpenMP over document rows)
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['parallel', 'nlp'])]
    public function benchTransformBow5k(): void
    {
        $ds = self::$wcv->transform(self::$rawNlpDs);
        unset($ds);
    }
}
