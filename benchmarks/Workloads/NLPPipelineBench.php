<?php
declare(strict_types=1);

namespace Pml\Benchmarks\Workloads;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Transformers\WordCountVectorizer;
use Pml\Transformers\TfIdfTransformer;
use Pml\Transformers\ZScaleStandardizer;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Layers\Softmax;
use Pml\Losses\CategoricalCrossEntropy;
use Pml\NeuralNetwork\Optimizers\Adam;

/**
 * End-to-end NLP pipeline benchmarks.
 *
 * Covers the full sentiment-classification pipeline:
 *   CSV ETL → WCV vocab build → pipeline_transform_batch → MLP step
 *
 * Groups:
 *   nlp         — all NLP benchmarks
 *   tokenizer   — vocab build / BoW transform
 *   pipeline    — C-side pipeline_transform_batch (zero PHP loop per batch)
 *   training    — MLP stepOnBatch throughput
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['nlp', 'workload'])]
final class NLPPipelineBench
{
    private static Dataset $rawDataset;
    private static WordCountVectorizer $wcv;
    private static TfIdfTransformer $tfidf;
    private static ZScaleStandardizer $zscale;
    private static Tensor $bowMatrix;
    private static Dataset $vectorizedDataset;
    private static Sequential $mlp;
    private static string $csvPath;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }

        self::$csvPath = \sys_get_temp_dir() . '/pml_nlp_bench_' . \getmypid() . '.csv';
        self::writeSyntheticCsv(self::$csvPath, 5000);

        // column 0 = 'text', column 1 = 'label' — must declare for WCV::transform() to carry labels
        self::$rawDataset = Dataset::load(self::$csvPath, hasHeader: true)->withLabelColumn(1);

        self::$wcv = new WordCountVectorizer(2000, textColumn: 'text');
        self::$wcv->fit(self::$rawDataset);

        self::$vectorizedDataset = self::$wcv->transform(self::$rawDataset);

        self::$tfidf = new TfIdfTransformer();
        self::$tfidf->fit(self::$vectorizedDataset);
        $tfidfDs = self::$tfidf->transform(self::$vectorizedDataset);

        self::$zscale = new ZScaleStandardizer();
        self::$zscale->fit($tfidfDs);

        self::$bowMatrix = self::$vectorizedDataset->samples();

        $vocabSize = self::$wcv->vocabSize();
        self::$mlp = new Sequential([
            new Dense($vocabSize, 256),
            new ReLU(),
            new Dense(256, 64),
            new ReLU(),
            new Dense(64, 2),
            new Softmax(),
        ], new CategoricalCrossEntropy(), new Adam(0.001));

        self::$initialized = true;
    }

    public function __destruct()
    {
        @\unlink(self::$csvPath);
    }

    private static function writeSyntheticCsv(string $path, int $n): void
    {
        $words = ['good', 'bad', 'great', 'terrible', 'excellent', 'poor', 'awesome',
                  'awful', 'nice', 'horrible', 'love', 'hate', 'best', 'worst', 'happy',
                  'sad', 'wonderful', 'dreadful', 'amazing', 'disappointing'];
        $fh = \fopen($path, 'w');
        \fputcsv($fh, ['text', 'label']);
        \mt_srand(42);
        for ($i = 0; $i < $n; $i++) {
            $len  = \mt_rand(6, 20);
            $text = '';
            for ($j = 0; $j < $len; $j++) {
                $text .= ($j ? ' ' : '') . $words[\mt_rand(0, \count($words) - 1)];
            }
            \fputcsv($fh, [$text, (string)($i % 2)]);
        }
        \fclose($fh);
    }

    // =========================================================================
    // VOCAB BUILD — measures _token_next_nb (zero-malloc) tokenizer
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['nlp', 'tokenizer'])]
    public function benchWcvFit5k(): void
    {
        $wcv = new WordCountVectorizer(2000, textColumn: 'text');
        $wcv->fit(self::$rawDataset);
        unset($wcv);
    }

    // =========================================================================
    // BOW TRANSFORM — OpenMP parallel tokenize+count per row
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['nlp', 'tokenizer'])]
    public function benchWcvTransform5k(): void
    {
        $ds = self::$wcv->transform(self::$rawDataset);
        unset($ds);
    }

    // =========================================================================
    // TFIDF + ZSCALE TRANSFORM (PHP transformer chain)
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['nlp', 'pipeline'])]
    public function benchTfIdfTransform(): void
    {
        $ds = self::$tfidf->transform(self::$vectorizedDataset);
        unset($ds);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['nlp', 'pipeline'])]
    public function benchZScaleTransform(): void
    {
        $tfidfDs = self::$tfidf->transform(self::$vectorizedDataset);
        $ds = self::$zscale->transform($tfidfDs);
        unset($tfidfDs, $ds);
    }

    // =========================================================================
    // MLP TRAINING — single-batch stepOnBatch throughput
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(10)]
    #[Bench\Groups(['nlp', 'training'])]
    public function benchMLPForwardOnBatch(): void
    {
        $x = self::$vectorizedDataset->samples()->slice(0, 0, 64);
        $out = self::$mlp->forward($x);
        unset($x, $out);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['nlp', 'training'])]
    public function benchMLPForwardBackwardStep64(): void
    {
        $batch = self::$vectorizedDataset->slice(0, 64);
        $x = $batch->samples();
        $labels = $batch->labels();

        // build one-hot labels from class indices (0 or 1)
        $yOneHot = Tensor::zeros(64, 2);
        $buf     = $yOneHot->buffer();
        $lBuf    = $labels->buffer();
        for ($i = 0; $i < 64; $i++) {
            $buf[$i * 2 + (int)$lBuf[$i]] = 1.0;
        }

        $cce  = new CategoricalCrossEntropy();
        $out  = self::$mlp->forward($x);
        $dY   = $cce->differentiate($out, $yOneHot);
        self::$mlp->backward($dY);
        unset($out, $dY, $batch, $x, $labels, $yOneHot);
    }
}
