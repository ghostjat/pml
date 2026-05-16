<?php
declare(strict_types=1);

namespace Pml\Benchmarks;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Estimators\Classifiers\GaussianNB;
use Pml\Estimators\Classifiers\RandomForestClassifier;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Layers\Softmax;
use Pml\Losses\CategoricalCrossEntropy;
use Pml\NeuralNetwork\Optimizers\Adam;
use Pml\Transformers\WordCountVectorizer;
use Pml\Transformers\ImageResizer;

#[Bench\BeforeMethods('setUp')]
#[Bench\Warmup(1)]
#[Bench\Iterations(3)]
final class PmlFrameworkBench
{
    private static Dataset $nnDataset;
    private static Dataset $rfDataset;
    private static Dataset $imgDataset;
    private static Dataset $textDataset;
    private static Tensor $matA;
    private static Tensor $matB;
    private static Sequential $nn;
    private static RandomForestClassifier $rf;
    private static WordCountVectorizer $wordVectorizer;
    private static ImageResizer $imageResizer;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }

        self::$matA = Tensor::randomUniform([1000, 1000]);
        self::$matB = Tensor::randomUniform([1000, 1000]);

        $samples = Tensor::randomNormal([10000, 50]);
        $rawLabels = Tensor::randomUniform([10000], 0, 1)->round();
        self::$rfDataset = new Dataset($samples, $rawLabels);

        $nnRows = 2048;
        $nnSamples = $samples->slice(0, 0, $nnRows);
        $oneHotLabels = Tensor::zeros($nnRows, 2);
        $labelBuffer = $oneHotLabels->buffer();
        $rawBuffer = $rawLabels->slice(0, 0, $nnRows)->buffer();
        for ($i = 0; $i < $nnRows; $i++) {
            $labelBuffer[$i * 2 + (int)$rawBuffer[$i]] = 1.0;
        }
        self::$nnDataset = new Dataset($nnSamples, $oneHotLabels);

        self::$imgDataset = new Dataset(Tensor::randomUniform([8, 3, 224, 224], 0, 255));

        $textFile = sys_get_temp_dir() . '/pml_bench_text_dataset.csv';
        if (!file_exists($textFile)) {
            $handle = fopen($textFile, 'w');
            fputcsv($handle, ['text', 'label']);
            for ($i = 0; $i < 500; $i++) {
                fputcsv($handle, ["The quick brown hardware accelerated PML engine jumps over the slow PHP loops.", $i % 2]);
            }
            fclose($handle);
        }
        self::$textDataset = Dataset::load($textFile, true);
        self::$wordVectorizer = new WordCountVectorizer(1000);
        self::$wordVectorizer->fit(self::$textDataset);

        self::$nn = new Sequential([
            new Dense(50, 512),
            new ReLU(),
            new Dense(512, 512),
            new ReLU(),
            new Dense(512, 2),
            new Softmax()
        ], new CategoricalCrossEntropy(), new Adam(0.001));

        self::$rf = new RandomForestClassifier(nEstimators: 10, maxDepth: 5);
        self::$rf->train(self::$rfDataset->slice(0, 1000));

        self::$imageResizer = new ImageResizer(112, 112);
        self::$initialized = true;
    }

    #[Bench\Groups(['tensor', 'linalg'])]
    #[Bench\Revs(5)]
    public function benchMatrixMatmul1000(): void
    {
        $result = self::$matA->matmul(self::$matB);
        unset($result);
    }

    #[Bench\Groups(['nn', 'training'])]
    #[Bench\Revs(1)]
    public function benchNeuralNetworkEpoch(): void
    {
        self::$nn->train(self::$nnDataset, epochs: 1, batchSize: 64);
    }

    #[Bench\Groups(['ensembles', 'inference'])]
    #[Bench\Revs(1)]
    public function benchRandomForestPredict(): void
    {
        self::$rf->predict(self::$rfDataset->slice(0, 1000));
    }

    #[Bench\Groups(['nlp', 'vectorizer'])]
    #[Bench\Revs(2)]
    public function benchNLPVectorizationThroughput(): void
    {
        $result = self::$wordVectorizer->transform(self::$textDataset);
        unset($result);
    }

    #[Bench\Groups(['images', 'resizer'])]
    #[Bench\Revs(2)]
    public function benchImageResizing8Batch(): void
    {
        $result = self::$imageResizer->transform(self::$imgDataset);
        unset($result);
    }
}
