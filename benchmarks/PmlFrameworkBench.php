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

/**
 * PML Framework Throughput Benchmark.
 * Measures the computational efficiency of each major subsystem.
 * Optimized to prevent process timeouts in restricted environments.
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Warmup(1)]
#[Bench\Iterations(3)]
final class PmlFrameworkBench
{
    private Dataset $nnDataset;
    private Dataset $rfDataset;
    private Dataset $imgDataset;
    private array $texts;
    private Sequential $nn;

    public function setUp(): void
    {
        // 1. Tabular Data Generation
        $samples = Tensor::randomNormal([10000, 50]);
        $rawLabels = Tensor::randomUniform([10000], 0, 1)->round();

        // 2. Random Forest Dataset (Uses full 10k set for slicing)
        $this->rfDataset = new Dataset($samples, $rawLabels);

        // 3. Neural Network Dataset (Optimized size to 2048 to prevent timeouts)
        $nnRows = 2048;
        $nnSamples = $samples->slice(0, 0, $nnRows);
        $oneHotLabels = Tensor::zeros($nnRows, 2);
        $flatLabels = $rawLabels->slice(0, 0, $nnRows)->toFlatArray();
        $buffer = $oneHotLabels->buffer();
        
        foreach ($flatLabels as $i => $val) {
            $buffer[$i * 2 + (int)$val] = 1.0;
        }
        $this->nnDataset = new Dataset($nnSamples, $oneHotLabels);

        // 4. Image Dataset: 8 Images (Reduced batch for speed), 3 Channels, 224x224
        $this->imgDataset = new Dataset(Tensor::randomUniform([8, 3, 224, 224], 0, 255));

        // 5. NLP Corpus: 500 Documents
        $this->texts = array_fill(0, 500, "The quick brown hardware accelerated PML engine jumps over the slow PHP loops.");

        // 6. Neural Network: 1M Parameters (Binary Classifier)
        $this->nn = new Sequential([
            new Dense(50, 512),
            new ReLU(),
            new Dense(512, 512),
            new ReLU(),
            new Dense(512, 2),
            new Softmax()
        ], new CategoricalCrossEntropy(), new Adam(0.001));
    }

    /**
     * Measure OpenBLAS matrix multiplication throughput.
     */
    #[Bench\Groups(['tensor', 'linalg'])]
    #[Bench\Revs(5)]
    public function benchMatrixMatmul1000(): void
    {
        $a = Tensor::randomUniform([1000, 1000]);
        $b = Tensor::randomUniform([1000, 1000]);
        $a->matmul($b);
    }

    /**
     * Measure Deep Learning Training Speed (Backprop + Adam).
     * Now processes 32 batches of 64 per iteration.
     */
    #[Bench\Groups(['nn', 'training'])]
    #[Bench\Revs(1)]
    public function benchNeuralNetworkEpoch(): void
    {
        $this->nn->train($this->nnDataset, epochs: 1, batchSize: 64);
    }

    /**
     * Measure Random Forest Inference Latency.
     */
    #[Bench\Groups(['ensembles', 'inference'])]
    #[Bench\Revs(1)]
    public function benchRandomForestPredict(): void
    {
        $rf = new RandomForestClassifier(nEstimators: 10, maxDepth: 5);
        $rf->train($this->rfDataset->slice(0, 500));
        $rf->predict($this->rfDataset->slice(0, 1000));
    }

    /**
     * Measure NLP Vectorization Speed.
     */
    #[Bench\Groups(['nlp', 'vectorizer'])]
    #[Bench\Revs(2)]
    public function benchNLPVectorizationThroughput(): void
    {
        $vec = new WordCountVectorizer(maxFeatures: 1000);
        $vec->fit($this->texts);
        $vec->transform($this->texts);
    }

    /**
     * Measure Image Processing Throughput.
     */
    #[Bench\Groups(['images', 'resizer'])]
    #[Bench\Revs(2)]
    public function benchImageResizing8Batch(): void
    {
        $resizer = new ImageResizer(112, 112);
        $resizer->transform($this->imgDataset);
    }
}