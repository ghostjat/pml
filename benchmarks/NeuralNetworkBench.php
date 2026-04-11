<?php

declare(strict_types=1);

namespace Pml\Benchmarks;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\Dataset;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\Softmax;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\Losses\CategoricalCrossEntropy;
use Pml\NeuralNetwork\Optimizers\Adam;

/**
 * Performance profile for the Deep Learning subsystem.
 * Evaluates Softmax throughput, Adam update overhead, and full forward/backward passes.
 * * NOTE: This benchmark profile stress tests the OpenBLAS/LAPACKE logic
 * for both contiguous and non-contiguous matrix operations.
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Warmup(2)]
#[Bench\Revs(5)]
#[Bench\Iterations(3)]
final class NeuralNetworkBench
{
    private Tensor $logits;
    private Tensor $labels;
    private Softmax $softmax;
    private CategoricalCrossEntropy $cce;
    
    private Sequential $model;
    private Dataset $dataset;

    public function setUp(): void
    {
        // 1. Stress Test Data (High Dimensionality)
        // Batch=1024, Classes=1000
        $this->logits = Tensor::randomNormal([1024, 1000]);
        $this->labels = Tensor::zeros(1024, 1000);
        
        // Populate one-hot labels correctly to prevent 0.0 division in CCE
        for ($i = 0; $i < 1024; $i++) {
            $row = $this->labels->row($i);
            $targetClass = rand(0, 999);
            // Ensure values are within FLOAT32 bounds
            $row->buffer()[$targetClass] = 1.0;
        }

        $this->softmax = new Softmax();
        $this->cce = new CategoricalCrossEntropy();

        // 2. Multi-Layer Perceptron (approx 2M Trainable Parameters)
        // Architecture: 512 -> 1024 -> 1024 -> 512 -> 10
        $this->model = new Sequential([
            new Dense(512, 1024),
            new ReLU(),
            new Dense(1024, 1024),
            new ReLU(),
            new Dense(1024, 512),
            new ReLU(),
            new Dense(512, 10),
            new Softmax()
        ], new CategoricalCrossEntropy(), new Adam(learningRate: 0.001));

        // Create a realistic training batch
        $samples = Tensor::randomNormal([128, 512]);
        $targetLabels = Tensor::zeros(128, 10);
        for ($i = 0; $i < 128; $i++) {
            $targetLabels->row($i)->buffer()[rand(0, 9)] = 1.0;
        }
        
        $this->dataset = new Dataset($samples, $targetLabels);
    }

    /**
     * Measures the speed of numerically stable Softmax + Categorical Cross Entropy.
     * Evaluates the AVX2 Max-Shift trick and log-sum-exp performance.
     */
    #[Bench\Groups(['nn', 'math'])]
    public function benchSoftmaxForwardBackward(): void
    {
        // This triggers the most complex row-wise aggregations in the C engine
        $probs = $this->softmax->forward($this->logits);
        $dY = $this->cce->differentiate($probs, $this->labels);
        $this->softmax->backward($dY);
    }

    /**
     * Measures the overhead of the Adam Optimizer's state management and 
     * in-place momentum updates on a deep network (approx 2M parameters).
     */
    #[Bench\Groups(['nn', 'optimizer'])]
    public function benchAdamStepThroughput(): void
    {
        $x = $this->dataset->samples();
        $y = $this->dataset->labels();

        $preds = $this->model->forward($x);
        $dY = $this->cce->differentiate($preds, $y);
        $this->model->backward($dY);
        
        // This is the core bottleneck: Updating millions of momenta in C-memory
        $this->model->train($this->dataset, epochs: 1, batchSize: 128);
    }

    /**
     * Stress tests the full forward pass JIT optimization.
     */
    #[Bench\Groups(['nn', 'forward'])]
    public function benchFullForwardPass(): void
    {
        $this->model->forward($this->dataset->samples());
    }
}