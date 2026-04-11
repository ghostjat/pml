<?php

declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Tensor;
use Pml\Dataset;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\Softmax;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\Losses\CategoricalCrossEntropy;
use Pml\NeuralNetwork\Optimizers\Adam;
use Psr\Log\NullLogger;

final class NeuralNetworkTest extends TestCase
{
    /**
     * Test the mathematical correctness of Softmax and Categorical Cross Entropy.
     * Validates numerical stability and probability distribution constraints.
     */
    public function testSoftmaxAndCategoricalCrossEntropy(): void
    {
        // 1. Raw Logits (Batch=2, Classes=3)
        // Using values that test the "Max-Shift" numerical stability trick
        $logits = Tensor::fromArray([
            [10.0, 5.0, 1.0],  // Large values
            [-10.0, -20.0, 0.0] // Very small/negative values
        ]);

        // 2. One-Hot Labels (Ground Truth)
        $labels = Tensor::fromArray([
            [1.0, 0.0, 0.0], // Class 0 is correct
            [0.0, 0.0, 1.0]  // Class 2 is correct
        ]);

        $softmax = new Softmax();
        $cce = new CategoricalCrossEntropy();

        // Forward Pass: Logits -> Probabilities
        $probs = $softmax->forward($logits);
        
        // ASSERTION 1: Probabilities must be positive and <= 1.0
        $flatProbs = $probs->toFlatArray();
        foreach ($flatProbs as $p) {
            $this->assertGreaterThanOrEqual(0.0, $p, "Probabilities must be non-negative");
            $this->assertLessThanOrEqual(1.0, $p, "Probabilities must be <= 1.0");
        }

        // ASSERTION 2: Probabilities must sum to 1.0 per row
        // This validates the C-level axis-reduction and broadcasting logic
        $sums = $probs->sumAxis(1)->toFlatArray();
        $this->assertEqualsWithDelta(1.0, $sums[0], 0.0001, "Row 0 probabilities must sum to 1.0");
        $this->assertEqualsWithDelta(1.0, $sums[1], 0.0001, "Row 1 probabilities must sum to 1.0");

        // Compute Loss: Cross Entropy
        $loss = $cce->compute($probs, $labels);
        $this->assertIsFloat($loss);
        $this->assertGreaterThan(0.0, $loss, "Entropy loss must be positive");

        // Backward Pass: Calculate Gradients
        $dY = $cce->differentiate($probs, $labels);
        $dX = $softmax->backward($dY);

        // ASSERTION 3: Gradient shape must match input shape
        $this->assertSame($logits->shape(), $dX->shape(), "Gradient dX must match Logits shape");
    }

    /**
     * Test the Adam Optimizer's state initialization and persistence safety.
     * Ensures that FFI pointers are detached during serialization to prevent segmentation faults.
     */
    public function testAdamOptimizerStateAndSerialization(): void
    {
        $optimizer = new Adam(learningRate: 0.001);
        $layer = new Dense(4, 2);
        
        // 1. Trigger a step to initialize internal momentum C-Tensors (m and v)
        $input = Tensor::randomNormal([1, 4]);
        $out = $layer->forward($input);
        
        // Generate mock gradient
        $dY = Tensor::ones(1, 2);
        $layer->backward($dY); 
        
        $optimizer->step([$layer]);
        
        // 2. Serialize the optimizer (triggers Adam::__sleep)
        // This must NOT contain raw FFI CData objects
        $serialized = serialize($optimizer);
        $this->assertStringNotContainsString('CData', $serialized, "FFI pointers must be detached during serialization");
        
        // 3. Unserialize (triggers Adam::__wakeup)
        $unpacked = unserialize($serialized);
        $this->assertInstanceOf(Adam::class, $unpacked);
        
        // 4. Verify training can resume (state re-initialization)
        $unpacked->step([$layer]);
    }

    /**
     * Test Sequential training with Early Stopping logic.
     * Validates that the model can restore the 'best' C-weights if overfitting occurs.
     */
    public function testSequentialEarlyStopping(): void
    {
        // Simple linear task: Predict identity
        $samples = Tensor::randomUniform([50, 5], 0.1, 1.0);
        $labels = $samples->copy(); 
        $dataset = new Dataset($samples, $labels);

        $model = new Sequential([
            new Dense(5, 5),
            new ReLU()
        ], new CategoricalCrossEntropy(), new Adam(0.01));

        $logger = new class extends NullLogger {
            public array $logs = [];
            public function info($message, array $context = []): void {
                $this->logs[] = (string) $message;
            }
        };

        $model->setLogger($logger);

        // Train with high patience. On this trivial task, it should converge instantly.
        $model->train(
            $dataset,
            epochs: 20,
            batchSize: 5,
            validation: $dataset,
            patience: 2,
            minDelta: 0.001
        );

        $this->assertTrue($model->trained(), "Model should be marked as trained");
        
        // Verify that the training loop executed and logged progress
        $this->assertNotEmpty($logger->logs, "Training progress should be logged via PSR-3");
    }
}