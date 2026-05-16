<?php
declare(strict_types=1);

namespace Pml\Benchmarks\Macro;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\Dataset;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Optimizers\SGD;
use Pml\Losses\MeanSquaredError;

#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['macro', 'training', 'inference'])]
final class TrainingMacroBench
{
    private static Sequential $model;
    private static Dataset $dataset;
    private static Tensor $inferenceInput;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }

        $layers = [
            new Dense(784, 128),
            new ReLU(),
            new Dense(128, 10)
        ];
        $loss = new MeanSquaredError();
        $optimizer = new SGD(0.01);
        self::$model = new Sequential($layers, $loss, $optimizer);
        self::$dataset = self::createSyntheticDataset();
        self::$inferenceInput = Tensor::randomNormal([1, 784]);
        self::$initialized = true;
    }

    private static function createSyntheticDataset(): Dataset
    {
        $xData = Tensor::randomNormal([1000, 784]);
        $yData = Tensor::randomNormal([1000, 10]);
        return new Dataset($xData, $yData);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchFullTrainingLoop(): void
    {
        $batchSize = 32;

        foreach (self::$dataset->batches($batchSize) as $batch) {
            $x = $batch->samples();
            $y = $batch->labels();
            $pred = self::$model->forward($x);
            $dLoss = self::$model->getLoss()->differentiate($pred, $y);
            self::$model->backward($dLoss);
            self::$model->getOptimizer()->step(self::$model->getLayers());
            unset($pred, $dLoss);
        }
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    public function benchDatasetThroughput(): void
    {
        $batchSize = 64;
        foreach (self::$dataset->batches($batchSize) as $batch) {
            $batch->samples();
            $batch->labels();
        }
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    public function benchInference(): void
    {
        $output = self::$model->forward(self::$inferenceInput);
        unset($output);
    }
}
