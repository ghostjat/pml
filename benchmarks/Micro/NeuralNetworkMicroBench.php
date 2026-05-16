<?php
declare(strict_types=1);

namespace Pml\Benchmarks\Micro;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Optimizers\SGD;

#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['micro', 'neuralnetwork', 'ffi'])]
final class NeuralNetworkMicroBench
{
    private static Dense $layer;
    private static Tensor $input;
    private static Tensor $dY;
    private static SGD $optimizer;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }

        self::$layer = new Dense(128, 64);
        self::$input = Tensor::randomNormal([32, 128]);
        self::$dY = Tensor::randomNormal([32, 64]);
        self::$optimizer = new SGD(0.01);
        self::$initialized = true;
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    public function benchForwardPass(): void
    {
        $output = self::$layer->forward(self::$input);
        unset($output);
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    public function benchBackwardPass(): void
    {
        $output = self::$layer->forward(self::$input);
        $dX = self::$layer->backward(self::$dY);
        unset($output, $dX);
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    public function benchOptimizerStep(): void
    {
        $output = self::$layer->forward(self::$input);
        $dX = self::$layer->backward(self::$dY);
        self::$optimizer->step([self::$layer]);
        unset($output, $dX);
    }
}
