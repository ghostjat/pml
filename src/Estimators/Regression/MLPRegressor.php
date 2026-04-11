<?php
declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Layers\Dropout;
use Pml\NeuralNetwork\Optimizers\Adam;
use RuntimeException;

/**
 * MLP Regressor — Neural Network for continuous target prediction.
 *
 * JIT & Memory Optimized:
 * - Delegates all forward/backward passes to Sequential / C-level ops.
 * - Output is a single linear unit (no activation) producing raw real-valued predictions.
 */
final class MLPRegressor implements Learner
{
    private ?Sequential $network = null;

    public function __construct(
        private readonly array $hidden       = [100],
        private readonly int   $epochs       = 100,
        private readonly int   $batchSize    = 32,
        private readonly float $learningRate = 0.001,
        private readonly float $dropout      = 0.0
    ) {}

    public function train(Dataset $dataset): void
    {
        if ($dataset->labels() === null) {
            throw new \InvalidArgumentException("MLPRegressor requires labeled data.");
        }

        $d      = $dataset->numColumns();
        $layers = [];
        $inSize = $d;

        foreach ($this->hidden as $units) {
            $layers[] = new Dense($inSize, $units);
            $layers[] = new ReLU();
            if ($this->dropout > 0.0) {
                $layers[] = new Dropout($this->dropout);
            }
            $inSize = $units;
        }
        // Linear output — no activation for regression
        $layers[] = new Dense($inSize, 1);

        $this->network = new Sequential($layers, new Adam($this->learningRate));
        $this->network->train($dataset, $this->epochs, $this->batchSize);
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("MLPRegressor is not trained.");
        }
        return $this->network->predict($dataset)->squeeze();           // [N]
    }

    public function trained(): bool
    {
        return $this->network !== null;
    }
}
