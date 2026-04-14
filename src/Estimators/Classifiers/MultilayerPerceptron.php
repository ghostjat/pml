<?php
declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use Pml\Tensor;
use Pml\Dataset;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Layers\Softmax;
use Pml\NeuralNetwork\Layers\Dropout;
use Pml\NeuralNetwork\Optimizers\Adam;
use Pml\Losses\CategoricalCrossEntropy;
use RuntimeException;

/**
 * Multilayer Perceptron Classifier.
 * Wraps the native Sequential model with a classification-ready head.
 *
 * JIT & Memory Optimized:
 * - Delegates all computation to Sequential / C-level tensor ops.
 * - One-hot encoding lives in C memory; PHP only reads argmax integer indices.
 */
final class MultilayerPerceptron implements Learner, Probabilistic
{
    private ?Sequential $network = null;
    private array $classMap      = [];
    private array $indexMap      = [];

    /**
     * @param int[]  $hidden       Hidden layer sizes, e.g. [128, 64]
     * @param float  $dropout      Dropout rate per hidden layer (0 = no dropout)
     */
    public function __construct(
        private readonly array $hidden       = [100],
        private readonly int   $epochs       = 100,
        private readonly int   $batchSize    = 32,
        private readonly float $learningRate = 0.001,
        private readonly float $dropout      = 0.0
    ) {}

    public function train(Dataset $dataset): void
    {
        $labels = $dataset->labels();
        if ($labels === null) {
            throw new \InvalidArgumentException("MultilayerPerceptron requires labeled data.");
        }

        $flat   = $labels->toFlatArray();
        $unique = array_values(array_unique($flat));
        sort($unique);
        // array_flip() cannot handle float keys; use string representation
        $this->classMap = array_flip(array_map('strval', $unique));
        $this->indexMap = $unique;
        $k = count($unique);
        $d = $dataset->numColumns();

        // Build network architecture
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
        $layers[] = new Dense($inSize, $k);
        $layers[] = new Softmax();

        $this->network = new Sequential(
            $layers,
            new CategoricalCrossEntropy(),
            new Adam($this->learningRate)
        );

        // CategoricalCrossEntropy expects one-hot labels [N × K].
        // Build the one-hot matrix in PHP, then wrap in a new Dataset.
        $nSamples   = count($flat);
        $oneHotData = array_fill(0, $nSamples, array_fill(0, $k, 0.0));
        foreach ($flat as $i => $label) {
            $classIdx = $this->classMap[(string) $label];
            $oneHotData[$i][$classIdx] = 1.0;
        }
        $trainDataset = new Dataset($dataset->samples(), Tensor::fromArray($oneHotData));

        $this->network->train($trainDataset, $this->epochs, $this->batchSize);
    }

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("MultilayerPerceptron is not trained.");
        }
        return $this->network->predict($dataset);                       // [N × K]
    }

    public function predict(Dataset $dataset): Tensor
    {
        $proba   = $this->proba($dataset);
        $k       = count($this->indexMap);
        $indices = $proba->argsort(1)->col($k - 1)->toFlatArray();

        $preds = [];
        foreach ($indices as $idx) {
            $preds[] = $this->indexMap[(int) $idx] ?? 0;
        }
        return Tensor::fromArray($preds);
    }

    public function trained(): bool
    {
        return $this->network !== null;
    }
}
