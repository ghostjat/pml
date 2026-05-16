<?php
declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
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
final class MLPClassifier implements Learner, Probabilistic, Persistable
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

    public function train(Dataset $dataset, mixed ...$options): void
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

        // One-hot in C: map float labels → class indices → [N,K] one-hot (single FFI call)
        $idxArr  = array_map(fn($l) => (float)($this->classMap[(string)$l] ?? 0), $flat);
        $idxT    = Tensor::fromArray($idxArr);
        $oneHot  = Tensor::onehot($idxT, $k);                    // [N × K], C-only
        $trainDataset = new Dataset($dataset->samples(), $oneHot);

        $this->network->train($trainDataset, $this->epochs, $this->batchSize);
    }

    /**
     * Incremental update — one epoch on new data, preserving existing weights.
     * Call train() first, then partial() for each incoming mini-batch.
     */
    public function partial(Dataset $dataset, int $epochs = 1): void
    {
        if (!$this->trained()) {
            throw new RuntimeException("Call train() before partial().");
        }
        $labels = $dataset->labels();
        if ($labels === null) {
            throw new \InvalidArgumentException("partial() requires labeled data.");
        }
        $flat = $labels->toFlatArray();
        $k    = count($this->indexMap);
        $idxArr       = array_map(fn($l) => (float)($this->classMap[(string)$l] ?? 0), $flat);
        $idxT         = Tensor::fromArray($idxArr);
        $oneHot       = Tensor::onehot($idxT, $k);
        $trainDataset = new Dataset($dataset->samples(), $oneHot);
        $this->network->train($trainDataset, epochs: $epochs, batchSize: $this->batchSize);
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
        $labelTable = Tensor::fromArray(array_map('floatval', $this->indexMap));
        return Tensor::gatherIndices($this->proba($dataset)->argmaxAxis(1), $labelTable);
    }

    public function trained(): bool
    {
        return $this->network !== null;
    }

    public function save(string $dir): void
    {
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }

        file_put_contents(
            $dir . \DIRECTORY_SEPARATOR . 'config.json',
            json_encode([
                'class'        => self::class,
                'hidden'       => $this->hidden,
                'epochs'       => $this->epochs,
                'batchSize'    => $this->batchSize,
                'learningRate' => $this->learningRate,
                'dropout'      => $this->dropout,
                'classMap'     => $this->classMap,
                'indexMap'     => $this->indexMap,
            ], \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES)
        );

        if ($this->network !== null) {
            $this->network->save($dir . \DIRECTORY_SEPARATOR . 'network');
        }
    }

    public static function load(string $dir): self
    {
        $raw = file_get_contents($dir . \DIRECTORY_SEPARATOR . 'config.json');
        if ($raw === false) {
            throw new \RuntimeException("MultilayerPerceptron::load — config.json missing in '$dir'.");
        }
        $config = json_decode($raw, true, 512, \JSON_THROW_ON_ERROR);

        $instance = new self(
            (array) $config['hidden'],
            (int)   $config['epochs'],
            (int)   $config['batchSize'],
            (float) $config['learningRate'],
            (float) $config['dropout']
        );
        $instance->classMap = $config['classMap'];
        $instance->indexMap = $config['indexMap'];
        $instance->network  = \Pml\NeuralNetwork\Sequential::load(
            $dir . \DIRECTORY_SEPARATOR . 'network'
        );

        return $instance;
    }
}
