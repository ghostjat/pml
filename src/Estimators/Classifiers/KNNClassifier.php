<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * K-Nearest Neighbors (KNN) Classifier.
 * An instance-based "lazy learner" that classifies samples based on distance.
 * * JIT & Memory Optimized:
 * - 100% Zero-Copy Lazy Learning (merely caches C-pointers during fit).
 * - Inference utilizes AVX2 Vector Broadcasting for simultaneous multi-dimensional Euclidean Distance.
 * - Majority voting leverages C-level `bincount` and `argmax` to bypass PHP iterations.
 */
final class KNNClassifier implements Learner, Persistable
{
    private int $k;
    
    // Cached pointers to the training data (Zero-Copy)
    private ?Tensor $fitSamples = null;
    private ?Tensor $fitLabels = null;

    /**
     * @param int $k The number of closest neighbors to consider for the majority vote.
     */
    public function __construct(int $k = 5)
    {
        if ($k < 1) {
            throw new \InvalidArgumentException("K must be at least 1.");
        }
        $this->k = $k;
    }

    public function train(Dataset $dataset): void
    {
        $this->fitLabels = $dataset->labels();
        
        if ($this->fitLabels === null) {
            throw new \InvalidArgumentException("K-Nearest Neighbors requires a labeled dataset.");
        }

        // Lazy Learning: KNN does not "build" a model, it simply memorizes the training data.
        // We only store the references to the underlying FFI C-Pointers.
        $this->fitSamples = $dataset->samples();
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("K-Nearest Neighbors is not trained.");
        }

        $testX  = $dataset->samples();
        $nTrain = $this->fitSamples->shape()[0];
        $k      = min($this->k, $nTrain);

        // Single BLAS call: [nTest, nTrain] pairwise squared L2 distances
        $distMat    = Tensor::pairwiseSqL2($testX, $this->fitSamples);   // [nTest, nTrain]
        $sortedIdx  = $distMat->argsort(1);                               // [nTest, nTrain] ascending
        $kNeighbors = $sortedIdx->slice(1, 0, $k);                       // [nTest, k]

        // Majority vote per row: gather labels, bincount, argmax — all in C
        $flat    = $kNeighbors->reshape($testX->shape()[0] * $k);           // [nTest*k]
        $kLabels = $this->fitLabels->take($flat, 0)
                                   ->reshape($testX->shape()[0], $k);     // [nTest, k]

        // Per-row bincount + argmax — zero-copy row views, 2 C calls per row
        $nTest = $testX->shape()[0];
        $votes = [];
        for ($i = 0; $i < $nTest; $i++) {
            $votes[] = (float)$kLabels->row($i)->bincount()->argmax();
        }
        return Tensor::fromArray($votes);
    }

    public function trained(): bool
    {
        return $this->fitSamples !== null && $this->fitLabels !== null;
    }

    public function save(string $dir): void
    {
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }

        file_put_contents(
            $dir . \DIRECTORY_SEPARATOR . 'config.json',
            json_encode(
                ['class' => self::class, 'k' => $this->k],
                \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES
            )
        );

        if ($this->fitSamples !== null) {
            SafeTensorsIO::save(
                $dir . \DIRECTORY_SEPARATOR . 'model.safetensors',
                ['fit_samples' => $this->fitSamples, 'fit_labels' => $this->fitLabels]
            );
        }
    }

    public static function load(string $dir): self
    {
        $raw = file_get_contents($dir . \DIRECTORY_SEPARATOR . 'config.json');
        if ($raw === false) {
            throw new \RuntimeException("KNNClassifier::load — config.json missing in '$dir'.");
        }
        $config = json_decode($raw, true, 512, \JSON_THROW_ON_ERROR);

        $instance = new self((int) $config['k']);

        $stPath = $dir . \DIRECTORY_SEPARATOR . 'model.safetensors';
        if (is_file($stPath)) {
            $tensors = SafeTensorsIO::load($stPath);
            $instance->fitSamples = $tensors['fit_samples'] ?? null;
            $instance->fitLabels  = $tensors['fit_labels']  ?? null;
        }

        return $instance;
    }
}