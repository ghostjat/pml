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
    
    private ?Tensor $fitSamples = null;
    private ?Tensor $fitLabels  = null;
    private int     $numClasses = 0;

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

        $this->fitSamples  = $dataset->samples();
        $this->numClasses  = (int)($this->fitLabels->max() + 1);
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

        // Gather neighbor labels → [nTest, k], then majority vote in C (one call)
        $nTest   = $testX->shape()[0];
        $flat    = $kNeighbors->reshape($nTest * $k);
        $kLabels = $this->fitLabels->take($flat, 0)->reshape($nTest, $k);
        return Tensor::knnVote($kLabels, $this->numClasses);
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
                ['class' => self::class, 'k' => $this->k, 'numClasses' => $this->numClasses],
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
        $instance->numClasses = (int) ($config['numClasses'] ?? 0);

        $stPath = $dir . \DIRECTORY_SEPARATOR . 'model.safetensors';
        if (is_file($stPath)) {
            $tensors = SafeTensorsIO::load($stPath);
            $instance->fitSamples = $tensors['fit_samples'] ?? null;
            $instance->fitLabels  = $tensors['fit_labels']  ?? null;
        }

        return $instance;
    }
}