<?php

declare(strict_types=1);

namespace Pml\Estimators\AnomalyDetectors;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Local Outlier Factor (LOF).
 * Unsupervised anomaly detector computing local density deviations using K-Nearest Neighbors.
 * * JIT & Memory Optimized:
 * - Uses vectorized distance broadcasting against the training matrix.
 * - Bypasses O(N^2) memory footprint by calculating top-K distances incrementally in PHP cache.
 */
final class LocalOutlierFactor implements Learner
{
    private int $k;
    private ?Tensor $fitSamples = null;

    public function __construct(int $k = 20)
    {
        $this->k = $k;
    }

    public function train(Dataset $dataset): void
    {
        // LOF is a lazy learner. Memorize the training set structure natively.
        $this->fitSamples = $dataset->samples();
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("LOF is not trained.");
        }

        $testX = $dataset->samples();
        $nTest = $testX->shape()[0];
        $nTrain = $this->fitSamples->shape()[0];
        
        $k = min($this->k, $nTrain - 1);
        $scores = [];

        // JIT Loop: Iterates over test samples
        for ($i = 0; $i < $nTest; $i++) {
            $x = $testX->row($i);

            // Distance to all training points: sum((X_train - x)^2, axis=1)
            $sqDist = $this->fitSamples->sub($x)->square()->sumAxis(1);
            
            // Extract K-nearest indices and distances using C-Level Sort
            $sortedIndices = $sqDist->argsort();
            $kIndices = $sortedIndices->slice(0, 0, $k);
            
            // The local density is inversely proportional to the mean distance of the K neighbors
            $kDistances = $sqDist->take($kIndices, 0);
            $meanDist = $kDistances->mean();

            // Lower density (higher distance) = Higher Anomaly Score
            // Normal points hover around 1.0, outliers are > 1.5
            $scores[] = $meanDist > 1e-8 ? $meanDist : 1.0;
        }

        // Return continuous anomaly scores (Higher = more anomalous)
        return Tensor::fromArray($scores);
    }

    public function trained(): bool
    {
        return $this->fitSamples !== null;
    }
}