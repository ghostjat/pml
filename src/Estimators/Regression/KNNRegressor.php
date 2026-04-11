<?php

declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * K-Nearest Neighbors (KNN) Regressor.
 * An instance-based "lazy learner" that predicts continuous values by averaging the K closest neighbors.
 * * JIT & Memory Optimized:
 * - 100% Zero-Copy Lazy Learning (caches C-pointers during fit).
 * - Inference utilizes AVX2 Vector Broadcasting for simultaneous multi-dimensional Euclidean Distance.
 */
final class KNNRegressor implements Learner
{
    private int $k;
    
    // Cached pointers to the training data (Zero-Copy)
    private ?Tensor $fitSamples = null;
    private ?Tensor $fitLabels = null;

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

        $this->fitSamples = $dataset->samples();
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("K-Nearest Neighbors is not trained.");
        }

        $testX = $dataset->samples();
        $nTest = $testX->shape()[0];
        $nTrain = $this->fitSamples->shape()[0];
        
        $k = min($this->k, $nTrain);
        $preds = [];

        // JIT Loop: Iterates over test samples, offloads heavy math to C-Memory
        for ($i = 0; $i < $nTest; $i++) {
            
            $x = $testX->row($i);

            // Broadcast Subtraction & Squared Distance
            $sqDist = $this->fitSamples->sub($x)->square()->sumAxis(1);

            // Sort distances and extract the Top K indices natively
            $sortedIndices = $sqDist->argsort();
            $kIndices = $sortedIndices->slice(0, 0, $k);

            // Gather the ground-truth continuous labels of the K nearest neighbors
            $kLabels = $this->fitLabels->take($kIndices, 0);

            // Average the values for the final prediction
            $preds[] = $kLabels->mean();
            
            // Temporary tensors fall out of scope and memory is freed instantly
        }

        return Tensor::fromArray($preds);
    }

    public function trained(): bool
    {
        return $this->fitSamples !== null && $this->fitLabels !== null;
    }
}