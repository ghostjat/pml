<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
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
final class KNNClassifier implements Learner
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

        $testX = $dataset->samples();
        $nTest = $testX->shape()[0];
        $nTrain = $this->fitSamples->shape()[0];
        
        // Ensure K doesn't exceed the number of training samples
        $k = min($this->k, $nTrain);
        
        $preds = [];

        // JIT Loop: Iterates over test samples, but offloads ALL heavy math to OpenBLAS C-Memory
        for ($i = 0; $i < $nTest; $i++) {
            
            // 1. Extract the single test instance (Zero-copy view) -> Shape: [D]
            $x = $testX->row($i);

            // 2. Broadcast Subtraction: [N_train, D] - [D]
            // This computes the difference between the test point and EVERY training point simultaneously.
            $diff = $this->fitSamples->sub($x);

            // 3. Squared Euclidean Distance: sum( (X_train - x)^2, axis=1 )
            // We use Squared distance because sqrt() is monotonically increasing and unnecessary for sorting.
            $sqDist = $diff->square()->sumAxis(1);

            // 4. Sort distances and extract the Top K indices
            // argsort() executes natively via C QuickSort and returns the integer indices ascending
            $sortedIndices = $sqDist->argsort();
            $kIndices = $sortedIndices->slice(0, 0, $k);

            // 5. Gather the ground-truth labels of the K nearest neighbors
            $kLabels = $this->fitLabels->take($kIndices, 0);

            // 6. Majority Vote
            // tensor_bincount tallies the frequencies, argmax() safely returns the highest integer class.
            $vote = (float) $kLabels->bincount()->argmax();
            $preds[] = $vote;

            // Memory Lifecycle Note:
            // $x, $diff, $sqDist, $sortedIndices, $kIndices, and $kLabels fall out of scope here.
            // PHP cleanly calls their __destruct() methods, safely freeing the massive C-buffers instantly!
        }

        return Tensor::fromArray($preds);
    }

    public function trained(): bool
    {
        return $this->fitSamples !== null && $this->fitLabels !== null;
    }
}