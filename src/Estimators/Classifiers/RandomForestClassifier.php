<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Estimators\Classifiers\DecisionTreeClassifier;
use RuntimeException;

/**
 * Random Forest Ensemble.
 * Trains multiple Decision Trees on bootstrapped datasets to prevent overfitting.
 * * JIT & Memory Optimized:
 * - Bootstrapping implemented via fast PHP-index arrays driving C-level tensor_take().
 * - Inference voting executed entirely in PHP JIT cache.
 */
final class RandomForestClassifier implements Learner
{
    private int $nEstimators;
    private int $maxDepth;
    private int $minSamplesSplit;
    
    /** @var DecisionTreeClassifier[] */
    private array $trees = [];

    public function __construct(int $nEstimators = 100, int $maxDepth = 10, int $minSamplesSplit = 2)
    {
        $this->nEstimators = $nEstimators;
        $this->maxDepth = $maxDepth;
        $this->minSamplesSplit = $minSamplesSplit;
    }

    public function train(Dataset $dataset): void
    {
        $n = $dataset->numRows();
        $features = $dataset->numColumns();
        
        // Feature bagging: each tree only sees a random square root subset of features
        $maxFeatures = (int) max(1, sqrt($features));

        for ($i = 0; $i < $this->nEstimators; $i++) {
            
            // 1. Bootstrap Sampling (Random selection with replacement)
            $indices = [];
            for ($j = 0; $j < $n; $j++) {
                $indices[] = mt_rand(0, $n - 1);
            }
            
            // 2. Extract Bootstrap slice safely in C
            $idxT = Tensor::fromArray($indices);
            $bootX = $dataset->samples()->take($idxT, 0);
            $bootY = $dataset->labels()->take($idxT, 0);
            
            $bootDataset = new Dataset($bootX, $bootY);

            // 3. Train the sub-tree
            $tree = new DecisionTreeClassifier($this->maxDepth, $this->minSamplesSplit, $maxFeatures);
            $tree->train($bootDataset);
            
            $this->trees[] = $tree;
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Random Forest is not trained.");
        }

        $n = $dataset->numRows();
        $treePreds = [];
        
        // Gather all predictions
        foreach ($this->trees as $tree) {
            $treePreds[] = $tree->predict($dataset)->toFlatArray();
        }

        $finalPreds = [];
        
        // JIT Optimized Voting Process
        for ($i = 0; $i < $n; $i++) {
            $votes = [];
            foreach ($treePreds as $preds) {
                $v = (int) $preds[$i];
                $votes[$v] = ($votes[$v] ?? 0) + 1;
            }
            
            // Sort votes descending and pick the highest
            arsort($votes);
            $finalPreds[] = array_key_first($votes);
        }

        return Tensor::fromArray($finalPreds);
    }

    public function trained(): bool
    {
        return !empty($this->trees);
    }
}