<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Estimators\Trees\ExtraTreeClassifier;
use RuntimeException;

/**
 * Extremely Randomized Trees (ExtraTrees).
 * * JIT & Memory Optimized:
 * - Trades exhaustive optimization for extreme speed and higher variance.
 * - Bypasses bootstrap sampling to feed the whole dataset instantly to each weak learner.
 */
final class ExtraTreesClassifier implements Learner
{
    private int $nEstimators;
    private int $maxDepth;
    private int $minSamplesSplit;
    
    /** @var ExtraTreeClassifier[] */
    private array $trees = [];

    public function __construct(int $nEstimators = 100, int $maxDepth = 10, int $minSamplesSplit = 2)
    {
        $this->nEstimators = $nEstimators;
        $this->maxDepth = $maxDepth;
        $this->minSamplesSplit = $minSamplesSplit;
    }

    public function train(Dataset $dataset): void
    {
        $features = $dataset->numColumns();
        $maxFeatures = (int) max(1, sqrt($features));

        for ($i = 0; $i < $this->nEstimators; $i++) {
            // Note: ExtraTrees uses the whole dataset by default (no bootstrapping)
            // It relies purely on the extreme randomization of the node splits
            $tree = new ExtraTreeClassifier($this->maxDepth, $this->minSamplesSplit, $maxFeatures);
            $tree->train($dataset);
            
            $this->trees[] = $tree;
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("ExtraTrees Ensemble is not trained.");
        }

        $n = $dataset->numRows();
        $treePreds = [];
        
        foreach ($this->trees as $tree) {
            $treePreds[] = $tree->predict($dataset)->toFlatArray();
        }

        $finalPreds = [];
        
        // JIT Optimized Majority Voting
        for ($i = 0; $i < $n; $i++) {
            $votes = [];
            foreach ($treePreds as $preds) {
                $v = (int) $preds[$i];
                $votes[$v] = ($votes[$v] ?? 0) + 1;
            }
            
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