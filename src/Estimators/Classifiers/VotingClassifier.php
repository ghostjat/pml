<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;
use InvalidArgumentException;

/**
 * Voting Classifier (Heterogeneous Ensemble).
 * Combines completely different estimators (e.g., a Neural Network, SVM, and Random Forest)
 * and aggregates their predictions via majority vote.
 * * JIT & Memory Optimized:
 * - Aggregates votes natively in the PHP JIT cache to avoid creating massive string-based arrays.
 */
final class VotingClassifier implements Learner
{
    /** @var Learner[] */
    private array $estimators;

    /**
     * @param Learner[] $estimators An array of untrained estimators to ensemble.
     */
    public function __construct(array $estimators)
    {
        if (empty($estimators)) {
            throw new InvalidArgumentException("VotingClassifier requires at least one estimator.");
        }
        $this->estimators = $estimators;
    }

    public function train(Dataset $dataset): void
    {
        foreach ($this->estimators as $estimator) {
            $estimator->train($dataset);
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("VotingClassifier is not trained.");
        }

        $n = $dataset->numRows();
        $allPreds = [];
        
        // Gather 1D flat predictions from every heterogeneous estimator
        foreach ($this->estimators as $estimator) {
            $allPreds[] = $estimator->predict($dataset)->toFlatArray();
        }

        $finalPreds = [];
        
        // JIT Optimized Majority Voting
        for ($i = 0; $i < $n; $i++) {
            $votes = [];
            foreach ($allPreds as $preds) {
                $v = (string) $preds[$i]; // Cast to string to safely handle float/int classes as hash keys
                $votes[$v] = ($votes[$v] ?? 0) + 1;
            }
            
            arsort($votes);
            $finalPreds[] = (float) array_key_first($votes);
        }

        return Tensor::fromArray($finalPreds);
    }

    public function trained(): bool
    {
        foreach ($this->estimators as $estimator) {
            if (!$estimator->trained()) return false;
        }
        return true;
    }
}