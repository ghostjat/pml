<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
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
final class VotingClassifier implements Learner, Persistable
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

    public function train(Dataset $dataset, mixed ...$options): void
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

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        $manifest = [];
        foreach ($this->estimators as $idx => $estimator) {
            if (!($estimator instanceof Persistable)) {
                throw new RuntimeException("VotingClassifier::save() requires all estimators to implement Persistable.");
            }
            $subDir = $dir . '/estimator_' . $idx;
            $estimator->save($subDir);
            $manifest[] = get_class($estimator);
        }
        file_put_contents($dir . '/config.json', json_encode(['classes' => $manifest]));
    }

    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $estimators = [];
        foreach ($c['classes'] as $idx => $class) {
            $estimators[] = $class::load($dir . '/estimator_' . $idx);
        }
        return new self($estimators);
    }
}