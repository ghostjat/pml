<?php

declare(strict_types=1);

namespace Pml\Estimators\Meta;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;
use ReflectionClass;

/**
 * Random Search Meta-Estimator.
 * Randomly samples combinations from a hyperparameter space to find optimal model configurations exponentially faster than exhaustive Grid Search.
 */
final class RandomSearch implements Learner
{
    private string $estimatorClass;
    private array $params;
    private int $nIter;
    
    private ?Learner $bestEstimator = null;
    private array $bestParams = [];

    public function __construct(string $estimatorClass, array $params, int $nIter = 10)
    {
        $this->estimatorClass = $estimatorClass;
        $this->params = $params;
        $this->nIter = $nIter;
    }

    public function train(Dataset $dataset): void
    {
        $bestScore = -INF;
        $reflector = new ReflectionClass($this->estimatorClass);

        for ($i = 0; $i < $this->nIter; $i++) {
            // 1. Randomly sample one configuration from the parameter space
            $sampledParams = [];
            foreach ($this->params as $key => $values) {
                $sampledParams[$key] = $values[array_rand($values)];
            }

            // 2. Instantiate and Train
            /** @var Learner $estimator */
            $estimator = $reflector->newInstanceArgs($sampledParams);
            
            // Simulate 80/20 train/validation split
            $validationRows = max(1, (int) ($dataset->numRows() * 0.2));
            $valSet = $dataset->slice(0, $validationRows);
            $trainSet = $dataset->slice($validationRows, $dataset->numRows() - $validationRows);
            
            $estimator->train($trainSet);
            $preds = $estimator->predict($valSet);

            // 3. Simple Accuracy/R2 Scoring logic
            $score = $preds->equal($valSet->labels())->mean();

            if ($score > $bestScore) {
                $bestScore = $score;
                $this->bestEstimator = $estimator;
                $this->bestParams = $sampledParams;
            }
        }
        
        // Retrain the absolute best configuration on the entire dataset
        $this->bestEstimator->train($dataset);
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) throw new RuntimeException("RandomSearch is not trained.");
        return $this->bestEstimator->predict($dataset);
    }

    public function trained(): bool
    {
        return $this->bestEstimator !== null;
    }
}