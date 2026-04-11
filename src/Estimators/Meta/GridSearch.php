<?php

declare(strict_types=1);

namespace Pml\Estimators\Meta;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Verbose;
use Pml\Metrics\Metric;
use Pml\Tensor;
use Pml\Dataset;
use Psr\Log\LoggerInterface;
use RuntimeException;
use ReflectionClass;

/**
 * Grid Search Meta-Estimator.
 * Exhaustively searches a hyperparameter grid to find the best performing model setup.
 * * JIT & Memory Optimized:
 * - Employs zero-copy K-Fold cross validation natively via Dataset->fold().
 * - Aggressively isolates scopes to ensure discarded C-tensors are instantly garbage collected.
 */
final class GridSearch implements Learner, Verbose
{
    private Learner $baseEstimator;
    private array $grid;
    private Metric $metric;
    private int $cvFolds;
    
    private ?Learner $bestEstimator = null;
    private array $bestParams = [];
    private float $bestScore = -INF;

    private ?LoggerInterface $logger = null;

    /**
     * @param Learner $baseEstimator An untrained instance of the estimator to tune.
     * @param array $grid Associative array of parameters (e.g., ['maxDepth' => [5, 10, 15]])
     * @param Metric $metric The evaluation metric to maximize (e.g., new Accuracy())
     * @param int $cvFolds Number of folds for Cross Validation (default: 5)
     */
    public function __construct(Learner $baseEstimator, array $grid, Metric $metric, int $cvFolds = 5)
    {
        if ($cvFolds < 2) {
            throw new \InvalidArgumentException("Cross Validation requires at least 2 folds.");
        }

        $this->baseEstimator = $baseEstimator;
        $this->grid = $grid;
        $this->metric = $metric;
        $this->cvFolds = $cvFolds;
    }

    public function setLogger(LoggerInterface $logger): void
    {
        $this->logger = $logger;
    }

    public function train(Dataset $dataset): void
    {
        $combinations = $this->generateCombinations($this->grid);
        $totalCombos = count($combinations);

        if ($this->logger) {
            $this->logger->info(sprintf("Starting Grid Search: Evaluating %d combinations with %d-Fold CV.", $totalCombos, $this->cvFolds));
        }

        foreach ($combinations as $i => $params) {
            $scoreSum = 0.0;
            $foldsEvaluated = 0;

            // Zero-Copy Cross Validation Loop
            foreach ($dataset->fold($this->cvFolds) as [$trainFold, $valFold]) {
                
                // 1. Clone a fresh, untrained estimator instance
                $model = clone $this->baseEstimator;
                $this->applyParams($model, $params);
                
                // 2. Train on the Train Slice
                $model->train($trainFold);
                
                // 3. Evaluate on the Validation Slice
                $predictions = $model->predict($valFold);
                $scoreSum += $this->metric->score($predictions, $valFold->labels());
                $foldsEvaluated++;
                
                // The $model falls out of scope here! 
                // Any FFI Tensors inside it trigger __destruct() and instantly free their C memory.
            }

            $avgScore = $scoreSum / $foldsEvaluated;

            if ($this->logger) {
                $paramStr = json_encode($params);
                $this->logger->info(sprintf("[%d/%d] Score: %.6f | Params: %s", $i + 1, $totalCombos, $avgScore, $paramStr));
            }

            // Track the Best Model
            if ($avgScore > $this->bestScore) {
                $this->bestScore = $avgScore;
                $this->bestParams = $params;
            }
        }

        if ($this->logger) {
            $this->logger->info(sprintf("Grid Search Complete. Best Score: %.6f | Best Params: %s", $this->bestScore, json_encode($this->bestParams)));
            $this->logger->info("Retraining best estimator on the entire dataset...");
        }

        // Final Phase: Train the winning model configuration on the ENTIRE dataset
        $this->bestEstimator = clone $this->baseEstimator;
        $this->applyParams($this->bestEstimator, $this->bestParams);
        $this->bestEstimator->train($dataset);
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("GridSearch has not been trained.");
        }

        return $this->bestEstimator->predict($dataset);
    }

    public function trained(): bool
    {
        return $this->bestEstimator !== null && $this->bestEstimator->trained();
    }

    /**
     * Returns the best parameters discovered during training.
     */
    public function bestParams(): array
    {
        return $this->bestParams;
    }

    // ========================================================================
    // INTERNAL HELPERS
    // ========================================================================

    /**
     * Generates a Cartesian product of all hyperparameter arrays.
     */
    private function generateCombinations(array $grid): array
    {
        $keys = array_keys($grid);
        $values = array_values($grid);
        $combinations = [[]];

        for ($i = 0; $i < count($keys); $i++) {
            $tmp = [];
            foreach ($combinations as $v1) {
                foreach ($values[$i] as $v2) {
                    $tmp[] = array_merge($v1, [$keys[$i] => $v2]);
                }
            }
            $combinations = $tmp;
        }

        return $combinations;
    }

    /**
     * Uses Reflection to inject hyperparameters directly into private/protected properties.
     * Safely bypasses constructors to reset the estimator topology exactly as required.
     */
    private function applyParams(Learner $estimator, array $params): void
    {
        $refClass = new ReflectionClass($estimator);
        
        foreach ($params as $property => $value) {
            if ($refClass->hasProperty($property)) {
                $prop = $refClass->getProperty($property);
                $prop->setAccessible(true);
                $prop->setValue($estimator, $value);
            } else {
                throw new \InvalidArgumentException(sprintf("Estimator %s does not have property '%s'", $refClass->getShortName(), $property));
            }
        }
    }
}