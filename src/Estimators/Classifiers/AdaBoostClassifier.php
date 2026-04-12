<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Estimators\Classifiers\DecisionTreeClassifier;
use RuntimeException;

/**
 * AdaBoost (Adaptive Boosting) Classifier.
 * Sequentially trains "Decision Stumps" (Depth 1), assigning higher sample weights to misclassified points.
 * * JIT & Memory Optimized:
 * - Executes SAMME algorithm updates natively via OpenBLAS broadcasting.
 * - Extracts `weightedSample` subsets via an O(log N) binary search routing to `tensor_take()`.
 */
final class AdaBoostClassifier implements Learner
{
    private int $nEstimators;
    private float $learningRate;

    /** @var DecisionTreeClassifier[] */
    private array $estimators = [];
    /** @var float[] */
    private array $alphas = [];
    
    private array $classes = [];

    public function __construct(int $nEstimators = 50, float $learningRate = 1.0)
    {
        $this->nEstimators = $nEstimators;
        $this->learningRate = $learningRate;
    }

    public function train(Dataset $dataset): void
    {
        $n = $dataset->numRows();
        $y = $dataset->labels();
        
        $this->classes = $y->unique()->sort(0)->toFlatArray();
        $k = count($this->classes);
        
        if ($k < 2) {
            throw new \InvalidArgumentException("AdaBoost requires at least 2 distinct classes.");
        }

        // Initialize uniform sample weights: W_i = 1 / N
        $w = Tensor::ones($n)->mulScalarInplace(1.0 / $n);

        for ($i = 0; $i < $this->nEstimators; $i++) {
            
            // 1. Resample Dataset based on current Weights
            $wFlat = $w->toFlatArray();
            $indices = $this->weightedSample($wFlat, $n);
            
            $idxT = Tensor::fromArray($indices);
            $subDataset = new Dataset(
                $dataset->samples()->take($idxT, 0),
                $y->take($idxT, 0)
            );

            // 2. Train a Decision Stump
            $stump = new DecisionTreeClassifier(maxDepth: 1);
            $stump->train($subDataset);

            // 3. Evaluate Stump on the full original dataset
            $preds = $stump->predict($dataset);
            $incorrectMask = $preds->notEqual($y);
            
            // Weighted Error: sum(W_i * Incorrect_i)
            $error = $w->mul($incorrectMask)->sum();

            if ($error <= 0.0) {
                // Perfect estimator, assign maximum weight and stop
                $this->estimators[] = $stump;
                $this->alphas[] = 1.0;
                break;
            }

            if ($error >= 1.0 - (1.0 / $k)) {
                // Estimator is worse than random guessing, abort
                break;
            }

            // 4. Calculate Estimator Weight (SAMME Algorithm)
            $alpha = $this->learningRate * (log((1.0 - $error) / $error) + log($k - 1));
            
            $this->estimators[] = $stump;
            $this->alphas[] = $alpha;

            // 5. Update Sample Weights natively in C
            // incorrectMask = 1.0 -> exp(alpha), incorrectMask = 0.0 -> exp(0) = 1.0
            $modifier = $incorrectMask->mulScalar($alpha)->exp();
            $w->mulInplace($modifier);
            
            // Normalize weights
            $w->mulScalarInplace(1.0 / $w->sum());
        }
    }

    /**
     * O(N log N) Fast Weighted Sampling using a Cumulative Sum and Binary Search.
     */
    private function weightedSample(array $weights, int $n): array
    {
        $cumsum = [];
        $sum = 0.0;
        foreach ($weights as $w) {
            $sum += $w;
            $cumsum[] = $sum;
        }
        
        $indices = [];
        $maxIdx = count($cumsum) - 1;
        
        for ($i = 0; $i < $n; $i++) {
            $r = lcg_value() * $sum;
            
            $low = 0; 
            $high = $maxIdx;
            
            while ($low < $high) {
                $mid = (int) (($low + $high) / 2);
                if ($r > $cumsum[$mid]) {
                    $low = $mid + 1;
                } else {
                    $high = $mid;
                }
            }
            $indices[] = $low;
        }
        
        return $indices;
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("AdaBoost is not trained.");
        }

        $n = $dataset->numRows();
        $classScores = array_fill(0, count($this->classes), array_fill(0, $n, 0.0));

        // Aggregate alpha-weighted votes across all estimators
        foreach ($this->estimators as $i => $estimator) {
            $preds = $estimator->predict($dataset)->toFlatArray();
            $alpha = $this->alphas[$i];

            for ($row = 0; $row < $n; $row++) {
                $predictedClass = (float) $preds[$row];

                // Map the original class value to its sequential index [0...K]
                $classIdx = array_search($predictedClass, $this->classes);
                if ($classIdx !== false) {
                    $classScores[$classIdx][$row] += $alpha;
                }
            }
        }

        // Argmax across the weighted scores
        $finalPreds = [];
        for ($row = 0; $row < $n; $row++) {
            $bestScore = -INF;
            $bestClass = 0;
            
            foreach ($this->classes as $classIdx => $classVal) {
                if ($classScores[$classIdx][$row] > $bestScore) {
                    $bestScore = $classScores[$classIdx][$row];
                    $bestClass = $classVal;
                }
            }
            $finalPreds[] = $bestClass;
        }

        return Tensor::fromArray($finalPreds);
    }

    public function trained(): bool
    {
        return !empty($this->estimators);
    }
}