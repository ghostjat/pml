<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Categorical Naive Bayes.
 * Optimized for datasets where all features are categorical (e.g., label encoded).
 * * JIT & Memory Optimized:
 * - Employs C-level feature-value extraction and tensor masking to compute probabilities instantly.
 */
final class CategoricalNB implements Learner, Probabilistic
{
    private float $alpha;
    
    private array $classes = [];
    private array $classPriors = [];
    
    // Structure: [classKey][featureIndex][categoryValue] => log(prob)
    private array $featureLogProbs = [];

    public function __construct(float $alpha = 1.0)
    {
        $this->alpha = $alpha;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $y = $dataset->labels();

        if ($y === null) throw new \InvalidArgumentException("CategoricalNB requires labeled data.");

        $n = (float) $x->shape()[0];
        $numFeatures = $x->shape()[1];
        
        $this->classes = $y->unique()->sort(0)->toFlatArray();
        
        // Find the maximum category integer per feature to size our probability arrays
        $maxCategories = $x->maxAxis(0)->toFlatArray();

        foreach ($this->classes as $c) {
            $classKey = (string) $c;
            
            $cVal = Tensor::zeros(1)->addScalarInplace((float) $c);
            $classMask = $y->equal($cVal);
            $classCount = $classMask->sum();

            if ($classCount < 1.0) continue;

            $this->classPriors[$classKey] = log($classCount / $n);
            $this->featureLogProbs[$classKey] = [];

            // Extract the subset of features belonging only to this class
            $classX = $x->booleanIndex($classMask);

            for ($f = 0; $f < $numFeatures; $f++) {
                $featureCol = $classX->col($f);
                $numCats = (int) $maxCategories[$f] + 1;
                
                // C-Level bincount tallies the category occurrences instantly
                $catCounts = $featureCol->bincount();
                
                // Pad if not all categories were present in this class slice
                if ($catCounts->size() < $numCats) {
                    $catCounts = $catCounts->pad([0, $numCats - $catCounts->size()]);
                }
                
                // Laplace smoothing: (count + alpha) / (classCount + alpha * numCats)
                $smoothedCounts = $catCounts->addScalarInplace($this->alpha);
                $denominator = $classCount + ($this->alpha * $numCats);
                
                $logProbs = $smoothedCounts->divInplace(Tensor::zeros(1)->addScalarInplace($denominator))->log();
                
                $this->featureLogProbs[$classKey][$f] = $logProbs->toFlatArray();
            }
        }
    }

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) throw new RuntimeException("CategoricalNB is not trained.");

        $x = $dataset->samples()->toFlatArray();
        $rows = $dataset->numRows();
        $cols = $dataset->numColumns();
        
        $logProbs = [];

        // JIT Optimized lookup loop
        for ($i = 0; $i < $rows; $i++) {
            $rowOffset = $i * $cols;
            $rowProbs = [];
            
            foreach ($this->classes as $c) {
                $classKey = (string) $c;
                $logProb = $this->classPriors[$classKey];
                
                for ($f = 0; $f < $cols; $f++) {
                    $catVal = (int) $x[$rowOffset + $f];
                    
                    // Fallback to a very low probability if unseen category
                    $logProb += $this->featureLogProbs[$classKey][$f][$catVal] ?? log(1e-8);
                }
                $rowProbs[] = $logProb;
            }
            $logProbs[] = $rowProbs;
        }

        return Tensor::fromArray($logProbs);
    }

    public function predict(Dataset $dataset): Tensor
    {
        return $this->proba($dataset)->argmax();
    }

    public function trained(): bool
    {
        return !empty($this->classes);
    }
}