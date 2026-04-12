<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Bernoulli Naive Bayes.
 * Optimized for classification with multivariate Bernoulli distributions (binary/boolean features).
 */
final class BernoulliNB implements Learner, Probabilistic
{
    private float $alpha;
    
    private array $classes = [];
    private array $classPriors = [];
    private array $featureLogProbs = [];
    private array $featureNegLogProbs = [];

    public function __construct(float $alpha = 1.0)
    {
        $this->alpha = $alpha;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $y = $dataset->labels();

        $n = (float) $x->shape()[0];
        $this->classes = $y->unique()->sort(0)->toFlatArray();

        foreach ($this->classes as $c) {
            $classKey = (string) $c;
            
            $cVal = Tensor::zeros(1)->addScalarInplace((float) $c);
            $mask = $y->equal($cVal);
            $classCount = $mask->sum();

            if ($classCount < 1.0) continue;

            $this->classPriors[$classKey] = log($classCount / $n);

            // Binarize input (X > 0.0) -> 1.0
            $zeroT = Tensor::zeros(1);
            $xBin = $x->greater($zeroT);
            
            // Count occurrences of 1s per feature
            $maskExpanded = $mask->expandDims(1);
            $featureCounts = $xBin->mul($maskExpanded)->sumAxis(0);
            
            // Laplace smoothing
            $smoothedCounts = $featureCounts->addScalarInplace($this->alpha);
            $smoothedTotal = $classCount + ($this->alpha * 2.0);
            
            $prob = $smoothedCounts->mulScalarInplace(1.0 / $smoothedTotal);

            $this->featureLogProbs[$classKey] = $prob->copy()->log();
            // 1 - prob, computed in-place on a same-shape copy to avoid broadcast issues
            $this->featureNegLogProbs[$classKey] = $prob->copy()->mulScalar(-1.0)->addScalarInplace(1.0)->log();
        }
    }

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) throw new RuntimeException("BernoulliNB is not trained.");

        $zeroT = Tensor::zeros(1);
        $xBin = $dataset->samples()->greater($zeroT);
        // 1 - xBin, computed in-place on a copy to avoid scalar-broadcast shape mismatch
        $notXBin = $xBin->copy()->mulScalar(-1.0)->addScalarInplace(1.0);
        
        $logProbs = [];

        foreach ($this->classes as $c) {
            $classKey = (string) $c;
            
            // log(P) = X * log(p) + (1-X) * log(1-p) + log(prior)
            $term1 = $xBin->matmul($this->featureLogProbs[$classKey]->expandDims(1));
            $term2 = $notXBin->matmul($this->featureNegLogProbs[$classKey]->expandDims(1));
            
            $logProb = $term1->addInplace($term2)->addScalarInplace($this->classPriors[$classKey]);
            $logProbs[] = $logProb;
        }

        return Tensor::concat($logProbs, 1);
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