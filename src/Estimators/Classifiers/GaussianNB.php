<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Gaussian Naive Bayes Classifier.
 * A probabilistic model that assumes features are independent and normally distributed.
 * * JIT & Memory Optimized:
 * - Employs C-Level broadcasting masks instead of row-filtering to compute Mean & Variance instantly.
 * - Inference computes log-likelihoods concurrently using AVX2 vectorized math.
 */
final class GaussianNB implements Learner, Probabilistic
{
    private array $priors = [];
    private array $means = [];
    private array $variances = [];
    private array $classes = [];
    
    // Smoothing factor to prevent division by zero in variance calculations
    private float $epsilon = 1e-9;

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $y = $dataset->labels();

        if ($y === null) {
            throw new \InvalidArgumentException("GaussianNB requires labeled classification data.");
        }

        $n = $x->shape()[0];
        $this->classes = $y->unique()->sort(0)->toFlatArray();

        foreach ($this->classes as $c) {
            $classKey = (string) $c;
            
            // 1. Generate Boolean Mask for the specific class
            $cVal = Tensor::zeros($n)->addScalarInplace((float) $c);
            $mask = $y->equal($cVal);
            $count = $mask->sum();

            if ($count < 1.0) continue;

            $this->priors[$classKey] = log($count / $n);

            // 2. Expand mask to [N, 1] to zero out non-class rows across all features
            $maskExpanded = $mask->expandDims(1);
            $maskedX = $x->mul($maskExpanded);

            // 3. Compute the Class Mean natively in C
            $mean = $maskedX->sumAxis(0)->mulScalarInplace(1.0 / $count);
            
            // 4. Compute the Class Variance (E[X^2] - E[X]^2) natively in C
            $meanOfSquares = $maskedX->square()->sumAxis(0)->mulScalarInplace(1.0 / $count);
            $variance = $meanOfSquares->sub($mean->square())->addScalarInplace($this->epsilon);

            $this->means[$classKey] = $mean;
            $this->variances[$classKey] = $variance;
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        // proba() returns [N, K] log-probabilities; pick the class with the highest score per row.
        $logProbs = $this->proba($dataset)->toFlatArray();
        $n = $dataset->numRows();
        $k = \count($this->classes);
        $preds = [];
        for ($i = 0; $i < $n; $i++) {
            $row   = \array_slice($logProbs, $i * $k, $k);
            $preds[] = (float) \array_search(\max($row), $row);
        }
        return Tensor::fromArray($preds);
    }

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("GaussianNB has not been trained.");
        }

        $x = $dataset->samples();
        $logProbs = [];

        // Compute the log probability for each class across all inference rows simultaneously
        foreach ($this->classes as $c) {
            $classKey = (string) $c;
            $mean = $this->means[$classKey];
            $var = $this->variances[$classKey];
            $prior = $this->priors[$classKey];

            // Gaussian Log-Likelihood Formula:
            // -0.5 * sum(log(2 * pi * var)) - 0.5 * sum((x - mean)^2 / var) + log(prior)
            
            // 1. -0.5 * sum((x - mean)^2 / var, axis=1)
            $diffSquared = $x->sub($mean)->square();
            $term1 = $diffSquared->divInplace($var)->sumAxis(1)->mulScalarInplace(-0.5);

            // 2. -0.5 * sum(log(2 * pi * var))
            // This is a scalar constant for the class, computed efficiently in C
            $term2Const = $var->mulScalar(2.0 * M_PI)->log()->sum() * -0.5;

            // Combine and add Prior
            $classLogProb = $term1->addScalarInplace($term2Const + $prior)->expandDims(1);
            $logProbs[] = $classLogProb;
        }

        // Concatenate class log-probabilities into a continuous [N, K] matrix
        return Tensor::concat($logProbs, 1);
    }

    public function trained(): bool
    {
        return !empty($this->classes);
    }
}