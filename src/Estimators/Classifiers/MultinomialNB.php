<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use Pml\Interfaces\Persistable;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Multinomial Naive Bayes.
 * Optimized for discrete/count data (e.g., text classification with WordCountVectorizer).
 * * JIT & Memory Optimized:
 * - Uses OpenBLAS matrix accumulations to compute feature probabilities instantly.
 * - Applies Laplace (alpha) smoothing safely via inplace scalar addition.
 */
final class MultinomialNB implements Learner, Probabilistic, Persistable
{
    private float $alpha;
    
    private array $classes = [];
    private array $classPriors = [];
    private array $featureLogProbs = [];

    public function __construct(float $alpha = 1.0)
    {
        $this->alpha = $alpha;
    }

    public function train(Dataset $dataset, mixed ...$options): void
    {
        $x = $dataset->samples();
        $y = $dataset->labels();

        if ($y === null) throw new \InvalidArgumentException("MultinomialNB requires labeled data.");

        $n = (float) $x->shape()[0];
        $features = (float) $x->shape()[1];
        
        $this->classes = $y->unique()->sort(0)->toFlatArray();

        foreach ($this->classes as $c) {
            $classKey = (string) $c;
            
            // Mask instances of this class natively in C
            $cVal = Tensor::zeros(1)->addScalarInplace((float) $c);
            $mask = $y->equal($cVal);
            $classCount = $mask->sum();

            if ($classCount < 1.0) continue;

            $this->classPriors[$classKey] = log($classCount / $n);

            // Sum feature counts for this class
            $maskExpanded = $mask->expandDims(1);
            $classFeatureCounts = $x->mul($maskExpanded)->sumAxis(0);
            
            // Apply Laplace Smoothing: (N_yi + alpha)
            $smoothedCounts = $classFeatureCounts->addScalarInplace($this->alpha);
            
            // Denominator: N_y + alpha * n_features
            $smoothedTotal = $smoothedCounts->sum() + ($this->alpha * $features);
            
            // Log Probability: log( smoothedCounts / smoothedTotal )
            $this->featureLogProbs[$classKey] = $smoothedCounts->divInplace(
                Tensor::zeros(1)->addScalarInplace($smoothedTotal)
            )->log();
        }
    }

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) throw new RuntimeException("MultinomialNB is not trained.");

        $x = $dataset->samples();
        $logProbs = [];

        foreach ($this->classes as $c) {
            $classKey = (string) $c;
            $featureLogProb = $this->featureLogProbs[$classKey];
            $prior = $this->classPriors[$classKey];

            // log(P(x|y)) = sum(x_i * log(p_i)) + log(prior)
            // Handled as a fast Dot-Product (matmul) in OpenBLAS
            $logProb = $x->matmul($featureLogProb->expandDims(1))->addScalarInplace($prior);
            $logProbs[] = $logProb;
        }

        return Tensor::concat($logProbs, 1);
    }

    public function predict(Dataset $dataset): Tensor
    {
        return $this->proba($dataset)->argmaxAxis(1);
    }

    public function trained(): bool
    {
        return !empty($this->classes);
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['alpha' => $this->alpha, 'classes' => $this->classes, 'classPriors' => $this->classPriors]));
        if (!empty($this->featureLogProbs)) {
            $tensors = [];
            foreach ($this->featureLogProbs as $k => $t) { $tensors['flp.' . $k] = $t; }
            SafeTensorsIO::save($dir . '/model.safetensors', $tensors);
        }
    }

    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((float) $c['alpha']);
        $i->classes = $c['classes'] ?? [];
        $i->classPriors = $c['classPriors'] ?? [];
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) {
            $t = SafeTensorsIO::load($stPath);
            foreach ($t as $key => $tensor) {
                if (str_starts_with($key, 'flp.')) { $i->featureLogProbs[substr($key, 4)] = $tensor; }
            }
        }
        return $i;
    }
}