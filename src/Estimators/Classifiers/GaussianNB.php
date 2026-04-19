<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use Pml\Interfaces\Persistable;
use Pml\Interfaces\Saveable;
use Pml\Interfaces\Stateful;
use Pml\Lib\SafeTensorsIO;
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
final class GaussianNB implements Learner, Probabilistic, Persistable, Saveable, Stateful
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
        // argmaxAxis(1) finds the class index with max log-prob per row — single C call
        return $this->proba($dataset)->argmaxAxis(1);
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

    // ── Saveable ─────────────────────────────────────────────────────────────

    public function getConfig(): array
    {
        return ['epsilon' => $this->epsilon];
    }

    public static function fromConfig(array $config): static
    {
        $instance          = new static();
        $instance->epsilon = (float) ($config['epsilon'] ?? 1e-9);
        return $instance;
    }

    public function getPhpState(): array
    {
        return [
            'classes' => $this->classes,
            'priors'  => $this->priors,
        ];
    }

    public function setPhpState(array $state): void
    {
        $this->classes = $state['classes'] ?? [];
        $this->priors  = $state['priors']  ?? [];
    }

    // ── Stateful ─────────────────────────────────────────────────────────────
    // means[classKey] and variances[classKey] are array<string,Tensor> — invisible
    // to Reflection. Flattened with a dot-prefix so SafeTensors keys are unique.

    public function getStateDict(string $prefix = ''): array
    {
        $dict = [];
        foreach ($this->means     as $k => $t) { $dict["{$prefix}means.{$k}"]     = $t; }
        foreach ($this->variances as $k => $t) { $dict["{$prefix}variances.{$k}"] = $t; }
        return $dict;
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->means     = [];
        $this->variances = [];
        $mPfx = "{$prefix}means.";
        $vPfx = "{$prefix}variances.";

        foreach ($dict as $key => $tensor) {
            if (str_starts_with($key, $mPfx)) {
                $this->means[substr($key, strlen($mPfx))] = $tensor;
            } elseif (str_starts_with($key, $vPfx)) {
                $this->variances[substr($key, strlen($vPfx))] = $tensor;
            }
        }
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['epsilon' => $this->epsilon, 'classes' => $this->classes, 'priors' => $this->priors]));
        if (!empty($this->means)) {
            SafeTensorsIO::save($dir . '/model.safetensors', $this->getStateDict());
        }
    }

    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self();
        $i->epsilon = (float) ($c['epsilon'] ?? 1e-9);
        $i->classes = $c['classes'] ?? [];
        $i->priors  = $c['priors']  ?? [];
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) {
            $i->loadStateDict(SafeTensorsIO::load($stPath));
        }
        return $i;
    }
}