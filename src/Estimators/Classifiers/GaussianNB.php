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
    private float $epsilon = 1e-9;

    // Fused-kernel matrices (rebuilt after train() and load())
    private ?Tensor $meansMatrix  = null;   // [K, D]
    private ?Tensor $varsMatrix   = null;   // [K, D]
    private ?Tensor $logNormsVec  = null;   // [K]

    public function train(Dataset $dataset, mixed ...$options): void
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
        $this->buildMatrices();
    }

    private function buildMatrices(): void
    {
        if (empty($this->classes) || empty($this->means)) return;
        $meanRows = [];
        $varRows  = [];
        $logNorms = [];
        foreach ($this->classes as $c) {
            $ck = (string)$c;
            $meanRows[] = $this->means[$ck]->expandDims(0);      // [1, D]
            $varRows[]  = $this->variances[$ck]->expandDims(0);  // [1, D]
            // log_norm[k] = log_prior[k] − 0.5·Σ_d log(2π·var[k,d])
            $logNorms[] = $this->priors[$ck]
                        - 0.5 * $this->variances[$ck]->mulScalar(2.0 * M_PI)->log()->sum();
        }
        $this->meansMatrix = Tensor::concat($meanRows, 0);  // [K, D]
        $this->varsMatrix  = Tensor::concat($varRows,  0);  // [K, D]
        $this->logNormsVec = Tensor::fromArray($logNorms);  // [K]
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
        // Single fused C call: K×(sub+square+div+sum+addScalar) → [N, K]
        return Tensor::gnbLogLikelihood(
            $dataset->samples(),
            $this->meansMatrix,
            $this->varsMatrix,
            $this->logNormsVec
        );
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
        $i->buildMatrices();
        return $i;
    }
}