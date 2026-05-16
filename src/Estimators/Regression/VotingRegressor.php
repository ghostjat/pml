<?php

declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;
use InvalidArgumentException;

/**
 * Voting Regressor (Heterogeneous Ensemble).
 * Combines different regressors (e.g., SVR, Ridge, GradientBoosting) and averages their continuous predictions.
 * * JIT & Memory Optimized:
 * - 100% Zero-Copy C-level execution.
 * - Concatenates predictions into a [Batch, Estimators] matrix and reduces via `meanAxis(1)`.
 */
final class VotingRegressor implements Learner, Persistable
{
    /** @var Learner[] */
    private array $estimators;

    public function __construct(array $estimators)
    {
        if (empty($estimators)) {
            throw new InvalidArgumentException("VotingRegressor requires at least one estimator.");
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
            throw new RuntimeException("VotingRegressor is not trained.");
        }

        $predTensors = [];
        
        foreach ($this->estimators as $estimator) {
            // Expand [N] to [N, 1] for horizontal concatenation
            $predTensors[] = $estimator->predict($dataset)->expandDims(1);
        }

        // Hardware Concatenation: Shape [N, NumEstimators]
        $ensembleMatrix = Tensor::concat($predTensors, 1);

        // Calculate the mean across the estimators natively in OpenBLAS
        return $ensembleMatrix->meanAxis(1)->squeeze();
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
                throw new RuntimeException("VotingRegressor::save() requires all estimators to implement Persistable.");
            }
            $estimator->save($dir . '/estimator_' . $idx);
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