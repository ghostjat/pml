<?php

declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
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
final class VotingRegressor implements Learner
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

    public function train(Dataset $dataset): void
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
}