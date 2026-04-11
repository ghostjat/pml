<?php

declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Ordinary Least Squares (OLS) Linear Regression.
 * Solves exactly in one step using the LAPACKE Moore-Penrose Pseudo-Inverse.
 * * JIT & Memory Optimized:
 * - 100% Closed-form C execution.
 * - Zero PHP iteration overhead.
 */
final class LinearRegression implements Learner
{
    private ?Tensor $weights = null;

    /**
     * Train the model using the closed-form equation: W = X^+ * Y
     * (Where X^+ is the Pseudo-Inverse of X).
     */
    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $y = $dataset->labels();

        if ($y === null) {
            throw new \InvalidArgumentException("Linear Regression requires a labeled dataset.");
        }

        // Ensure Y is a 2D column vector [N, 1] for matrix multiplication
        $yCol = $y->ndim() === 1 ? $y->expandDims(1) : $y;

        // X^+ * Y
        // pinv() securely handles multicollinearity by using SVD under the hood.
        $this->weights = $x->pinv()->matmul($yCol);
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new \RuntimeException("Estimator has not been trained.");
        }

        // Y_pred = X * W
        // Returns the raw predictions as a C-Tensor, squeezed to 1D
        return $dataset->samples()->matmul($this->weights)->squeeze();
    }

    public function trained(): bool
    {
        return $this->weights !== null;
    }
}