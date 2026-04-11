<?php

declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Ridge Regression (Linear Regression with L2 Regularization).
 * Prevents coefficient explosion by shrinking weights toward zero.
 * * JIT & Memory Optimized:
 * - Executes Gradient Descent rapidly using hardware BLAS.
 */
final class Ridge implements Learner
{
    private float $alpha;
    private int $epochs;
    private float $learningRate;
    private int $batchSize;
    
    private ?Tensor $weights = null;
    private float $bias = 0.0;

    /**
     * @param float $alpha The L2 penalty multiplier. Larger values shrink weights more aggressively.
     */
    public function __construct(float $alpha = 1.0, int $epochs = 100, float $learningRate = 0.01, int $batchSize = 32)
    {
        $this->alpha = $alpha;
        $this->epochs = $epochs;
        $this->learningRate = $learningRate;
        $this->batchSize = $batchSize;
    }

    public function train(Dataset $dataset): void
    {
        $features = $dataset->numColumns();
        $this->weights = Tensor::randomNormal([$features, 1], 0.0, 0.01);
        $this->bias = 0.0;

        for ($e = 0; $e < $this->epochs; $e++) {
            $dataset->randomize();

            foreach ($dataset->batches($this->batchSize) as $batch) {
                $x = $batch->samples();
                $y = $batch->labels();
                $y = $y->ndim() === 1 ? $y->expandDims(1) : $y;
                $n = (float) $x->shape()[0];

                // Y_pred = X * W + b
                $predictions = $x->matmul($this->weights)->addScalarInplace($this->bias);

                // dZ = Y_pred - Y
                $dz = $predictions->sub($y);
                
                // dW = (X^T * dZ) / N + (alpha * W)  <- Includes L2 Penalty
                $dw = $x->transpose()->matmul($dz)->mulScalarInplace(1.0 / $n);
                $l2Penalty = $this->weights->mulScalar($this->alpha);
                $dw->addInplace($l2Penalty);
                
                $db = $dz->mean();

                // Update In-Place
                $dw->mulScalarInplace($this->learningRate);
                $this->weights->subInplace($dw);
                $this->bias -= $db * $this->learningRate;
            }
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Ridge Regression has not been trained.");
        }
        return $dataset->samples()->matmul($this->weights)->addScalarInplace($this->bias)->squeeze();
    }

    public function trained(): bool
    {
        return $this->weights !== null;
    }
}