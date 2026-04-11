<?php

declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Lasso Regression (Linear Regression with L1 Regularization).
 * Encourages sparsity by forcing less important feature weights exactly to 0.0.
 * * JIT & Memory Optimized:
 * - Executes Gradient Descent rapidly using hardware BLAS.
 * - Extracts absolute L1 Penalty signs natively in C without loops.
 */
final class Lasso implements Learner
{
    private float $alpha;
    private int $epochs;
    private float $learningRate;
    private int $batchSize;
    
    private ?Tensor $weights = null;
    private float $bias = 0.0;

    /**
     * @param float $alpha The L1 penalty multiplier.
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
                
                // dW = (X^T * dZ) / N + (alpha * sign(W))
                $dw = $x->transpose()->matmul($dz)->mulScalarInplace(1.0 / $n);
                
                // L1 Penalty calculation: sign(W) * alpha
                $l1Penalty = $this->weights->sign()->mulScalarInplace($this->alpha);
                $dw->addInplace($l1Penalty);
                
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
            throw new RuntimeException("Lasso Regression has not been trained.");
        }
        return $dataset->samples()->matmul($this->weights)->addScalarInplace($this->bias)->squeeze();
    }

    public function trained(): bool
    {
        return $this->weights !== null;
    }
}