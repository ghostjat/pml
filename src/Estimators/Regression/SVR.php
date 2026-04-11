<?php

declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Support Vector Regressor (Linear SVR).
 * Learns a continuous regression function using an epsilon-insensitive tube.
 * * JIT & Memory Optimized:
 * - Employs vectorized epsilon-violation masking via AVX2 `abs()` and `greater()`.
 * - Subgradient updates are processed 100% In-Place natively in C.
 */
final class SVR implements Learner
{
    private float $c;
    private float $epsilon;
    private int $epochs;
    private float $learningRate;
    private int $batchSize;
    
    private ?Tensor $weights = null;
    private float $bias = 0.0;

    /**
     * @param float $c The penalty parameter for margin violations.
     * @param float $epsilon The width of the epsilon-tube where no penalty is associated with errors.
     */
    public function __construct(
        float $c = 1.0, 
        float $epsilon = 0.1, 
        int $epochs = 100, 
        float $learningRate = 0.01, 
        int $batchSize = 32
    ) {
        $this->c = $c;
        $this->epsilon = $epsilon;
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

                // Forward Pass: Z = X * W + b
                $z = $x->matmul($this->weights)->addScalarInplace($this->bias);
                
                // Diff: Z - Y
                $diff = $z->sub($y);
                
                // Find Epsilon-tube violations
                $epsilonT = Tensor::zeros(1)->addScalarInplace($this->epsilon);
                $violationMask = $diff->abs()->greater($epsilonT);

                // Subgradient: sign(diff) ONLY for violations
                $dZ = $diff->sign()->mulInplace($violationMask);

                // dW = W + C * (X^T * dZ / N)
                $dw = $x->transpose()->matmul($dZ)->mulScalarInplace($this->c / $n)->addInplace($this->weights);
                $db = $dZ->mean() * $this->c;

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
            throw new RuntimeException("SVR has not been trained.");
        }

        return $dataset->samples()->matmul($this->weights)->addScalarInplace($this->bias)->squeeze();
    }

    public function trained(): bool
    {
        return $this->weights !== null;
    }
}