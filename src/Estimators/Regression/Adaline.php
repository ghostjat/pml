<?php
declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * ADALINE — Adaptive Linear Neuron (Widrow-Hoff LMS rule).
 * A single linear unit trained with batch gradient descent on MSE loss.
 *
 * JIT & Memory Optimized:
 * - All arithmetic is pure in-place BLAS (matmul + scalar ops).
 * - No intermediate arrays cross the FFI boundary during training.
 */
final class Adaline implements Learner
{
    private ?Tensor $weights = null;
    private float   $bias    = 0.0;

    public function __construct(
        private readonly int   $epochs       = 100,
        private readonly float $learningRate = 0.01,
        private readonly int   $batchSize    = 32
    ) {}

    public function train(Dataset $dataset): void
    {
        $d = $dataset->numColumns();
        $this->weights = Tensor::randomNormal([$d, 1], 0.0, 0.001);
        $this->bias    = 0.0;

        for ($e = 0; $e < $this->epochs; $e++) {
            $dataset->randomize();

            foreach ($dataset->batches($this->batchSize) as $batch) {
                $x   = $batch->samples();
                $y   = $batch->labels();
                $y   = $y->ndim() === 1 ? $y->expandDims(1) : $y;
                $n   = (float) $x->shape()[0];

                // Net input: Z = X*W + b
                $z   = $x->matmul($this->weights)->addScalarInplace($this->bias);

                // Error: dZ = Z - Y  (MSE gradient)
                $dz  = $z->sub($y);

                // Weight gradient: dW = X^T * dZ / N
                $dw  = $x->transpose()->matmul($dz)->mulScalarInplace(1.0 / $n);
                $db  = $dz->mean();

                $this->weights->subInplace($dw->mulScalarInplace($this->learningRate));
                $this->bias -= $db * $this->learningRate;
            }
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Adaline is not trained.");
        }
        return $dataset->samples()->matmul($this->weights)->addScalarInplace($this->bias)->squeeze();
    }

    public function trained(): bool
    {
        return $this->weights !== null;
    }
}
