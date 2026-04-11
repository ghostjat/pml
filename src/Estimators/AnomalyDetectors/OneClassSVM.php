<?php

declare(strict_types=1);

namespace Pml\Estimators\AnomalyDetectors;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Kernels\SVM\Kernel;
use Pml\Kernels\SVM\RBF;
use RuntimeException;

/**
 * One-Class Support Vector Machine.
 * Unsupervised outlier detection that learns a boundary maximizing the margin from the origin 
 * mapped into a high-dimensional kernel space.
 */
final class OneClassSVM implements Learner
{
    private float $nu;
    private Kernel $kernel;
    private int $epochs;
    private float $learningRate;
    
    private ?Tensor $weights = null; // Dual alpha coefficients
    private float $rho = 0.0;
    private ?Tensor $supportVectors = null;

    public function __construct(float $nu = 0.1, ?Kernel $kernel = null, int $epochs = 100, float $learningRate = 0.01)
    {
        $this->nu = $nu;
        $this->kernel = $kernel ?? new RBF(0.1);
        $this->epochs = $epochs;
        $this->learningRate = $learningRate;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $n = (float) $x->shape()[0];
        
        $this->supportVectors = $x;
        // Alphas bounded by [0, 1 / (nu * N)]
        $this->weights = Tensor::randomUniform([$x->shape()[0], 1], 0.0, 1.0 / ($this->nu * $n));
        $this->rho = 1.0;

        $kMatrix = $this->kernel->compute($x, $this->supportVectors);

        for ($e = 0; $e < $this->epochs; $e++) {
            // Z = K * alpha
            $z = $kMatrix->matmul($this->weights);
            
            // Subgradient mask where (Z < rho)
            $rhoT = Tensor::zeros(1)->addScalarInplace($this->rho);
            $violationMask = $z->less($rhoT);

            // Update Alphas
            $kMatrixT = $kMatrix->transpose();
            $dZ = $violationMask->mulScalar(-1.0);
            
            // dAlpha = Alpha + (1 / (nu * N)) * K^T * dZ
            $dw = $kMatrixT->matmul($dZ)->mulScalarInplace(1.0 / ($this->nu * $n))->addInplace($this->weights);
            
            $this->weights->subInplace($dw->mulScalarInplace($this->learningRate));
            
            // Bound alphas: 0 <= alpha <= 1 / (nu * N)
            $this->weights = $this->weights->clip(0.0, 1.0 / ($this->nu * $n));
            
            // Update Rho: rho -= lr * (-1 + sum(violationMask) / (nu * N))
            $dRho = -1.0 + ($violationMask->sum() / ($this->nu * $n));
            $this->rho -= $this->learningRate * $dRho;
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) throw new RuntimeException("OneClassSVM is not trained.");

        $kTest = $this->kernel->compute($dataset->samples(), $this->supportVectors);
        $z = $kTest->matmul($this->weights);
        
        // Anomalies are instances where Z < rho
        $rhoT = Tensor::zeros(1)->addScalarInplace($this->rho);
        return $z->less($rhoT);
    }

    public function trained(): bool
    {
        return $this->weights !== null;
    }
}