<?php

declare(strict_types=1);

namespace Pml\Estimators\Mainfold;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * t-Distributed Stochastic Neighbor Embedding (t-SNE).
 * Non-linear dimensionality reduction specifically designed for high-dimensional data visualization.
 * * JIT & Memory Optimized:
 * - Vectorizes O(N^2) pairwise affinities using highly-optimized OpenBLAS GEMM.
 * - Gradient Descent executes in pure C via analytical Matrix formulations: 4 * (diag(M1)Y - MY).
 * - Completely bypasses PHP loops during the complex Kullback-Leibler divergence minimization.
 */
final class TSNE implements Learner
{
    private int $nComponents;
    private int $maxIter;
    private float $learningRate;
    private float $momentum;
    
    private ?Tensor $embedding = null;

    /**
     * @param int $nComponents Dimension of the embedded space (usually 2 or 3 for visualization).
     * @param int $maxIter Maximum number of iterations for the optimization.
     * @param float $learningRate The learning rate for the gradient descent.
     * @param float $momentum The momentum to speed up convergence and escape local minima.
     */
    public function __construct(
        int $nComponents = 2,
        int $maxIter = 1000,
        float $learningRate = 200.0,
        float $momentum = 0.8
    ) {
        $this->nComponents = $nComponents;
        $this->maxIter = $maxIter;
        $this->learningRate = $learningRate;
        $this->momentum = $momentum;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $n = (float) $x->shape()[0];

        if ($n < 2.0) {
            throw new \InvalidArgumentException("t-SNE requires at least 2 samples.");
        }

        // 1. Compute Exact Pairwise Affinities (High Dimensional Space)
        // Vectorized Euclidean Distance: D = sum(X^2, 1) + sum(X^2, 1)^T - 2 * X * X^T
        $xSq = $x->square()->sumAxis(1)->expandDims(1);
        $xSqT = $xSq->transpose();
        
        $dist = $x->matmul($x->transpose())
                  ->mulScalarInplace(-2.0)
                  ->addInplace($xSq)
                  ->addInplace($xSqT)
                  ->clip(0.0, INF);

        // Dynamic global variance scaling for numerical stability
        $gamma = 1.0 / ($dist->mean() + 1e-8);
        
        // P = exp(-Dist * gamma)
        $p = $dist->mulScalar(-$gamma)->exp();
        
        // Zero out the diagonal (which is exactly 1.0 because exp(0) = 1) using AVX2 Boolean Masking
        $zero = Tensor::zeros(1);
        $diagonalMask = $dist->greater($zero);
        $p->mulInplace($diagonalMask);
        
        // Symmetrize the matrix
        $p->addInplace($p->transpose());
        
        // Normalize P to sum to 1.0 across all pairs
        $pSum = $p->sum();
        $p->mulScalarInplace(1.0 / $pSum);
        
        // Early Exaggeration: Multiply P by 4.0 to encourage tighter clustering initially
        $p->mulScalarInplace(4.0);

        // 2. Initialize Low-Dimensional Embeddings & Momentum Velocity
        $y = Tensor::randomNormal([(int)$n, $this->nComponents], 0.0, 1e-4);
        $yVelocity = Tensor::zeros((int)$n, $this->nComponents);

        // 3. Vectorized Gradient Descent Loop
        for ($iter = 0; $iter < $this->maxIter; $iter++) {
            
            // Stop Early Exaggeration after 250 iterations
            if ($iter === 250) {
                $p->mulScalarInplace(0.25);
            }

            // Compute Low-Dimensional Euclidean Distances
            $ySq = $y->square()->sumAxis(1)->expandDims(1);
            $ySqT = $ySq->transpose();
            
            $distY = $y->matmul($y->transpose())
                       ->mulScalarInplace(-2.0)
                       ->addInplace($ySq)
                       ->addInplace($ySqT)
                       ->clip(0.0, INF);

            // Compute Student-t distribution unnormalized affinities: Q_unnorm = (1 + DistY)^(-1)
            $qUnnorm = Tensor::ones((int)$n, (int)$n)->divInplace($distY->addScalar(1.0));
            
            // Mask out the diagonal natively in C
            $qUnnorm->mulInplace($diagonalMask);

            $qSum = $qUnnorm->sum() + 1e-8;
            $q = $qUnnorm->mulScalar(1.0 / $qSum);

            // Compute Analytical Gradient: dC/dY = 4 * (diag(M * 1) * Y - M * Y)
            // Where M = (P - Q) * Q_unnorm
            $m = $p->sub($q)->mulInplace($qUnnorm);
            
            $sumM = $m->sumAxis(1)->expandDims(1);
            
            // (diag(M * 1) * Y)  -> Element-wise broadcast multiplication
            $term1 = $y->mul($sumM);
            
            // (M * Y) -> Fast BLAS Matrix Multiplication
            $term2 = $m->matmul($y);
            
            $grad = $term1->subInplace($term2)->mulScalarInplace(4.0);

            // Update Embeddings with Momentum
            $yVelocity->mulScalarInplace($this->momentum)->subInplace($grad->mulScalarInplace($this->learningRate));
            $y->addInplace($yVelocity);
            
            // Re-center Y to prevent coordinate drift
            $yMean = $y->meanAxis(0);
            $y->subInplace($yMean);
        }

        $this->embedding = $y;
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("t-SNE has not been trained.");
        }
        
        // Standard t-SNE is a transductive learner; it cannot reliably embed new points post-training.
        // Returning the generated embedding serves as the transformation output for the training set.
        return $this->embedding;
    }

    public function trained(): bool
    {
        return $this->embedding !== null;
    }

    /**
     * Returns the finalized low-dimensional embeddings.
     */
    public function embedding(): ?Tensor
    {
        return $this->embedding;
    }
}