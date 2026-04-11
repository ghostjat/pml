<?php

declare(strict_types=1);

namespace Pml\Losses;

use Pml\Tensor;

/**
 * Binary Cross Entropy (Log Loss).
 * Standard loss function for binary classification and Sigmoid activations.
 * * JIT & Memory Optimized:
 * - Leverages AVX2 SIMD via `log1p` for numerical stability and speed.
 * - Maximizes `*Inplace` C-mutations to prevent PHP heap fragmentation.
 * - Operates entirely via Zero-Copy C-Pointers.
 */
final class BinaryCrossEntropy implements Loss
{
    public function compute(Tensor $predictions, Tensor $labels): float
    {
        // 1. Clip predictions to prevent log(0) explosions (-INF)
        $clipped = $predictions->clip(1e-7, 1.0 - 1e-7);
        
        // 2. Term 1: y * log(y_pred)
        $term1 = $labels->mul($clipped->log());
        
        // 3. Term 2: (1 - y) * log(1 - y_pred)
        // OPTIMIZATION: log(1 - x) is mathematically identical to log1p(-x).
        // This saves an entire AVX2 array traversal (addScalarInplace) and improves precision.
        $minusClipped = $clipped->mulScalar(-1.0);
        $logOneMinusPred = $minusClipped->log1p();
        
        // (1 - y)
        $oneMinusY = $labels->mulScalar(-1.0)->addScalarInplace(1.0);
        
        // Combine into Term 2 natively in C-memory
        $term2 = $oneMinusY->mulInplace($logOneMinusPred);
        
        // 4. Loss = -mean(term1 + term2)
        // All aggregations happen in the CPU cache without allocating new Tensors
        return $term1->addInplace($term2)->mean() * -1.0;
    }

    public function differentiate(Tensor $predictions, Tensor $labels): Tensor
    {
        $clipped = $predictions->clip(1e-7, 1.0 - 1e-7);
        $n = $predictions->size();
        
        // Formula: dY = (y_pred - y_true) / (y_pred * (1 - y_pred) * N)
        
        // 1. Calculate Denominator: [y_pred * (1 - y_pred) * N]
        $oneMinusYPred = $clipped->mulScalar(-1.0)->addScalarInplace(1.0);
        $denominator = $clipped->mul($oneMinusYPred)->mulScalarInplace((float) $n);
        
        // 2. Calculate Numerator & Final Gradient In-Place: [y_pred - y_true] / Denominator
        $diff = $clipped->sub($labels);
        
        return $diff->divInplace($denominator);
    }
}