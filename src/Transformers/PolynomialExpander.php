<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Polynomial Expander.
 * Generates higher-order interaction features (e.g., [a, b] becomes [a, b, a^2, ab, b^2]).
 * * JIT & Memory Optimized:
 * - Uses zero-copy axis slices to multiply columns.
 * - Flushes all interaction matrices into a single C-level memory concatenation.
 */
final class PolynomialExpander implements Transformer
{
    public function fit(Dataset $dataset): void
    {
        // Polynomial expansion is stateless. No fitting is required.
    }

    public function transform(Dataset $dataset): Dataset
    {
        $x = $dataset->samples();
        $cols = $x->shape()[1];
        
        // Start with the original features
        $tensors = [$x];

        // Generate Degree-2 Interaction Features
        for ($i = 0; $i < $cols; $i++) {
            
            // Extract the base column as a zero-copy view: Shape [N, 1]
            $colI = $x->slice(1, $i, 1);
            
            for ($j = $i; $j < $cols; $j++) {
                
                // Multiply against interacting columns
                $colJ = $x->slice(1, $j, 1);
                
                // AVX2 element-wise multiplication
                $tensors[] = $colI->mul($colJ);
            }
        }

        // Execute a massive multi-pointer memory concatenation directly in C
        $expandedSamples = Tensor::concat($tensors, 1);

        return new Dataset($expandedSamples, $dataset->labels());
    }

    public function fitted(): bool
    {
        return true;
    }
}