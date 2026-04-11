<?php

declare(strict_types=1);

namespace Pml\Metrics\Reports;

use Pml\Tensor;

/**
 * Confusion Matrix Generator.
 * * JIT & Memory Optimized:
 * - Completely bypasses PHP loops. 
 * - Generates a 2D classification matrix instantly using an AVX2 Math-Hashing trick.
 */
final class ConfusionMatrix
{
    /**
     * Generate the Confusion Matrix for Multi-Class predictions.
     * @return array<int, array<int, int>> A 2D array [True Class][Predicted Class]
     */
    public static function generate(Tensor $predictions, Tensor $labels): array
    {
        $preds = $predictions->round();
        $y = $labels->round();

        // Determine the dimensions of the matrix
        $maxLabel = $y->max();
        $maxPred = $preds->max();
        $numClasses = (int) max($maxLabel, $maxPred) + 1;

        if ($numClasses <= 0) return [];

        // THE AVX2 HASHING TRICK:
        // Hash formula: (y_true * numClasses) + y_pred
        // This generates a unique integer index for every possible cell in the 2D matrix
        $scaledLabels = $y->mulScalar((float) $numClasses);
        $hash = $scaledLabels->addInplace($preds);
        
        // C-Level Bincount counts the occurrences of every hash instantly
        $bincount = $hash->bincount();
        
        // Pad the 1D tensor in C if the max hash wasn't encountered
        $expectedSize = $numClasses * $numClasses;
        $actualSize = $bincount->size();
        
        if ($actualSize < $expectedSize) {
            $bincount = $bincount->pad([0, $expectedSize - $actualSize]);
        }

        // Zero-copy reshape the flat counts directly into the 2D matrix dimensions
        $matrixTensor = $bincount->reshape($numClasses, $numClasses);
        
        // Extract to PHP array for final reporting
        $flat = $matrixTensor->toFlatArray();
        $matrix = [];
        
        for ($i = 0; $i < $numClasses; $i++) {
            $row = array_slice($flat, $i * $numClasses, $numClasses);
            $matrix[$i] = array_map('intval', $row);
        }
        
        return $matrix;
    }
}