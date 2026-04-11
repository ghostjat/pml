<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use InvalidArgumentException;
use RuntimeException;

/**
 * Select K-Best Feature Selector.
 * Selects the top K most predictive features using a statistical scoring function.
 * * JIT & Memory Optimized:
 * - Uses a Vectorized Pearson Correlation algorithm.
 * - Broadcasts [N, 1] labels against [N, D] features to compute the predictive scores 
 * of thousands of columns in a single OpenBLAS pass.
 */
final class SelectKBest implements Transformer
{
    private int $k;
    private ?array $selectedColumns = null;

    /**
     * @param int $k The number of top features to select.
     */
    public function __construct(int $k = 10)
    {
        if ($k < 1) {
            throw new InvalidArgumentException("SelectKBest requires K to be at least 1.");
        }
        $this->k = $k;
    }

    public function fit(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $y = $dataset->labels();
        
        if ($y === null) {
            throw new InvalidArgumentException("SelectKBest requires a labeled dataset to compute predictive scores.");
        }
        
        $cols = $dataset->numColumns();
        $k = min($this->k, $cols);
        
        // Fast Vectorized Pearson Correlation Formula:
        // Score = | sum((X - uX) * (Y - uY)) / sqrt(sum((X - uX)^2) * sum((Y - uY)^2)) |
        
        $yCol = $y->ndim() === 1 ? $y->expandDims(1) : $y;
        $yCentered = $yCol->sub($yCol->meanAxis(0));
        
        // Compute Label variance sum (Scalar extraction)
        $yVarSum = $yCentered->square()->sumAxis(0)->toFlatArray()[0];
        
        if ($yVarSum <= 1e-8) {
            // Labels have absolutely no variance. Fallback to selecting the first K features.
            $this->selectedColumns = range(0, $k - 1);
            return;
        }
        
        // Center all Features
        $xCentered = $x->sub($x->meanAxis(0));
        $xVarSum = $xCentered->square()->sumAxis(0); // Shape: [1, D]
        
        // Numerator: (X_c * Y_c).
        // Y_c [N, 1] automatically broadcasts to multiply against all [N, D] feature columns.
        $numerator = $xCentered->mul($yCentered)->sumAxis(0);
        
        // Denominator: sqrt( X_var_sum * Y_var_sum )
        $denominator = $xVarSum->mulScalar($yVarSum)->sqrt()->addScalarInplace(1e-8);
        
        // Absolute Correlation Scores: [1, D] -> Flat Array
        $scores = $numerator->divInplace($denominator)->abs()->toFlatArray();
        
        // Sort scores descending while maintaining their original column indices as keys
        arsort($scores);
        
        // Extract the Top K indices
        $this->selectedColumns = array_slice(array_keys($scores), 0, $k);
        
        // Sort the remaining indices ascending to preserve original structural order of features
        sort($this->selectedColumns);
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("SelectKBest is not fitted.");
        }
        
        return $dataset->select($this->selectedColumns);
    }

    public function fitted(): bool
    {
        return $this->selectedColumns !== null;
    }
}