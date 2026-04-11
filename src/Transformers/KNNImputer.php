<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * K-Nearest Neighbors Imputer.
 * Fills missing NaN values by averaging the features of the K most similar data points.
 * Provides far more accurate reconstructions than a global mean Imputer.
 */
final class KNNImputer implements Transformer
{
    private int $k;
    private ?Tensor $fitSamples = null;

    public function __construct(int $k = 5)
    {
        $this->k = $k;
    }

    public function fit(Dataset $dataset): void
    {
        // Save the valid, non-NaN rows as the basis for neighborhood search
        $x = $dataset->samples();
        
        // Find rows that have NO NaNs
        $hasNanMask = $x->isNan()->maxAxis(1); // 1.0 if any NaN exists in the row
        $validRowsMask = $hasNanMask->logicalNot();
        
        $this->fitSamples = $x->booleanIndex($validRowsMask);
        
        if ($this->fitSamples->shape()[0] < $this->k) {
            throw new RuntimeException("KNNImputer: Not enough fully valid rows to satisfy K={$this->k}.");
        }
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) throw new RuntimeException("KNNImputer is not fitted.");

        $x = $dataset->samples()->copy();
        $rows = $x->shape()[0];
        $cols = $x->shape()[1];
        
        // Convert to flat array for fast point-by-point inspection in JIT
        $flatX = $x->toFlatArray();
        
        for ($i = 0; $i < $rows; $i++) {
            $rowOffset = $i * $cols;
            $missingCols = [];
            
            for ($c = 0; $c < $cols; $c++) {
                if (is_nan($flatX[$rowOffset + $c])) {
                    $missingCols[] = $c;
                }
            }

            if (!empty($missingCols)) {
                // Find K-Nearest Neighbors ignoring the NaN columns
                $targetRow = $x->row($i)->nanToNumInplace(0.0, 0.0, 0.0);
                
                // Euclidean Distance against the valid Fit Samples
                $sqDist = $this->fitSamples->sub($targetRow)->square()->sumAxis(1);
                
                $sortedIndices = $sqDist->argsort();
                $kIndices = $sortedIndices->slice(0, 0, $this->k);
                
                $neighbors = $this->fitSamples->take($kIndices, 0);
                $neighborMeans = $neighbors->meanAxis(0)->toFlatArray();
                
                // Impute the missing columns in the C-Tensor
                foreach ($missingCols as $c) {
                    $x->buffer()[$rowOffset + $c] = $neighborMeans[$c];
                }
            }
        }

        return new Dataset($x, $dataset->labels());
    }

    public function fitted(): bool
    {
        return $this->fitSamples !== null;
    }
}