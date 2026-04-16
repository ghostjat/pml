<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Stateful;
use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Term Frequency - Inverse Document Frequency (TF-IDF) Transformer.
 * Scales word frequencies to penalize words that appear too commonly across all documents.
 * * JIT & Memory Optimized:
 * - 100% pure C-level computation using OpenBLAS Broadcasting.
 * - Extracts Document Frequency (DF) instantly via `greater` masking and `sumAxis`.
 */
final class TfIdfTransformer implements Transformer, Stateful
{
    private ?Tensor $idf = null;

    public function fit(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $n = (float) $x->shape()[0];

        // 1. Document Frequency (DF)
        // Creates a boolean mask (1.0 if term count > 0, else 0.0)
        $zero = Tensor::zeros(1);
        $mask = $x->greater($zero); 
        
        // Summing the mask vertically gives us the number of documents containing each term
        $df = $mask->sumAxis(0);

        // 2. Inverse Document Frequency (IDF)
        // Standard smooth formulation: log( (N + 1) / (DF + 1) ) + 1.0
        $dfPlusOne = $df->addScalar(1.0);
        
        // Broadcast the Scalar Numerator across the entire DF shape in C
        $numerator = Tensor::zeros(...$df->shape())->addScalarInplace($n + 1.0);
        
        // Execute the element-wise division and log()
        $division = $numerator->divInplace($dfPlusOne);
        
        $this->idf = $division->log()->addScalarInplace(1.0);
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("TfIdfTransformer has not been fitted.");
        }

        // TF-IDF = TF * IDF
        // The IDF vector [VocabSize] automatically broadcasts against the TF matrix [Batch, VocabSize]
        // This calculates the complete transformation for thousands of docs natively in C.
        $tfidf = $dataset->samples()->mul($this->idf);

        return new Dataset($tfidf, $dataset->labels());
    }

    public function fitted(): bool
    {
        return $this->idf !== null;
    }

    public function getStateDict(string $prefix = ''): array
    {
        $dict = [];
        if ($this->idf !== null) { $dict[$prefix . 'idf'] = $this->idf; }
        return $dict;
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->idf = $dict[$prefix . 'idf'] ?? null;
    }
}