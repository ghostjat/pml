<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Stateful;
use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Okapi BM25 Transformer (NLP).
 * An advanced alternative to TF-IDF that saturates Term Frequency to prevent 
 * overly repetitive words from dominating the relevance score.
 * * JIT & Memory Optimized:
 * - 100% Vectorized AVX2 OpenBLAS execution.
 * - Extracts Document Lengths and IDFs natively without traversing PHP arrays.
 */
final class BM25Transformer implements Transformer, Stateful
{
    private float $k1;
    private float $b;
    
    private ?Tensor $idf = null;
    private float $avgdl = 0.0;

    /**
     * @param float $k1 Term frequency saturation parameter (typically 1.2 to 2.0).
     * @param float $b Length normalization parameter (typically 0.75).
     */
    public function __construct(float $k1 = 1.2, float $b = 0.75)
    {
        $this->k1 = $k1;
        $this->b = $b;
    }

    public function fit(Dataset $dataset): void
    {
        $tf = $dataset->samples();
        $n = (float) $tf->shape()[0];

        // 1. Document Frequency (DF)
        $zero = Tensor::zeros(1);
        $df = $tf->greater($zero)->sumAxis(0);

        // 2. IDF = ln( ((N - DF + 0.5) / (DF + 0.5)) + 1 )
        $numerator = $df->mulScalar(-1.0)->addScalarInplace($n + 0.5);
        $denominator = $df->addScalar(0.5);
        $this->idf = $numerator->divInplace($denominator)->addScalarInplace(1.0)->log();

        // 3. Average Document Length (avgdl)
        // Sum of TF across the row (axis 1) gives the total words in the document
        $docLengths = $tf->sumAxis(1);
        $this->avgdl = $docLengths->mean();
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("BM25Transformer is not fitted.");
        }

        $tf = $dataset->samples();
        
        // Extract Document Lengths for inference data [Batch, 1]
        $docLengths = $tf->sumAxis(1)->expandDims(1);

        // Length Normalization Factor: (1 - b) + b * (doc_len / avgdl)
        $lengthNorm = $docLengths->mulScalar(1.0 / ($this->avgdl + 1e-8))
                                 ->mulScalarInplace($this->b)
                                 ->addScalarInplace(1.0 - $this->b);

        // Denominator: TF + k1 * length_norm
        $denominator = $lengthNorm->mulScalar($this->k1)->addInplace($tf);

        // Numerator: TF * (k1 + 1)
        $numerator = $tf->mulScalar($this->k1 + 1.0);

        // Final BM25: IDF * (Numerator / Denominator)
        // Entire chain executes purely in OpenBLAS C-memory
        $bm25 = $numerator->divInplace($denominator)->mulInplace($this->idf);

        return new Dataset($bm25, $dataset->labels());
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