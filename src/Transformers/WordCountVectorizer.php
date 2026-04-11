<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Word Count Vectorizer (Bag of Words).
 * Converts a collection of raw text documents into a C-Level Tensor of token counts.
 * * JIT & Memory Optimized:
 * - Completely bypasses nested PHP arrays.
 * - Writes token counts directly into a pre-allocated OpenBLAS CData buffer pointer.
 */
final class WordCountVectorizer
{
    private ?int $maxFeatures;
    private array $vocabulary = [];
    private bool $fitted = false;

    /**
     * @param int|null $maxFeatures The maximum size of the vocabulary. Keeps top most frequent words.
     */
    public function __construct(?int $maxFeatures = null)
    {
        $this->maxFeatures = $maxFeatures;
    }

    /**
     * Learns the vocabulary dictionary from the training corpus.
     * @param string[] $texts Array of raw text strings.
     */
    public function fit(array $texts): void
    {
        $wordCounts = [];
        
        foreach ($texts as $text) {
            $words = $this->tokenize($text);
            foreach ($words as $word) {
                $wordCounts[$word] = ($wordCounts[$word] ?? 0) + 1;
            }
        }

        // Sort vocabulary by frequency descending
        arsort($wordCounts);

        if ($this->maxFeatures !== null) {
            $wordCounts = array_slice($wordCounts, 0, $this->maxFeatures, true);
        }

        $index = 0;
        foreach ($wordCounts as $word => $count) {
            $this->vocabulary[$word] = $index++;
        }

        $this->fitted = true;
    }

    /**
     * Transforms text documents into a high-performance Term Frequency Dataset.
     * @param string[] $texts Array of raw text strings.
     * @param array|null $labels Optional ground-truth labels.
     */
    public function transform(array $texts, ?array $labels = null): Dataset
    {
        if (!$this->fitted) {
            throw new RuntimeException("WordCountVectorizer has not been fitted.");
        }

        $numDocs = count($texts);
        $vocabSize = count($this->vocabulary);

        if ($numDocs === 0 || $vocabSize === 0) {
            throw new RuntimeException("Cannot transform empty texts or empty vocabulary.");
        }

        // 1. Allocate a massive continuous zero-tensor natively in C
        $tensor = Tensor::zeros($numDocs, $vocabSize);
        
        // 2. Extract the raw C pointer buffer to completely bypass PHP Array overhead
        $cdata = $tensor->buffer();

        foreach ($texts as $i => $text) {
            $words = $this->tokenize($text);
            $counts = [];
            
            // Tally counts using PHP hash maps (extremely fast for strings)
            foreach ($words as $word) {
                if (isset($this->vocabulary[$word])) {
                    $idx = $this->vocabulary[$word];
                    $counts[$idx] = ($counts[$idx] ?? 0) + 1;
                }
            }

            // Write the non-zero sparse counts directly into the C-Tensor memory!
            $rowOffset = $i * $vocabSize;
            foreach ($counts as $idx => $count) {
                $cdata[$rowOffset + $idx] = (float) $count;
            }
        }

        $labelTensor = null;
        if ($labels !== null) {
            if (count($labels) !== $numDocs) {
                throw new \InvalidArgumentException("Number of labels must match number of documents.");
            }
            $labelTensor = Tensor::fromArray($labels);
        }

        return new Dataset($tensor, $labelTensor);
    }

    /**
     * Simple internal tokenizer. Lowercases and splits on non-word boundaries.
     */
    private function tokenize(string $text): array
    {
        $text = strtolower($text);
        return preg_split('/\W+/u', $text, -1, PREG_SPLIT_NO_EMPTY) ?: [];
    }

    public function vocabulary(): array
    {
        return $this->vocabulary;
    }
}