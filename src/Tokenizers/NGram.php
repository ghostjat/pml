<?php

declare(strict_types=1);

namespace Pml\Tokenizers;

/**
 * N-Gram Tokenizer.
 * Extracts sequences of N consecutive words (e.g., Bigrams: "New York", Trigrams: "Wall Street Journal").
 * * JIT & Memory Optimized:
 * - Bypasses slow `array_slice` and `implode` in loops.
 * - Uses direct string concatenation which the JIT resolves into raw C memory copies.
 */
final class NGram implements Tokenizer
{
    private int $min;
    private int $max;
    private Tokenizer $wordTokenizer;

    /**
     * @param int $min The minimum number of words in a single gram.
     * @param int $max The maximum number of words in a single gram.
     * @param Tokenizer|null $wordTokenizer The underlying tokenizer used to extract base words.
     */
    public function __construct(int $min = 2, int $max = 2, ?Tokenizer $wordTokenizer = null)
    {
        $this->min = max(1, $min);
        $this->max = max($this->min, $max);
        $this->wordTokenizer = $wordTokenizer ?? new Word();
    }

    public function tokenize(string $text): array
    {
        $words = $this->wordTokenizer->tokenize($text);
        $numWords = count($words);
        $nGrams = [];

        if ($numWords === 0) {
            return [];
        }

        // JIT Optimized Loop Setup
        for ($n = $this->min; $n <= $this->max; $n++) {
            $limit = $numWords - $n + 1;
            
            for ($i = 0; $i < $limit; $i++) {
                // Direct concatenation is exponentially faster than array_slice() + implode()
                $gram = $words[$i];
                for ($j = 1; $j < $n; $j++) {
                    $gram .= ' ' . $words[$i + $j];
                }
                $nGrams[] = $gram;
            }
        }

        return $nGrams;
    }
}