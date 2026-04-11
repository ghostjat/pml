<?php

declare(strict_types=1);

namespace Pml\Tokenizers;

/**
 * Character Gram Tokenizer.
 * Extracts sequences of N consecutive characters. 
 * Excellent for sub-word feature extraction, DNA sequence analysis, and language detection.
 * * JIT & Memory Optimized:
 * - Uses highly efficient `mb_str_split` to securely map multi-byte unicode characters.
 * - String combinations are processed via raw continuous concatenation.
 */
final class CharGram implements Tokenizer
{
    private int $min;
    private int $max;

    /**
     * @param int $min The minimum number of characters in a single gram.
     * @param int $max The maximum number of characters in a single gram.
     */
    public function __construct(int $min = 3, int $max = 3)
    {
        $this->min = max(1, $min);
        $this->max = max($this->min, $max);
    }

    public function tokenize(string $text): array
    {
        // Normalize multiple spaces down to a single space to prevent dead token noise
        $text = strtolower(preg_replace('/\s+/u', ' ', $text) ?? '');
        
        // Instant C-level multi-byte string separation
        $chars = mb_str_split($text);
        $numChars = count($chars);
        $charGrams = [];

        if ($numChars === 0) {
            return [];
        }

        // JIT Optimized Loop Setup
        for ($n = $this->min; $n <= $this->max; $n++) {
            $limit = $numChars - $n + 1;
            
            for ($i = 0; $i < $limit; $i++) {
                // Direct concatenation prevents array allocation overhead
                $gram = $chars[$i];
                for ($j = 1; $j < $n; $j++) {
                    $gram .= $chars[$i + $j];
                }
                $charGrams[] = $gram;
            }
        }

        return $charGrams;
    }
}