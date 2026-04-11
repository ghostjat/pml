<?php

declare(strict_types=1);

namespace Pml\Transformers;

/**
 * Stop Word Filter.
 * Removes common or uninformative words (e.g., 'the', 'is', 'at') from raw text documents.
 * * JIT & Memory Optimized:
 * - Utilizes an O(1) hash map lookup to bypass slow `in_array` loops.
 * - Processes large arrays natively using fast PCRE regex splits.
 */
final class StopWordFilter
{
    private array $stopWords;

    /**
     * @param string[] $stopWords Array of words to filter out.
     */
    public function __construct(array $stopWords)
    {
        // Convert to a hash map for instant O(1) lookups during the transformation loop
        $this->stopWords = array_flip(array_map('strtolower', $stopWords));
    }

    /**
     * Filters the stop words out of an array of text documents.
     * @param string[] $texts Array of raw text strings.
     * @return string[] The filtered texts.
     */
    public function transform(array $texts): array
    {
        $filtered = [];
        
        foreach ($texts as $text) {
            // Highly optimized native C PCRE split
            $words = preg_split('/\W+/u', strtolower($text), -1, PREG_SPLIT_NO_EMPTY) ?: [];
            
            $kept = [];
            foreach ($words as $word) {
                // O(1) dictionary lookup
                if (!isset($this->stopWords[$word])) {
                    $kept[] = $word;
                }
            }
            
            // Reconstruct the document
            $filtered[] = implode(' ', $kept);
        }
        
        return $filtered;
    }
}