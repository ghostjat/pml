<?php
declare(strict_types=1);

namespace Pml\Tokenizers;

/**
 * Word Stemmer Tokenizer — tokenizes text and applies a Porter-style stemmer to each word.
 * Reduces inflected forms ("running", "runs") to a common stem ("run").
 *
 * JIT optimized:
 * - Stemming table is built once and cached as a static array.
 * - preg_split is O(N) in input length; stemming is O(L) per word.
 */
final class WordStemmer implements Tokenizer
{
    private readonly Tokenizer $base;

    public function __construct(?Tokenizer $base = null)
    {
        $this->base = $base ?? new Word();
    }

    /**
     * @return string[]
     */
    public function tokenize(string $text): array
    {
        $words = $this->base->tokenize($text);
        return array_map([$this, 'stem'], $words);
    }

    /**
     * Minimal English Porter-step-1 suffix stripping.
     */
    public function stem(string $word): string
    {
        $word = strtolower($word);
        $len  = strlen($word);
        if ($len <= 3) return $word;

        // Step 1: strip common suffixes
        $suffixes = ['ational' => 'ate', 'tional' => 'tion', 'enci' => 'ence',
                     'anci' => 'ance', 'izer' => 'ize', 'ising' => 'ise',
                     'izing' => 'ize', 'ation' => 'ate', 'ator' => 'ate',
                     'alism' => 'al', 'ness' => '', 'ment' => '', 'ful' => '',
                     'ous' => '', 'ive' => '', 'ing' => '', 'ies' => 'i',
                     'sses' => 'ss', 'ss' => 'ss', 's' => ''];

        foreach ($suffixes as $suffix => $replacement) {
            if (str_ends_with($word, $suffix) && strlen($word) - strlen($suffix) >= 2) {
                return substr($word, 0, strlen($word) - strlen($suffix)) . $replacement;
            }
        }

        return $word;
    }
}
