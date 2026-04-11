<?php 
declare(strict_types=1);

namespace Pml\Tokenizers;

/**
 * Word Tokenizer.
 * Extracts individual words from a block of text.
 * * JIT & Memory Optimized:
 * - Employs PHP's native C-compiled PCRE regex engine to extract words instantly.
 * - Safely handles multi-byte unicode strings without loop overhead.
 */
final class Word implements Tokenizer
{
    public function tokenize(string $text): array
    {
        // strtolower is heavily optimized in PHP 8.x
        // \W+ matches any non-word character (safely ignoring unicode letters/numbers)
        return preg_split('/\W+/u', strtolower($text), -1, PREG_SPLIT_NO_EMPTY) ?: [];
    }
}