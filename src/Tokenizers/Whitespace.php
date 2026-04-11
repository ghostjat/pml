<?php
declare(strict_types=1);

namespace Pml\Tokenizers;

/**
 * Whitespace Tokenizer — splits text on any whitespace sequence.
 * Fastest possible tokenizer: single preg_split with no extra processing.
 *
 * JIT optimized: no regex compilation per call (pattern is constant string).
 */
final class Whitespace implements Tokenizer
{
    /**
     * @return string[]
     */
    public function tokenize(string $text): array
    {
        return preg_split('/\s+/u', trim($text), -1, PREG_SPLIT_NO_EMPTY) ?: [];
    }
}
