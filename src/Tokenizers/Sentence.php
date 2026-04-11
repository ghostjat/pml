<?php
declare(strict_types=1);

namespace Pml\Tokenizers;

/**
 * Sentence Tokenizer — splits text into individual sentences.
 * Uses a heuristic boundary detector: sentence-ending punctuation (.!?)
 * followed by whitespace and an uppercase letter.
 *
 * JIT optimized: single preg_split — no FFI needed.
 */
final class Sentence implements Tokenizer
{
    /**
     * @return string[]
     */
    public function tokenize(string $text): array
    {
        // Split on .!? followed by whitespace — keep the delimiter with the preceding sentence
        $sentences = preg_split(
            '/(?<=[.!?])\s+(?=[A-Z])/u',
            trim($text),
            -1,
            PREG_SPLIT_NO_EMPTY
        ) ?: [];

        return array_values(array_filter($sentences, fn($s) => trim($s) !== ''));
    }
}
