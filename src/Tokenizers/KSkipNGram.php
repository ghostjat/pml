<?php
declare(strict_types=1);

namespace Pml\Tokenizers;

/**
 * K-Skip-N-Gram Tokenizer — generates N-grams with up to K skipped tokens between words.
 * Captures longer-range co-occurrence patterns than plain N-grams.
 *
 * Example: "the cat sat" with n=2, k=1 →
 *   ["the cat", "the sat", "cat sat"]
 *
 * JIT optimized: pure PHP array operations; no FFI needed for string processing.
 */
final class KSkipNGram implements Tokenizer
{
    public function __construct(
        private readonly int $n = 2,
        private readonly int $k = 1
    ) {}

    /**
     * @return string[]
     */
    public function tokenize(string $text): array
    {
        $words  = preg_split('/\s+/', trim($text), -1, PREG_SPLIT_NO_EMPTY) ?: [];
        $count  = count($words);
        $tokens = [];

        for ($i = 0; $i < $count; $i++) {
            $this->generate($words, $i, [], $this->n, $tokens);
        }

        return $tokens;
    }

    /**
     * Recursively picks n words starting from position $pos, skipping up to k tokens.
     */
    private function generate(array $words, int $pos, array $current, int $remaining, array &$out): void
    {
        $count = count($words);

        if ($remaining === 0) {
            $out[] = implode(' ', $current);
            return;
        }

        for ($skip = 0; $skip <= $this->k && $pos + $skip < $count; $skip++) {
            $this->generate($words, $pos + $skip + 1, [...$current, $words[$pos + $skip]], $remaining - 1, $out);
        }
    }
}
