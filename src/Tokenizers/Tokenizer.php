<?php

declare(strict_types=1);

namespace Pml\Tokenizers;

/**
 * Interface for all Text Tokenizers.
 */
interface Tokenizer
{
    /**
     * Tokenize a block of text into an array of string tokens.
     * * @param string $text The raw string to tokenize.
     * @return string[] The extracted tokens.
     */
    public function tokenize(string $text): array;
}