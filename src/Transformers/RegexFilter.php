<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Dataset;

/**
 * Regex Filter — removes tokens matching a pattern from text strings.
 * Operates on PHP-side string data; identity pass on numeric Tensor datasets.
 */
final class RegexFilter implements Transformer
{
    private bool $fitted = false;

    /**
     * @param string[] $patterns  PCRE patterns to remove (e.g. ['/\d+/', '/https?:\S+/'])
     * @param string   $replace   Replacement string (default: empty string)
     */
    public function __construct(
        private readonly array  $patterns = [],
        private readonly string $replace  = ''
    ) {}

    public function fit(Dataset $dataset): void { $this->fitted = true; }

    public function transform(Dataset $dataset): Dataset
    {
        return $dataset;   // identity for numeric Tensor pipeline
    }

    /**
     * Apply all regex patterns to a string.
     */
    public function filter(string $text): string
    {
        foreach ($this->patterns as $pattern) {
            $text = preg_replace($pattern, $this->replace, $text) ?? $text;
        }
        return trim($text);
    }

    public function fitted(): bool { return $this->fitted; }
}
