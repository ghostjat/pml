<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Dataset;

/**
 * Multibyte Text Normalizer — Unicode-aware lowercase, NFC normalization,
 * and optional diacritic stripping for non-ASCII scripts.
 */
final class MultibyteTextNormalizer implements Transformer
{
    private bool $fitted = false;

    public function __construct(private readonly bool $stripDiacritics = false) {}

    public function fit(Dataset $dataset): void { $this->fitted = true; }

    public function transform(Dataset $dataset): Dataset
    {
        // Numeric Tensor pipeline: identity pass
        return $dataset;
    }

    /**
     * Normalize a UTF-8 string.
     */
    public static function normalize(string $text, bool $stripDiacritics = false): string
    {
        // NFC normalization
        if (class_exists('\Normalizer')) {
            $text = \Normalizer::normalize($text, \Normalizer::FORM_C) ?: $text;
        }

        $text = mb_strtolower($text, 'UTF-8');

        if ($stripDiacritics) {
            // Decompose to NFD then strip combining marks (U+0300–U+036F)
            if (class_exists('\Normalizer')) {
                $nfd  = \Normalizer::normalize($text, \Normalizer::FORM_D) ?: $text;
                $text = preg_replace('/[\x{0300}-\x{036f}]/u', '', $nfd) ?? $text;
            }
        }

        return trim((string) preg_replace('/\s+/u', ' ', $text));
    }

    public function fitted(): bool { return $this->fitted; }
}
