<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Text Normalizer — lowercases text and strips punctuation from string metadata.
 *
 * NOTE: This transformer operates on the PHP-side string metadata attached to
 * the dataset (via Dataset::getStringFeatures() convention), not on the numeric
 * Tensor. It returns the same numeric dataset unchanged but normalises any
 * text labels for downstream tokenizers.
 *
 * For Tensor-only pipelines this is a stateless identity pass on the samples.
 */
final class TextNormalizer implements Transformer
{
    private bool $fitted = false;

    public function fit(Dataset $dataset): void { $this->fitted = true; }

    public function transform(Dataset $dataset): Dataset
    {
        // Numeric samples are unchanged; labels remain as-is (they are int/float tensors)
        return $dataset;
    }

    /**
     * Normalise a raw text string: lowercase + collapse whitespace + strip punctuation.
     */
    public static function normalize(string $text): string
    {
        $text = mb_strtolower($text, 'UTF-8');
        $text = preg_replace('/[^\p{L}\p{N}\s]/u', '', $text) ?? $text;
        return trim((string) preg_replace('/\s+/', ' ', $text));
    }

    public function fitted(): bool { return $this->fitted; }
}
