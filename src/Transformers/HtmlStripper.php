<?php

declare(strict_types=1);

namespace Pml\Transformers;

/**
 * HTML Stripper (NLP).
 * Removes HTML/XML tags and decodes HTML entities from text documents.
 * * JIT & Memory Optimized:
 * - Operates entirely within PHP's native C-compiled text extensions (`strip_tags`, `html_entity_decode`).
 */
final class HtmlStripper
{
    /**
     * Transform the texts by stripping HTML and decoding entities.
     * @param string[] $texts Array of raw text strings.
     * @return string[] The purified texts.
     */
    public function transform(array $texts): array
    {
        $stripped = [];
        
        foreach ($texts as $text) {
            // strip_tags removes the structural HTML.
            // html_entity_decode safely transforms things like '&amp;' to '&'.
            $clean = strip_tags($text);
            $stripped[] = html_entity_decode($clean, ENT_QUOTES | ENT_HTML5, 'UTF-8');
        }
        
        return $stripped;
    }
}