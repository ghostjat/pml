<?php

declare(strict_types=1);

namespace Pml\Classic\FeatureExtraction\Text;

use Pml\Tensor;

// ═══════════════════════════════════════════════════════════════════════════
//  CountVectorizer — sklearn.feature_extraction.text.CountVectorizer
//
//  Converts a collection of raw text documents to a 2-D integer (stored as
//  float32) count matrix:
//
//    X[i, j] = number of times N-gram j appears in document i
//
//  ── Pipeline note ─────────────────────────────────────────────────────────
//
//  This class does NOT implement Pml\Classic\Transformer because its input
//  is array<string> rather than Tensor.  Chain it manually before passing
//  the resulting Tensor into a standard Pipeline or TfidfTransformer.
//
//  ── Tokenisation ──────────────────────────────────────────────────────────
//
//  Documents are tokenised with a single preg_match_all('/[a-z0-9]+/', …)
//  call (after optional lowercasing).  This matches sklearn's default
//  pattern, which discards pure-punctuation tokens.
//
//  N-grams of every size in [min_n, max_n] are extracted by sliding a
//  window of the appropriate width over the unigram token list, then
//  joining with a single space — the sklearn canonical representation.
//
//  ── Vocabulary ordering ───────────────────────────────────────────────────
//
//  After scanning the corpus, vocabulary keys are sorted alphabetically
//  (ksort) and indices are reassigned 0…|V|-1 in that order.  This mirrors
//  sklearn, which uses Python's sorted() on the seen token set before
//  assigning column indices.
//
//  ── PHP-native speed notes ────────────────────────────────────────────────
//
//  All heavy work stays in PHP arrays before the Tensor bridge is crossed:
//
//    • preg_match_all — one C regex call per document (not per word)
//    • array_slice / implode for N-gram windows — native C extensions
//    • Vocabulary stored as hash map: isset() for O(1) lookup
//    • Count matrix accumulated in a flat PHP float[] array; packed into
//      a Tensor in a single pass at the end — avoids per-element FFI overhead
//
//  The hot inner loop (document × vocabulary column) is pure PHP integer
//  arithmetic on flat array offsets — no FFI per element.
// ═══════════════════════════════════════════════════════════════════════════

final class CountVectorizer
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Map of N-gram string → column index.
     * Sorted alphabetically, indices 0…|V|-1.
     * @var array<string, int>
     */
    public readonly array $vocabulary_;

    /** Number of unique N-grams = number of output features. */
    public readonly int $n_features_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param array{int,int} $ngram_range  [min_n, max_n] — e.g. [1,1] for unigrams,
     *                                     [1,2] for unigrams + bigrams.
     * @param bool           $lowercase    Lowercase all text before tokenising.
     */
    public function __construct(
        private readonly array $ngram_range = [1, 1],
        private readonly bool  $lowercase   = true,
    ) {
        [$min, $max] = $ngram_range;
        if ($min < 1 || $max < $min) {
            throw new \InvalidArgumentException(
                "CountVectorizer: ngram_range must satisfy 1 ≤ min_n ≤ max_n; "
                . "got [{$min}, {$max}]."
            );
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Learn the vocabulary from a corpus of raw strings.
     *
     * @param string[] $raw_documents
     */
    public function fit(array $raw_documents): static
    {
        $seen = [];
        foreach ($raw_documents as $doc) {
            foreach ($this->extractNgrams($this->tokenize($doc)) as $ng) {
                $seen[$ng] = true;
            }
        }

        // Sort alphabetically — sklearn behaviour
        ksort($seen);

        $vocab = [];
        $idx   = 0;
        foreach ($seen as $ng => $_) {
            $vocab[$ng] = $idx++;
        }

        $this->vocabulary_ = $vocab;
        $this->n_features_  = $idx;

        return $this;
    }

    /**
     * Map documents to a count matrix.
     *
     * @param  string[] $raw_documents  [n_docs]
     * @return Tensor                   [n_docs, |vocabulary_|]  float32 counts
     */
    public function transform(array $raw_documents): Tensor
    {
        if (!isset($this->vocabulary_)) {
            throw new \RuntimeException(
                'CountVectorizer is not fitted yet. Call fit() before transform().'
            );
        }

        $nDocs    = count($raw_documents);
        $nFeats   = $this->n_features_;
        $vocab    = $this->vocabulary_;

        // ── Accumulate into a flat PHP float[] first ───────────────────────
        //
        // Accumulating counts in a plain PHP array (O(1) offset arithmetic)
        // is dramatically faster than n_docs × n_features FFI writes.
        // We pack the finished array into a Tensor in one shot at the end.
        $counts = array_fill(0, $nDocs * $nFeats, 0.0);

        foreach ($raw_documents as $i => $doc) {
            $base  = $i * $nFeats;
            foreach ($this->extractNgrams($this->tokenize($doc)) as $ng) {
                if (isset($vocab[$ng])) {
                    $counts[$base + $vocab[$ng]] += 1.0;
                }
            }
        }

        // ── Pack into Tensor via a single FFI memcpy ───────────────────────
        $out  = Tensor::zeros([$nDocs, $nFeats]);
        $pack = pack('f*', ...$counts);
        \FFI::memcpy($out->buffer, $pack, $nDocs * $nFeats * 4);

        return $out;
    }

    /**
     * Fit then immediately transform — equivalent to fit($docs)->transform($docs).
     *
     * @param  string[] $raw_documents
     * @return Tensor   [n_docs, |vocabulary_|]
     */
    public function fit_transform(array $raw_documents): Tensor
    {
        return $this->fit($raw_documents)->transform($raw_documents);
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * Tokenise a document into a list of lowercase word tokens.
     *
     * Uses a single preg_match_all call — one C regex execution, not a loop.
     * Matches sklearn's default token pattern: one or more alphanumeric chars.
     *
     * @return string[]
     */
    private function tokenize(string $doc): array
    {
        if ($this->lowercase) {
            $doc = strtolower($doc);
        }
        preg_match_all('/[a-z0-9]+/', $doc, $m);
        return $m[0];
    }

    /**
     * Extract all N-grams for sizes in [min_n, max_n] from a token list.
     *
     * Uses array_slice (native C) + implode to build each gram string.
     * For the unigram-only case [1,1] this is a direct identity pass.
     *
     * @param  string[] $tokens
     * @return string[]
     */
    private function extractNgrams(array $tokens): array
    {
        [$minN, $maxN] = $this->ngram_range;
        $nTokens = count($tokens);
        $ngrams  = [];

        for ($size = $minN; $size <= $maxN; $size++) {
            $limit = $nTokens - $size + 1;
            for ($i = 0; $i < $limit; $i++) {
                // array_slice + implode: both implemented in C — no PHP loop per token
                $ngrams[] = implode(' ', array_slice($tokens, $i, $size));
            }
        }

        return $ngrams;
    }
}
