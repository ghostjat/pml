<?php

declare(strict_types=1);

namespace Pml\Generation;

// ═══════════════════════════════════════════════════════════════════════════
//  BPE TOKENIZER
//
//  Parses a HuggingFace tokenizer.json and implements the core Byte-Pair
//  Encoding (BPE) merge algorithm in pure PHP.
//
//  Supported pre-tokenizer styles:
//    • ByteLevel  (GPT-2, Phi, Mistral, Llama-3 …)
//        - Every input byte is mapped to a unique Unicode surrogate character
//          before BPE is applied.  This makes the vocabulary closed over all
//          possible byte sequences and eliminates the need for an <unk> token
//          for arbitrary UTF-8 input.
//        - Pre-tokenises with the standard GPT-2 regex that respects English
//          contractions, numbers, punctuation, and whitespace.
//    • Whitespace / null  (simple models, legacy vocabs)
//        - Splits on whitespace only; uses Ġ-prefix convention for words
//          that start after a space (sentencepiece-compatible).
//
//  tokenizer.json layout (HuggingFace tokenizers library output):
//    {
//      "model": {
//        "type": "BPE",
//        "vocab":  { "<unk>": 0, "Ġ": 1, … },
//        "merges": [ "Ġ t", "Ġ a", … ]   // pairs separated by a single space
//      },
//      "added_tokens": [ { "id": 0, "content": "<unk>", "special": true }, … ],
//      "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false }
//    }
//
//  BPE algorithm (per word):
//    1. Represent the word as a sequence of tokens (one per byte-level char).
//    2. Scan all adjacent pairs and find the one with the lowest merge rank
//       (= its index in the merges list).
//    3. Replace every occurrence of that pair with the merged token.
//    4. Repeat until no known merge pair remains.
//    5. Map each surviving token to its vocabulary ID.
//
//  Time complexity per token: O(n² × log M) worst case (n = word length,
//  M = merge count), but in practice words are short (≤ 20 chars) and
//  the merge table fits in L2 cache, making it very fast.
//
//  Usage:
//    $tok = new BPETokenizer('/path/to/tokenizer.json');
//    $ids = $tok->encode("Should I take PCM or Commerce?");
//    echo $tok->decode($ids);
// ═══════════════════════════════════════════════════════════════════════════

final class BPETokenizer
{
    // ── Special token IDs (populated from tokenizer.json) ─────────────────
    public readonly int $bosId;
    public readonly int $eosId;
    public readonly int $unkId;
    public readonly int $padId;

    // ── Core tables ───────────────────────────────────────────────────────
    /** @var array<string, int>  token-string → vocabulary ID */
    private array $vocab;

    /** @var array<int, string>  vocabulary ID → token-string */
    private array $invVocab;

    /**
     * Merge priority table.
     * Key  = "left_token right_token"  (single space separator, matching the
     *         tokenizer.json "merges" format).
     * Value = rank (index in the merges list; lower = apply first).
     *
     * @var array<string, int>
     */
    private array $merges;

    // ── Pre-tokenisation ──────────────────────────────────────────────────
    /** Whether to use GPT-2-style byte-level pre-tokenisation. */
    private bool $isByteLevel;

    /** Whether to prepend a space to the first word (GPT-2 add_prefix_space). */
    private bool $addPrefixSpace;

    /**
     * GPT-2 byte → unicode-character lookup.
     * Maps each byte value (0-255) to its representative Unicode codepoint
     * as a UTF-8 string.  Bytes that are printable ASCII / latin-1 map to
     * themselves; others map to codepoints starting at U+0100.
     *
     * @var array<int, string>  byte_value → utf8_char
     */
    private array $byteEncoder;

    /**
     * Inverse of $byteEncoder: utf8_char → byte_value.
     *
     * @var array<string, int>
     */
    private array $byteDecoder;

    /**
     * Per-word BPE cache: word-string → int[]  (memoised merge results).
     * Amortises repeated encoding of the same surface form.
     *
     * @var array<string, int[]>
     */
    private array $cache = [];

    // ── Constructor ───────────────────────────────────────────────────────

    public function __construct(string $tokenizerJsonPath)
    {
        if (!file_exists($tokenizerJsonPath)) {
            throw new \RuntimeException("BPETokenizer: file not found: {$tokenizerJsonPath}");
        }

        $json = json_decode(
            file_get_contents($tokenizerJsonPath),
            associative: true,
            flags: JSON_THROW_ON_ERROR
        );

        $model = $json['model'] ?? throw new \RuntimeException(
            "BPETokenizer: tokenizer.json has no 'model' key."
        );

        if (($model['type'] ?? '') !== 'BPE') {
            throw new \RuntimeException(
                "BPETokenizer: only BPE models are supported, got '{$model['type']}'."
            );
        }

        // ── Vocabulary ────────────────────────────────────────────────────
        $this->vocab    = $model['vocab'];
        $this->invVocab = array_flip($this->vocab);

        // ── Merge rules ───────────────────────────────────────────────────
        // Convert the list of merge strings into a hash-map for O(1) lookup.
        $this->merges = [];
        foreach ($model['merges'] as $rank => $merge) {
            $this->merges[$merge] = $rank;
        }

        // ── Special tokens ────────────────────────────────────────────────
        $addedTokens = [];
        foreach ($json['added_tokens'] ?? [] as $t) {
            $addedTokens[$t['content']] = $t['id'];
            // Also ensure added tokens are in the vocab table
            $this->vocab[$t['content']]    ??= $t['id'];
            $this->invVocab[$t['id']]      ??= $t['content'];
        }

        // Common special-token names — extend if your model uses different ones
        $this->bosId = $addedTokens['<s>']   ?? $addedTokens['<|startoftext|>']
                    ?? $this->vocab['<s>']    ?? $this->vocab['<|startoftext|>']
                    ?? 1;
        $this->eosId = $addedTokens['</s>']  ?? $addedTokens['<|endoftext|>']
                    ?? $this->vocab['</s>']   ?? $this->vocab['<|endoftext|>']
                    ?? 2;
        $this->unkId = $addedTokens['<unk>'] ?? $this->vocab['<unk>'] ?? 0;
        $this->padId = $addedTokens['<pad>'] ?? $this->vocab['<pad>'] ?? $this->eosId;

        // ── Pre-tokeniser detection ────────────────────────────────────────
        $preTok             = $json['pre_tokenizer'] ?? null;
        $preTokType         = $preTok['type'] ?? 'Whitespace';
        $this->isByteLevel  = (strtolower($preTokType) === 'bytelevel')
                           || ($model['byte_level'] ?? false);
        $this->addPrefixSpace = (bool) ($preTok['add_prefix_space'] ?? false);

        if ($this->isByteLevel) {
            $this->buildByteEncoder();
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Encode a string of text into a sequence of token IDs.
     *
     * @param  string $text      The raw input text (UTF-8).
     * @param  bool   $addBos    Prepend BOS token (default true).
     * @param  bool   $addEos    Append EOS token (default false).
     * @return int[]             Token ID sequence.
     */
    public function encode(string $text, bool $addBos = true, bool $addEos = false): array
    {
        $ids = $addBos ? [$this->bosId] : [];

        if ($text === '') {
            if ($addEos) $ids[] = $this->eosId;
            return $ids;
        }

        // Optionally prepend a space so the first word gets its Ġ-prefixed form
        if ($this->addPrefixSpace && $text[0] !== ' ') {
            $text = ' ' . $text;
        }

        // Pre-tokenise into surface-form words, then BPE-encode each word
        foreach ($this->preTokenize($text) as $word) {
            foreach ($this->encodeWord($word) as $id) {
                $ids[] = $id;
            }
        }

        if ($addEos) $ids[] = $this->eosId;
        return $ids;
    }

    /**
     * Decode a sequence of token IDs back to a UTF-8 string.
     *
     * For byte-level models, reconstructs the original byte sequence from
     * the encoded Unicode surrogates and re-interprets as UTF-8.
     */
    public function decode(array $tokenIds): string
    {
        $text = '';
        foreach ($tokenIds as $id) {
            if ($id === $this->bosId || $id === $this->eosId || $id === $this->padId) {
                continue;
            }
            $text .= $this->invVocab[$id] ?? '';
        }

        if ($this->isByteLevel) {
            // Convert back from byte-level unicode surrogates to raw bytes
            $bytes = '';
            // Iterate over UTF-8 characters in the decoded string
            $chars = preg_split('//u', $text, -1, PREG_SPLIT_NO_EMPTY);
            foreach ($chars as $char) {
                if (isset($this->byteDecoder[$char])) {
                    $bytes .= chr($this->byteDecoder[$char]);
                }
            }
            return $bytes;
        }

        // Non-byte-level: replace Ġ (U+0120) with space (sentencepiece convention)
        return str_replace('Ġ', ' ', $text);
    }

    /** Decode a single token ID to its surface string (useful for streaming). */
    public function decodeToken(int $id): string
    {
        return $this->decode([$id]);
    }

    public function vocabSize(): int { return count($this->vocab); }

    // ── Pre-tokenisation ──────────────────────────────────────────────────

    /**
     * Split raw text into pre-token surface forms.
     *
     * ByteLevel (GPT-2) style:
     *   Uses the canonical GPT-2 regex that handles English contractions,
     *   letter runs, digit runs, punctuation runs, and whitespace.
     *   Each match is then byte-encoded: every byte of the UTF-8 representation
     *   becomes a single Unicode character from the byte-encoder table.
     *
     * Whitespace style:
     *   Splits on whitespace, prepending Ġ (U+0120) to all but the first word
     *   (matching the SentencePiece Ġ-prefix convention used by Llama-1/2).
     *
     * @return string[]  Surface-form word pieces, ready for BPE.
     */
    private function preTokenize(string $text): array
    {
        if ($this->isByteLevel) {
            return $this->preTokenizeByteLevel($text);
        }
        return $this->preTokenizeWhitespace($text);
    }

    /**
     * GPT-2 byte-level pre-tokenisation.
     *
     * Regex (Unicode-aware, identical to GPT-2's cl100k/p50k patterns):
     *   's | 't | 're | 've | 'm | 'll | 'd   — English contractions
     *   | ?\p{L}+                              — optional-space + letters
     *   | ?\p{N}+                              — optional-space + digits
     *   | ?[^\s\p{L}\p{N}]+                    — optional-space + other
     *   | \s+(?!\S)                            — trailing whitespace
     *   | \s+                                  — leading/mid whitespace
     *
     * Each matched chunk is then byte-encoded character-by-character.
     *
     * @return string[]
     */
    private function preTokenizeByteLevel(string $text): array
    {
        // GPT-2 tokenisation regex — single-quoted strings intentional
        $pattern = <<<'REGEX'
        /(?:'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)/u
        REGEX;

        preg_match_all($pattern, $text, $matches);
        $words = [];

        foreach ($matches[0] as $piece) {
            // Encode every byte of this UTF-8 chunk through the byte table
            $encoded = '';
            $len     = strlen($piece); // byte length, not character count
            for ($i = 0; $i < $len; $i++) {
                $encoded .= $this->byteEncoder[ord($piece[$i])];
            }
            if ($encoded !== '') {
                $words[] = $encoded;
            }
        }

        return $words;
    }

    /**
     * Simple whitespace pre-tokenisation with Ġ prefix (SentencePiece convention).
     *
     * @return string[]
     */
    private function preTokenizeWhitespace(string $text): array
    {
        $rawWords = preg_split('/(\s+)/u', $text, -1, PREG_SPLIT_NO_EMPTY);
        $words    = [];
        $first    = true;

        foreach ($rawWords as $word) {
            if (trim($word) === '') continue;
            // All words except the very first (or those after a space) get Ġ prefix
            $words[] = $first ? $word : ('Ġ' . $word);
            $first   = false;
        }

        return $words;
    }

    // ── BPE encoding ──────────────────────────────────────────────────────

    /**
     * Encode a single pre-tokenised word to its BPE token IDs.
     * Results are memoised in $this->cache.
     *
     * @return int[]
     */
    private function encodeWord(string $word): array
    {
        if (isset($this->cache[$word])) {
            return $this->cache[$word];
        }

        // Split word into individual Unicode characters (each is one BPE symbol)
        $chars = preg_split('//u', $word, -1, PREG_SPLIT_NO_EMPTY);
        if (empty($chars)) {
            return $this->cache[$word] = [];
        }
        if (count($chars) === 1) {
            return $this->cache[$word] = [$this->vocab[$chars[0]] ?? $this->unkId];
        }

        $ids = $this->bpeMerge($chars);
        return $this->cache[$word] = $ids;
    }

    /**
     * Core BPE merge algorithm.
     *
     * Starting from a list of single-character tokens, repeatedly finds the
     * adjacent pair with the lowest merge rank and merges all its occurrences,
     * until no more merges apply.
     *
     * The inner scan is O(n) per iteration where n is the current word length.
     * In practice words are very short (≤ 20 chars after pre-tokenisation) so
     * this is dominated by the hash lookups.
     *
     * @param  string[] $chars  Individual Unicode characters / byte-encoded chars.
     * @return int[]            Final token ID sequence.
     */
    private function bpeMerge(array $chars): array
    {
        $word = $chars; // mutable working copy

        while (count($word) > 1) {
            // ── Find the lowest-rank adjacent pair ─────────────────────────
            $bestRank = PHP_INT_MAX;
            $bestLeft = '';
            $bestRight = '';

            $n = count($word);
            for ($i = 0; $i < $n - 1; $i++) {
                $pair = $word[$i] . ' ' . $word[$i + 1];
                $rank = $this->merges[$pair] ?? PHP_INT_MAX;
                if ($rank < $bestRank) {
                    $bestRank  = $rank;
                    $bestLeft  = $word[$i];
                    $bestRight = $word[$i + 1];
                }
            }

            // No known merge pair remains — we're done
            if ($bestRank === PHP_INT_MAX) break;

            // ── Apply the merge: replace ALL occurrences of (left, right) ──
            // We must replace them left-to-right in a single pass to handle
            // overlapping (e.g. "a a a" with merge "a a" → "aa a", not "a aa").
            $merged  = $bestLeft . $bestRight;
            $newWord = [];
            $i       = 0;

            while ($i < count($word)) {
                if ($i < count($word) - 1
                    && $word[$i]     === $bestLeft
                    && $word[$i + 1] === $bestRight
                ) {
                    $newWord[] = $merged;
                    $i += 2; // skip both consumed tokens
                } else {
                    $newWord[] = $word[$i];
                    $i++;
                }
            }

            $word = $newWord;
        }

        // Map surviving tokens to their vocabulary IDs
        return array_map(
            fn(string $t): int => $this->vocab[$t] ?? $this->unkId,
            $word
        );
    }

    // ── Byte-level encoding table ─────────────────────────────────────────

    /**
     * Build the GPT-2 byte ↔ unicode character mapping.
     *
     * GPT-2 maps every byte (0-255) to a unique printable Unicode character so
     * that the BPE vocabulary is a closed set over arbitrary byte sequences.
     *
     * The mapping for "clean" bytes (printable ASCII and most latin-1) is the
     * identity: byte 0x41 ('A') → 'A'.  The remaining bytes (control codes,
     * DEL, non-breaking space, soft-hyphen) are mapped to codepoints starting
     * at U+0100 to keep the entire table within the basic multilingual plane.
     *
     * Reference implementation (Python):
     *   https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
     */
    private function buildByteEncoder(): void
    {
        // "Nice" bytes: printable ASCII (33-126) + most latin-1 (161-172, 174-255)
        $bs = array_merge(
            range(ord('!'), ord('~')),        // 33-126  (printable ASCII, no space)
            range(ord("\xA1"), ord("\xAC")),   // 161-172 (latin-1, no NBSP)
            range(ord("\xAE"), ord("\xFF"))    // 174-255 (latin-1, no soft-hyphen)
        );

        // Map clean bytes to themselves; dirty bytes go to U+0100 onward
        $cs = $bs; // unicode codepoint destinations — starts identical
        $n  = 0;

        for ($b = 0; $b < 256; $b++) {
            if (!in_array($b, $bs, true)) {
                $bs[] = $b;
                $cs[] = 256 + $n; // start at U+0100 to avoid collisions
                $n++;
            }
        }

        $this->byteEncoder = [];
        $this->byteDecoder = [];

        foreach ($bs as $idx => $byteVal) {
            $char = mb_chr($cs[$idx], 'UTF-8');
            $this->byteEncoder[$byteVal] = $char;
            $this->byteDecoder[$char]    = $byteVal;
        }
    }
}
