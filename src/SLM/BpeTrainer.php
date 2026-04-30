<?php

declare(strict_types=1);

namespace Pml\SLM;

/**
 * Byte-level BPE trainer compatible with the C tokenizer's tok_load() API.
 *
 * Output files:
 *   vocab.json   — flat {"token_str": id, ...}  (GPT-2 byte-to-unicode keys)
 *   merges.txt   — "#version: 0.2\ntok_a tok_b\n..." one merge per line
 *
 * Special tokens inserted at the top of the vocab so _resolve_specials() in C
 * can find them by their canonical names:
 *   id 0  <unk>   id 1  <s>   id 2  </s>   id 3  <pad>
 *
 * Bytes 4-259 are the 256 GPT-2 byte-to-unicode single-char base tokens.
 * BPE merge tokens start at id 260.
 *
 * Corpus representation: each text is converted to a space-delimited string
 * of GPT-2 unicode chars (e.g. "Ā Ġ H e l l o").  Merges are learned on
 * this representation and stored as pairs of those strings.
 */
final class BpeTrainer
{
    /** @var array<int,string>  byte → GPT-2 unicode char (UTF-8 encoded) */
    private array $b2u;

    /** @var array<string,int>  token_str → vocab_id */
    private array $vocab = [];

    /** @var array<array{string,string}>  learned merges [[str_a, str_b], ...] */
    private array $merges = [];

    /** Next available vocab id (incremented as tokens are added). */
    private int $nextId = 0;

    public function __construct(
        private readonly int $targetVocabSize = 8000,
        private readonly int $minFrequency    = 2
    ) {
        $this->b2u = self::buildByteToUnicode();
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Train BPE from an array of raw UTF-8 strings.
     * Safe to call multiple times (resets state each call).
     *
     * @param string[] $texts
     */
    public function train(array $texts): void
    {
        $this->vocab   = [];
        $this->merges  = [];
        $this->nextId  = 0;

        // ── 1. Seed vocabulary ───────────────────────────────────────────────
        foreach (['<unk>', '<s>', '</s>', '<pad>'] as $st) {
            $this->vocab[$st] = $this->nextId++;
        }
        // Insert 256 byte tokens in byte order (b2u is keyed by byte 0-255).
        $byteOrder = range(0, 255);
        foreach ($byteOrder as $byte) {
            $this->vocab[$this->b2u[$byte]] = $this->nextId++;
        }

        // ── 2. Build corpus ──────────────────────────────────────────────────
        // corpus: space-delimited unicode-char string → cumulative frequency
        $corpus = [];
        foreach ($texts as $text) {
            if ($text === '') continue;
            $charSeq = $this->textToCharSeq($text);
            $corpus[$charSeq] = ($corpus[$charSeq] ?? 0) + 1;
        }

        // ── 3. Iterative BPE merging ─────────────────────────────────────────
        while (count($this->vocab) < $this->targetVocabSize) {
            $pairs = $this->countPairs($corpus);
            if (empty($pairs)) break;

            arsort($pairs);
            reset($pairs);
            $best  = (string) key($pairs);
            $bestFreq = (int) current($pairs);

            if ($bestFreq < $this->minFrequency) break;

            $tabPos = strpos($best, "\t");
            if ($tabPos === false) break;
            $a      = substr($best, 0, $tabPos);
            $b      = substr($best, $tabPos + 1);
            $merged = $a . $b;

            $this->vocab[$merged] = $this->nextId++;
            $this->merges[]       = [$a, $b];

            $corpus = $this->applyMerge($corpus, $a, $b, $merged);
        }
    }

    /**
     * Save vocab.json and merges.txt for tok_load().
     */
    public function save(string $vocabPath, string $mergesPath): void
    {
        // vocab.json
        file_put_contents(
            $vocabPath,
            json_encode($this->vocab, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR)
        );

        // merges.txt
        $lines   = ["#version: 0.2\n"];
        foreach ($this->merges as [$a, $b]) {
            $lines[] = $a . ' ' . $b . "\n";
        }
        file_put_contents($mergesPath, implode('', $lines));
    }

    public function vocabSize(): int    { return count($this->vocab); }
    public function mergeCount(): int   { return count($this->merges); }

    // ── Internals ─────────────────────────────────────────────────────────────

    /**
     * Convert raw UTF-8 text to a space-delimited string of GPT-2 unicode chars.
     * e.g. "Hi" → "H i"  (bytes 72, 105 → pass-through chars H and i)
     */
    private function textToCharSeq(string $text): string
    {
        $len  = strlen($text);
        $out  = [];
        for ($i = 0; $i < $len; $i++) {
            $out[] = $this->b2u[ord($text[$i])];
        }
        return implode(' ', $out);
    }

    /**
     * Count adjacent pair frequencies across the corpus.
     * Key: "tok_a\ttok_b" (tab-separated to avoid collision with space-delimited format).
     *
     * @param array<string,int> $corpus
     * @return array<string,int>
     */
    private function countPairs(array $corpus): array
    {
        $pairs = [];
        foreach ($corpus as $word => $freq) {
            $tokens = explode(' ', $word);
            $n      = count($tokens);
            for ($i = 0; $i < $n - 1; $i++) {
                $key = $tokens[$i] . "\t" . $tokens[$i + 1];
                $pairs[$key] = ($pairs[$key] ?? 0) + $freq;
            }
        }
        return $pairs;
    }

    /**
     * Apply a single merge (a, b) → merged to every word in the corpus.
     *
     * @param array<string,int> $corpus
     * @return array<string,int>
     */
    private function applyMerge(array $corpus, string $a, string $b, string $merged): array
    {
        $target    = $a . ' ' . $b;
        $newCorpus = [];
        foreach ($corpus as $word => $freq) {
            // str_replace is safe here: tokens never contain literal spaces
            // (GPT-2 byte-to-unicode maps space-byte to Ġ, not ' ').
            $newWord = str_replace($target, $merged, $word);
            $newCorpus[$newWord] = ($newCorpus[$newWord] ?? 0) + $freq;
        }
        return $newCorpus;
    }

    // ── GPT-2 byte-to-unicode mapping ─────────────────────────────────────────

    /**
     * Returns the canonical GPT-2 byte→unicode map used by all HF BPE tokenizers.
     *
     * Passthrough bytes (33-126, 161-172, 174-255) map to themselves.
     * All other bytes (0-32, 127-160, 173) map to U+0100 onward.
     *
     * @return array<int,string>  byte (int) → UTF-8 encoded unicode char
     */
    public static function buildByteToUnicode(): array
    {
        // Passthrough set (188 bytes that map to themselves)
        $passthrough = array_merge(
            range(33, 126),
            range(161, 172),
            range(174, 255)
        );
        sort($passthrough, SORT_NUMERIC);

        // Start with a complete copy; remap non-passthrough bytes to U+0100+
        $codepoints = array_fill(0, 256, 0);
        foreach ($passthrough as $b) {
            $codepoints[$b] = $b;
        }

        $shift = 0;
        for ($b = 0; $b < 256; $b++) {
            if (!in_array($b, $passthrough, true)) {
                $codepoints[$b] = 256 + $shift;
                $shift++;
            }
        }

        $map = [];
        for ($b = 0; $b < 256; $b++) {
            $map[$b] = mb_chr($codepoints[$b], 'UTF-8');
        }
        return $map;
    }
}
