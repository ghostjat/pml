<?php

declare(strict_types=1);

namespace Pml\Training;

// ═══════════════════════════════════════════════════════════════════════════
//  DataLoader
//
//  Sliding-window batch sampler for next-token prediction with an optional
//  train / validation split.
//
//  Next-token prediction framing (causal language modelling):
//    Given the corpus as a flat sequence of integer token IDs, every
//    training example is a contiguous window of length seqLen:
//
//      X = [t_i,   t_{i+1}, ..., t_{i+seqLen-1}]   ← context
//      Y = [t_{i+1}, t_{i+2}, ..., t_{i+seqLen}]   ← target (shifted +1)
//
//  Train / validation split:
//    The corpus is divided at floor(corpusLen * splitRatio):
//      - Training windows are drawn from [0, splitAt)
//      - Validation windows are drawn from [splitAt, corpusLen)
//    getBatch($type) selects which region to sample from.
//    Both splits use the same sliding-window + random-start logic.
//
//  Corpus indexing:
//    Valid start positions within a region of length L:
//      [regionOffset, regionOffset + L − seqLen − 1]  (inclusive)
//    datasetSize() / valSize() return these counts.
// ═══════════════════════════════════════════════════════════════════════════

final class DataLoader
{
    /** First token index past the training region (= start of val region). */
    private readonly int $splitAt;

    /** Number of valid start positions in the training split. */
    private readonly int $trainN;

    /** Number of valid start positions in the validation split. */
    private readonly int $valN;

    /**
     * @param int[]  $tokens      Flat array of integer token IDs representing
     *                             the entire tokenised corpus.  Must have at
     *                             least $seqLen + 1 elements.
     * @param int    $seqLen      Length of each context / target window.
     * @param int    $batchSize   Number of random windows returned per getBatch() call.
     * @param float  $splitRatio  Fraction of corpus reserved for training.
     *                             Must be in (0, 1].  Default 1.0 = all data is
     *                             training; there is no validation set.
     *                             E.g. 0.9 → 90% train, 10% val.
     *
     * @throws \InvalidArgumentException When corpus is too small, or arguments are out of range.
     */
    public function __construct(
        private readonly array $tokens,
        private readonly int   $seqLen,
        private readonly int   $batchSize = 1,
        private readonly float $splitRatio = 1.0,
    ) {
        $corpusLen = count($this->tokens);

        if ($this->seqLen < 1) {
            throw new \InvalidArgumentException("DataLoader: seqLen must be >= 1, got {$seqLen}.");
        }

        if ($this->batchSize < 1) {
            throw new \InvalidArgumentException("DataLoader: batchSize must be >= 1, got {$batchSize}.");
        }

        if ($splitRatio <= 0.0 || $splitRatio > 1.0) {
            throw new \InvalidArgumentException(
                "DataLoader: splitRatio must be in (0, 1], got {$splitRatio}."
            );
        }

        if ($corpusLen < $this->seqLen + 1) {
            throw new \InvalidArgumentException(
                "DataLoader: corpus ({$corpusLen} tokens) is too small for seqLen={$seqLen}. "
                . 'Need at least seqLen + 1 tokens (one extra for the shifted target).'
            );
        }

        // ── Split the corpus ──────────────────────────────────────────────
        //
        // Training region: tokens[0 .. splitAt-1]  (length = splitAt)
        // Validation region: tokens[splitAt .. corpusLen-1] (length = corpusLen - splitAt)
        //
        // For a region of length L, the number of valid sliding windows is
        // L − seqLen (so that both X and Y windows fit inside the region).
        //
        $this->splitAt = (int) floor($corpusLen * $splitRatio);

        // Training windows: start in [0, splitAt − seqLen − 1]
        // Count = splitAt − seqLen  (but at least 0 to avoid negative counts)
        $this->trainN  = max(0, $this->splitAt - $this->seqLen);

        // Validation windows: start in [splitAt, corpusLen − seqLen − 1]
        // Count = (corpusLen − splitAt) − seqLen
        $this->valN    = max(0, ($corpusLen - $this->splitAt) - $this->seqLen);
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /**
     * Sample a mini-batch of (X, Y) pairs from the requested split.
     *
     * @param string $type  'train' (default) samples from the training region;
     *                       'val' samples from the validation region.
     *
     * @return array{0: int[][], 1: int[][]}
     *   [$xBatch, $yBatch] — each is an array of $batchSize int[] of length $seqLen.
     *
     * @throws \RuntimeException If the requested split has no valid windows.
     */
    public function getBatch(string $type = 'train'): array
    {
        if ($type === 'val') {
            $regionOffset = $this->splitAt;
            $n            = $this->valN;

            if ($n < 1) {
                throw new \RuntimeException(
                    'DataLoader: validation set has no valid windows. '
                    . 'Use splitRatio < 1.0 and ensure (1-splitRatio)*corpus >= seqLen+1.'
                );
            }
        } else {
            $regionOffset = 0;
            $n            = $this->trainN;

            if ($n < 1) {
                throw new \RuntimeException('DataLoader: training set has no valid windows.');
            }
        }

        $xBatch = [];
        $yBatch = [];

        for ($b = 0; $b < $this->batchSize; $b++) {
            // Draw a uniformly random position within the split region.
            $start = $regionOffset + mt_rand(0, $n - 1);

            // X: context window — tokens[start .. start + seqLen)
            $xBatch[] = array_slice($this->tokens, $start,     $this->seqLen);

            // Y: target window — tokens[start+1 .. start + seqLen]
            $yBatch[] = array_slice($this->tokens, $start + 1, $this->seqLen);
        }

        return [$xBatch, $yBatch];
    }

    // ── Metadata ───────────────────────────────────────────────────────────

    /** Number of valid start positions in the training split (= trainLen − seqLen). */
    public function datasetSize(): int { return $this->trainN; }

    /** Number of valid start positions in the validation split (= valLen − seqLen). */
    public function valSize(): int { return $this->valN; }

    /** Total tokens in the corpus. */
    public function corpusLength(): int { return count($this->tokens); }

    /** Token index at which the validation split begins. */
    public function splitAt(): int { return $this->splitAt; }

    /** Sequence length (window size) for each sample. */
    public function seqLen(): int { return $this->seqLen; }

    /** Number of sequences returned per getBatch() call. */
    public function batchSize(): int { return $this->batchSize; }
}
