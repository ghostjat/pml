<?php

declare(strict_types=1);

namespace Pml\Classic\ModelSelection;

use Pml\Tensor;

// ═══════════════════════════════════════════════════════════════════════════
//  KFold — sklearn.model_selection.KFold
//
//  K-Fold cross-validator.  Splits n_samples indices into n_splits
//  consecutive (non-overlapping) folds; each fold is used exactly once
//  as the test set while the remaining k-1 folds form the training set.
//
//  ── Fold Sizes ───────────────────────────────────────────────────────────
//
//  When n_samples is not divisible by n_splits, the first (n % n_splits)
//  folds each contain one extra sample — matching sklearn's behaviour:
//
//    foldSizes[i] = ⌊n / n_splits⌋ + (i < n % n_splits ? 1 : 0)
//
//  ── Shuffle ──────────────────────────────────────────────────────────────
//
//  If shuffle=true, the full index array [0 … n-1] is shuffled with mt_rand
//  (seeded by random_state if non-null) before splitting into folds.  The
//  fold boundaries are then defined in terms of the shuffled indices.
//
//  ── split() ──────────────────────────────────────────────────────────────
//
//  Returns a PHP Generator that yields [$train_indices, $test_indices] on
//  each iteration, where both are flat int[] arrays.  Lazily generating
//  each fold avoids materialising all splits at once.
//
//  The generator pattern mirrors sklearn's split() which returns an iterator
//  of (train_index, test_index) pairs.
// ═══════════════════════════════════════════════════════════════════════════

final class KFold
{
    /**
     * @param int      $n_splits     Number of folds.  Must be ≥ 2.
     * @param bool     $shuffle      Whether to shuffle before splitting.
     * @param int|null $random_state RNG seed for shuffle (null = system entropy).
     */
    public function __construct(
        private readonly int  $n_splits     = 5,
        private readonly bool $shuffle      = false,
        private readonly ?int $random_state = null,
    ) {
        if ($n_splits < 2) {
            throw new \InvalidArgumentException("KFold: n_splits must be ≥ 2, got {$n_splits}.");
        }
    }

    /**
     * Generate (train_indices, test_indices) pairs for each fold.
     *
     * @param Tensor $X  The dataset — only its first dimension (n_samples) is used.
     *
     * @return \Generator<int, array{0: int[], 1: int[]}>
     *   Yields [$train_indices, $test_indices] for each fold.
     *   Both are flat PHP int[] arrays.
     *
     * @throws \InvalidArgumentException If n_samples < n_splits.
     */
    public function split(Tensor $X): \Generator
    {
        $n = $X->shape[0];

        if ($n < $this->n_splits) {
            throw new \InvalidArgumentException(
                "KFold: n_samples={$n} must be ≥ n_splits={$this->n_splits}."
            );
        }

        // ── Build (and optionally shuffle) the index array ─────────────────
        $indices = range(0, $n - 1);

        if ($this->shuffle) {
            if ($this->random_state !== null) {
                mt_srand($this->random_state);
            }
            shuffle($indices);   // in-place, uses mt_rand
        }

        // ── Compute fold sizes ─────────────────────────────────────────────
        //
        // Base size = ⌊n / n_splits⌋.  The first (n % n_splits) folds are
        // one element larger to absorb the remainder.
        $baseSize  = intdiv($n, $this->n_splits);
        $remainder = $n % $this->n_splits;

        $foldSizes = array_fill(0, $this->n_splits, $baseSize);
        for ($i = 0; $i < $remainder; $i++) {
            $foldSizes[$i]++;
        }

        // ── Yield one fold at a time ───────────────────────────────────────
        //
        // $start tracks the beginning of the current test fold in $indices.
        // The training set is the complement: indices before and after the fold.
        $start = 0;
        for ($fold = 0; $fold < $this->n_splits; $fold++) {
            $foldSize = $foldSizes[$fold];
            $stop     = $start + $foldSize;

            // Test fold: a contiguous slice of the (shuffled) index array
            $testIdx  = array_slice($indices, $start, $foldSize);

            // Train fold: everything before + everything after the test slice
            $trainIdx = array_merge(
                array_slice($indices, 0, $start),
                array_slice($indices, $stop)
            );

            yield [$trainIdx, $testIdx];

            $start = $stop;
        }
    }

    /** Returns the configured number of folds. */
    public function get_n_splits(): int
    {
        return $this->n_splits;
    }
}
