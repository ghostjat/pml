<?php

declare(strict_types=1);

namespace Pml\Classic\Preprocess;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Transformer};

// ═══════════════════════════════════════════════════════════════════════════
//  OneHotEncoder — sklearn.preprocessing.OneHotEncoder
//
//  Maps integer categorical features to binary (0/1) indicator columns.
//
//  Input X [m, n_features_in] — each column contains float32 representations
//  of integer categories (e.g. 0.0, 1.0, 2.0).
//
//  Output [m, n_features_out] — a dense Float32 Tensor where:
//    - Each input column is replaced by |unique_values| binary columns.
//    - The indicator column for category c is 1.0 iff the input equals c.
//    - Categories are sorted in ascending order per column (sklearn default).
//
//  Memory layout of the output (row-major):
//    Column offsets are pre-computed in fit() and stored in $offsets_.
//    transform() iterates m×n_features_in samples and sets one bit per block.
//    All arithmetic is indexing — no BLAS call needed for the sparse scatter.
//    The output buffer is zeroed on construction (FFI::new zero-initialises).
//
//  handle_unknown behaviour:
//    'error'  — throw RuntimeException on unseen category (default)
//    'ignore' — leave the block all-zeros for unknown categories (sklearn
//               compatible silent drop; useful at predict-time)
// ═══════════════════════════════════════════════════════════════════════════

final class OneHotEncoder implements Estimator, Transformer
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Unique categories discovered per input feature, sorted ascending.
     * @var array<int, int[]>  Indexed [feature_index => [cat0, cat1, …]]
     */
    public readonly array $categories_;

    /**
     * Output column offset for each input feature in the transformed matrix.
     * @var int[]  $offsets_[$j] = index of the first output column for feature $j
     */
    public readonly array $offsets_;

    /** Total number of output columns after one-hot expansion. */
    public readonly int $n_features_out_;

    public readonly int $n_features_in_;

    /**
     * Reverse lookup: category value → output column index within each feature block.
     * Built in fit() to avoid O(k) array_search per element in transform().
     *
     * @var array<int, array<int, int>>  $catIndex_[$j][$cat] = position in block
     */
    private array $catIndex_;

    /**
     * @param string $handle_unknown  'error' or 'ignore' for unseen categories.
     */
    public function __construct(
        private readonly string $handle_unknown = 'error',
    ) {
        if (!in_array($handle_unknown, ['error', 'ignore'], true)) {
            throw new \InvalidArgumentException(
                "OneHotEncoder: handle_unknown must be 'error' or 'ignore', "
                . "got '{$handle_unknown}'."
            );
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Discover unique categories from training data.
     *
     * @param Tensor      $X  Integer-category matrix [n_samples, n_features]
     *                        or [n_samples] for single-feature input.
     * @param Tensor|null $y  Ignored.
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        // Support both 1-D [m] and 2-D [m, n] input — normalise to 2-D shape.
        $shape  = $X->shape;
        $is1d   = count($shape) === 1;
        $m      = $is1d ? $shape[0] : $shape[0];
        $n      = $is1d ? 1         : $shape[1];

        $categories = [];
        $catIndex   = [];
        $offsets    = [];
        $n_out      = 0;

        for ($j = 0; $j < $n; $j++) {
            // ── Collect unique integer values in column j ──────────────────
            //
            // We use an associative PHP array as a cheap hash-set.
            // Key = integer category value, value = true (presence marker).
            $seen = [];
            for ($i = 0; $i < $m; $i++) {
                $val        = $is1d
                    ? (int) round((float) $X->buffer[$i])
                    : (int) round((float) $X->buffer[$i * $n + $j]);
                $seen[$val] = true;
            }

            // Sort ascending to match sklearn's category ordering
            ksort($seen);
            $cats = array_keys($seen); // sorted int[]

            // Build O(1) reverse map: value → position in this block
            $revMap = [];
            foreach ($cats as $pos => $cat) {
                $revMap[$cat] = $pos;
            }

            $categories[$j] = $cats;
            $catIndex[$j]   = $revMap;
            $offsets[$j]    = $n_out;
            $n_out         += count($cats);
        }

        $this->categories_     = $categories;
        $this->catIndex_       = $catIndex;
        $this->offsets_        = $offsets;
        $this->n_features_in_  = $n;
        $this->n_features_out_ = $n_out;

        return $this;
    }

    // ── Transformer ────────────────────────────────────────────────────────

    /**
     * Encode X as a one-hot binary Float32 Tensor.
     *
     * Algorithm (O(m · n_features_in)):
     *   For each sample i, feature j:
     *     1. Read integer category value v = X[i, j].
     *     2. Look up position p in catIndex_[j][v]   (O(1) hash lookup).
     *     3. Set out[i, offsets_[j] + p] = 1.0.
     *   The output buffer is zero-initialised — all non-set entries stay 0.
     *
     * @param Tensor $X  [n_samples, n_features_in] or [n_samples] (1-D).
     * @return Tensor    [n_samples, n_features_out] one-hot encoded matrix.
     */
    public function transform(Tensor $X): Tensor
    {
        $this->checkFitted();

        $shape = $X->shape;
        $is1d  = count($shape) === 1;
        $m     = $shape[0];
        $n     = $is1d ? 1 : $shape[1];

        if ($n !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "OneHotEncoder::transform() expected {$this->n_features_in_} features, "
                . "got {$n}."
            );
        }

        $n_out = $this->n_features_out_;
        $out   = new Tensor([$m, $n_out]); // zeroed by FFI::new

        for ($i = 0; $i < $m; $i++) {
            $rowOut = $i * $n_out; // base offset in output row i

            for ($j = 0; $j < $n; $j++) {
                $val = $is1d
                    ? (int) round((float) $X->buffer[$i])
                    : (int) round((float) $X->buffer[$i * $n + $j]);

                if (isset($this->catIndex_[$j][$val])) {
                    $pos = $this->catIndex_[$j][$val];
                    $out->buffer[$rowOut + $this->offsets_[$j] + $pos] = 1.0;
                } elseif ($this->handle_unknown === 'error') {
                    throw new \RuntimeException(
                        "OneHotEncoder: unknown category {$val} in feature {$j}. "
                        . "Set handle_unknown='ignore' to suppress this error."
                    );
                }
                // 'ignore': block stays all-zeros → no action needed
            }
        }

        return $out;
    }

    /** Fit on $X then immediately encode it. */
    public function fit_transform(Tensor $X, ?Tensor $y = null): Tensor
    {
        return $this->fit($X, $y)->transform($X);
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    private function checkFitted(): void
    {
        if (!isset($this->categories_)) {
            throw new \RuntimeException('OneHotEncoder is not fitted. Call fit() first.');
        }
    }
}
