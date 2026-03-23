<?php

declare(strict_types=1);

namespace Pml\Classic\Compose;

use Pml\Tensor;

// ═══════════════════════════════════════════════════════════════════════════
//  ColumnTransformer — sklearn.compose.ColumnTransformer
//
//  Routes different columns of a raw PHP 2D array to different transformers,
//  then horizontally concatenates the outputs into a single Float32 Tensor.
//
//  This is the "Tensor bridge" for mixed-type tabular data: it accepts the
//  raw 2D PHP array returned by DataLoader::load_csv() when the dataset
//  contains string columns (e.g. Sex, Embarked, Name) and produces a
//  pure-numeric Tensor suitable for any Pml estimator.
//
//  ── Transformer specification (sklearn-identical) ─────────────────────────
//
//  Each entry in $transformers is a 3-element tuple:
//    ['name', $transformer, ['col1', 'col2', ...]]
//
//    'name'         Any unique string identifier for this step.
//    $transformer   A Pml\Classic\Transformer instance, OR one of:
//                     'drop'        Discard these columns entirely.
//                     'passthrough' Include these columns as floats (no transform).
//    ['col1', ...]  Column selectors — either column name strings (requires
//                   feature_names to be provided) or 0-based integer indices.
//
//  ── String columns and OneHotEncoder ─────────────────────────────────────
//
//  When a transformer group's columns contain string values (e.g. 'male',
//  'female'), ColumnTransformer builds an internal alphabetically-sorted
//  vocabulary (string → integer code) and passes integer-coded floats to the
//  underlying transformer.  This mirrors sklearn's native string handling in
//  OneHotEncoder.
//
//  Missing values (null / empty string) in string columns are encoded as
//  -1.0 so that OneHotEncoder's handle_unknown='ignore' path leaves the
//  corresponding output block as all-zeros.
//
//  ── Missing values in numeric columns ────────────────────────────────────
//
//  Null / empty-string values in numeric columns are converted to IEEE 754
//  NaN before packing into the Tensor.  SimpleImputer then detects and fills
//  them using the learned per-column statistic (mean / median / constant).
//
//  ── Output layout ────────────────────────────────────────────────────────
//
//  Transformer outputs are concatenated left-to-right in declaration order:
//    [transformer_0_output | transformer_1_output | passthrough_cols]
//
//  If remainder='passthrough', unassigned columns are appended at the right
//  in their original order.
// ═══════════════════════════════════════════════════════════════════════════

final class ColumnTransformer
{
    // ── Fitted state ──────────────────────────────────────────────────────

    /**
     * Maps column name → 0-based index in the raw array rows.
     * Built from $feature_names in fit().
     *
     * @var array<string, int>
     */
    private array $colIndex_ = [];

    /**
     * Per-transformer string vocabularies (for OHE-style string columns).
     *
     * $vocabs_[$name][$relColIdx][$string] = int_code
     *
     * Built in fit() for any transformer group whose columns contain strings.
     * Stored so transform() can reproduce the same encoding deterministically.
     *
     * @var array<string, array<int, array<string, int>>>
     */
    private array $vocabs_ = [];

    /**
     * Column indices to append as-is when remainder='passthrough'.
     * Populated in fit(); indices not covered by any transformer entry.
     *
     * @var int[]
     */
    private array $passthroughCols_ = [];

    /**
     * Number of input features (columns) seen during fit().
     * Zero before fit(); used to detect unfitted state.
     */
    private int $n_features_in_ = 0;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param array  $transformers   Ordered list of transformer entries:
     *                               [['name', transformer, ['col1', ...]], ...]
     *                               $transformer may be a Transformer instance,
     *                               'drop', or 'passthrough'.
     * @param string $remainder      Handling for columns not assigned to any
     *                               transformer entry:
     *                               'drop'        — discard them (default, sklearn)
     *                               'passthrough' — append as float columns
     * @param array  $feature_names  Column names parallel to each row's indices.
     *                               Required when $transformers use string column names.
     *                               Pass $bunch['feature_names'] from DataLoader.
     */
    public function __construct(
        private readonly array  $transformers,
        private readonly string $remainder     = 'drop',
        private readonly array  $feature_names = [],
    ) {
        if (!in_array($remainder, ['drop', 'passthrough'], true)) {
            throw new \InvalidArgumentException(
                "ColumnTransformer: remainder must be 'drop' or 'passthrough', got '{$remainder}'."
            );
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Fit all transformers to the raw feature array.
     *
     * For each transformer entry, the specified columns are extracted,
     * string values are label-encoded (if needed), the data is packed into
     * a Float32 Tensor, and the transformer's fit() is called.
     *
     * @param array       $X  Raw 2D PHP array [n_samples][n_cols].
     *                        Values may be float, string, null, or empty string.
     * @param Tensor|null $y  Passed through to transformers (usually ignored).
     */
    public function fit(array $X, ?Tensor $y = null): static
    {
        if (count($X) === 0) {
            throw new \RuntimeException('ColumnTransformer: X must be non-empty.');
        }

        $this->n_features_in_ = count($X[0]);

        // ── Build column name → index map ──────────────────────────────
        $this->colIndex_ = [];
        foreach ($this->feature_names as $idx => $name) {
            $this->colIndex_[$name] = (int) $idx;
        }

        // ── Fit each transformer entry ─────────────────────────────────
        $coveredIndices = [];

        foreach ($this->transformers as $entry) {
            [$name, $transformer, $colSelectors] = $entry;

            $indices = $this->resolveColumns($colSelectors);
            foreach ($indices as $idx) {
                $coveredIndices[$idx] = true;
            }

            if ($transformer === 'drop' || $transformer === 'passthrough') {
                continue; // nothing to fit
            }

            $n        = count($X);
            $k        = count($indices);
            $colData  = $this->extractColumns($X, $indices);

            if ($this->hasStringValues($colData)) {
                // Build vocabulary: string values → int codes (alphabetical order)
                $vocabs               = $this->buildVocabs($colData, $k);
                $this->vocabs_[$name] = $vocabs;
                $tensor               = $this->encodeWithVocabs($colData, $vocabs, $n, $k);
            } else {
                // Numeric: null/'' → NaN for downstream imputers
                $tensor = $this->numericToTensor($colData, $n, $k);
            }

            $transformer->fit($tensor, $y);
        }

        // ── Determine passthrough remainder columns ─────────────────────
        $this->passthroughCols_ = [];
        if ($this->remainder === 'passthrough') {
            for ($i = 0; $i < $this->n_features_in_; $i++) {
                if (!isset($coveredIndices[$i])) {
                    $this->passthroughCols_[] = $i;
                }
            }
        }

        return $this;
    }

    /**
     * Apply all fitted transformers and concatenate results into a Tensor.
     *
     * Output layout (left to right):
     *   [transformer_0_output | transformer_1_output | … | passthrough_cols]
     *
     * @param array $X  Raw 2D PHP array [n_samples][n_cols]
     * @return Tensor   Dense Float32 Tensor [n_samples, n_features_out]
     */
    public function transform(array $X): Tensor
    {
        if ($this->n_features_in_ === 0) {
            throw new \RuntimeException('ColumnTransformer is not fitted. Call fit() first.');
        }

        $n             = count($X);
        $outputTensors = [];

        foreach ($this->transformers as $entry) {
            [$name, $transformer, $colSelectors] = $entry;
            $indices = $this->resolveColumns($colSelectors);
            $k       = count($indices);

            if ($transformer === 'drop') {
                continue;
            }

            $colData = $this->extractColumns($X, $indices);

            if ($transformer === 'passthrough') {
                // Declared passthrough — include as floats (null/'' → 0.0)
                $outputTensors[] = $this->numericToTensor($colData, $n, $k);
                continue;
            }

            // Apply fitted transformer (with same encoding as fit())
            if (isset($this->vocabs_[$name])) {
                $tensor = $this->encodeWithVocabs($colData, $this->vocabs_[$name], $n, $k);
            } else {
                $tensor = $this->numericToTensor($colData, $n, $k);
            }

            $outputTensors[] = $transformer->transform($tensor);
        }

        // ── Append remainder passthrough columns ───────────────────────
        if (!empty($this->passthroughCols_)) {
            $passData        = $this->extractColumns($X, $this->passthroughCols_);
            $outputTensors[] = $this->numericToTensor($passData, $n, count($this->passthroughCols_));
        }

        if (empty($outputTensors)) {
            throw new \RuntimeException(
                'ColumnTransformer: all columns were dropped — no output produced.'
            );
        }

        return $this->hstack($outputTensors, $n);
    }

    /**
     * Convenience: fit($X, $y) → transform($X).
     * Equivalent to (and equally efficient as) two separate calls.
     *
     * @param array       $X  Raw 2D PHP array [n_samples][n_cols]
     * @param Tensor|null $y  Passed through to fit()
     * @return Tensor         Dense Float32 Tensor [n_samples, n_features_out]
     */
    public function fit_transform(array $X, ?Tensor $y = null): Tensor
    {
        return $this->fit($X, $y)->transform($X);
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * Resolve column selectors (string names or integer indices) → int indices.
     *
     * @param  array<int|string> $selectors
     * @return int[]
     */
    private function resolveColumns(array $selectors): array
    {
        $indices = [];
        foreach ($selectors as $sel) {
            if (is_int($sel)) {
                $indices[] = $sel;
            } elseif (is_string($sel)) {
                if (!isset($this->colIndex_[$sel])) {
                    throw new \InvalidArgumentException(
                        "ColumnTransformer: column '{$sel}' not found. "
                        . 'Available: ' . implode(', ', array_keys($this->colIndex_))
                    );
                }
                $indices[] = $this->colIndex_[$sel];
            } else {
                throw new \InvalidArgumentException(
                    'ColumnTransformer: column selectors must be int or string.'
                );
            }
        }
        return $indices;
    }

    /**
     * Extract the specified column indices from all rows of X.
     *
     * @param  array $X       Raw 2D PHP array [n_samples][n_cols]
     * @param  int[] $indices Column indices to extract
     * @return array          2D array [n_samples][k] of raw values
     */
    private function extractColumns(array $X, array $indices): array
    {
        $result = [];
        foreach ($X as $row) {
            $extracted = [];
            foreach ($indices as $idx) {
                $extracted[] = $row[$idx] ?? null;
            }
            $result[] = $extracted;
        }
        return $result;
    }

    /**
     * Return true if any cell in the 2D column array is a non-numeric string.
     */
    private function hasStringValues(array $colData): bool
    {
        foreach ($colData as $row) {
            foreach ($row as $v) {
                if (is_string($v) && $v !== '' && !is_numeric($v)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Build a per-column vocabulary: unique non-empty string values → int code.
     *
     * Vocabulary is sorted alphabetically (matching sklearn's default category
     * ordering in OneHotEncoder).
     *
     * @param  array $colData  2D column data [n][k]
     * @param  int   $k        Number of columns
     * @return array<int, array<string, int>>  $vocab[$colRelIdx][$str] = intCode
     */
    private function buildVocabs(array $colData, int $k): array
    {
        $vocabs = [];
        for ($j = 0; $j < $k; $j++) {
            $seen = [];
            foreach ($colData as $row) {
                $v = $row[$j];
                if ($v !== null && $v !== '') {
                    $seen[(string) $v] = true;
                }
            }
            ksort($seen); // alphabetical → reproducible, sklearn-compatible ordering
            $code  = 0;
            $vocab = [];
            foreach (array_keys($seen) as $str) {
                $vocab[$str] = $code++;
            }
            $vocabs[$j] = $vocab;
        }
        return $vocabs;
    }

    /**
     * Encode string column data to integer-coded float Tensor via pre-built vocabs.
     *
     * Missing values (null / empty string) and unknown strings are encoded as
     * -1.0.  Since valid integer codes start at 0, -1 is always out-of-vocabulary
     * and OneHotEncoder's handle_unknown='ignore' will leave that output block
     * as all-zeros — the correct behaviour for a missing categorical value.
     *
     * @param  array                             $colData  2D [n][k] raw values
     * @param  array<int, array<string, int>>    $vocabs   Built by buildVocabs()
     * @param  int                               $n        Sample count
     * @param  int                               $k        Column count
     * @return Tensor                                       [n, k] integer-coded floats
     */
    private function encodeWithVocabs(array $colData, array $vocabs, int $n, int $k): Tensor
    {
        $flat = [];
        foreach ($colData as $row) {
            for ($j = 0; $j < $k; $j++) {
                $v = $row[$j];
                if ($v === null || $v === '') {
                    $flat[] = -1.0; // missing → OHE 'ignore' → all-zeros block
                } elseif (isset($vocabs[$j][(string) $v])) {
                    $flat[] = (float) $vocabs[$j][(string) $v];
                } else {
                    $flat[] = -1.0; // unknown category → same treatment
                }
            }
        }
        return Tensor::fromArray($flat, [$n, $k]);
    }

    /**
     * Pack numeric column data into a Float32 Tensor.
     *
     * Null / empty-string values become IEEE 754 NaN so that SimpleImputer
     * can detect and fill them with the learned statistic.
     *
     * @param  array $colData  2D [n][k] raw values
     * @param  int   $n        Sample count
     * @param  int   $k        Column count
     * @return Tensor          [n, k] float32 tensor, NaN where values were missing
     */
    private function numericToTensor(array $colData, int $n, int $k): Tensor
    {
        $flat = [];
        foreach ($colData as $row) {
            for ($j = 0; $j < $k; $j++) {
                $v = $row[$j];
                if ($v === null || $v === '') {
                    $flat[] = NAN;
                } else {
                    $flat[] = (float) $v;
                }
            }
        }
        return Tensor::fromArray($flat, [$n, $k]);
    }

    /**
     * Horizontally stack multiple [n, k_i] Tensors into one [n, Σk_i] Tensor.
     *
     * Iterates row-by-row, copying each transformer's output columns in order.
     * This is a O(n · Σk_i) PHP loop — acceptable for preprocessing that runs
     * once before training.
     *
     * @param Tensor[] $tensors  All must be 2-D with the same first dimension n.
     * @param int      $n        Shared row count.
     * @return Tensor            [n, total_cols] concatenated output.
     */
    private function hstack(array $tensors, int $n): Tensor
    {
        $totalCols = 0;
        foreach ($tensors as $t) {
            $totalCols += $t->shape[1];
        }

        $flat = [];
        for ($i = 0; $i < $n; $i++) {
            foreach ($tensors as $t) {
                $tCols = $t->shape[1];
                for ($j = 0; $j < $tCols; $j++) {
                    $flat[] = (float) $t->buffer[$i * $tCols + $j];
                }
            }
        }

        return Tensor::fromArray($flat, [$n, $totalCols]);
    }
}
