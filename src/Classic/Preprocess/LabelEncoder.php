<?php

declare(strict_types=1);

namespace Pml\Classic\Preprocess;

use Pml\Tensor;

// ═══════════════════════════════════════════════════════════════════════════
//  LabelEncoder — sklearn.preprocessing.LabelEncoder
//
//  Encodes a 1-D array of arbitrary labels (strings, integers, or mixed)
//  into contiguous integer codes 0 … n_classes−1, stored as a float32
//  Pml\Tensor ready for use by any classifier or target vector.
//
//  ── Encoding scheme ───────────────────────────────────────────────────────
//
//  The encoder is fit on a flat PHP array of labels.  Unique values are
//  collected, sorted (ksort — lexicographic for strings, numeric for ints),
//  and indexed 0 … n_classes−1.  This mirrors sklearn's use of np.unique()
//  which returns sorted unique values.
//
//    classes_ = ['bird', 'cat', 'dog']   →  codes: 0, 1, 2
//    classes_ = [1, 3, 7]               →  codes: 0, 1, 2
//
//  ── API ───────────────────────────────────────────────────────────────────
//
//  LabelEncoder does NOT implement Pml\Classic\Transformer because its
//  input is a PHP array<string|int|float>, not a Tensor.  This is the same
//  design choice made for CountVectorizer: the "text-to-Tensor" bridge sits
//  outside the Tensor-native Pipeline.
//
//  Typical usage before Pipeline fit:
//
//    $le = new LabelEncoder();
//    $y  = $le->fit_transform(['cat', 'dog', 'cat', 'bird']);
//    // $y is Tensor([1.0, 2.0, 1.0, 0.0]) — float32 ready for XGBClassifier
//
//    $pipeline->fit($X, $y);
//
//    $rawPred  = $pipeline->predict($Xtest);            // Tensor of codes
//    $labels   = $le->inverse_transform($rawPred);      // ['cat', ...]
//
//  ── Tensor bridge ─────────────────────────────────────────────────────────
//
//  transform() packs the integer codes into a float32 Tensor via a single
//  FFI::memcpy after building a PHP float[] array — one bulk C copy instead
//  of n individual FFI writes.
//
//  inverse_transform() reads the Tensor buffer element-by-element (pure PHP,
//  O(n)) and looks up each integer code in the $codeToLabel_ reverse map.
//
//  ── Type handling ─────────────────────────────────────────────────────────
//
//  PHP arrays may contain strings, integers, or floats as labels.
//  fit() normalises all values to string for consistent identity comparison:
//
//    - Integers 0, 1, 2 and floats 0.0, 1.0, 2.0 are treated as distinct
//      unless PHP's string cast collapses them (e.g. (string)0 === '0').
//    - Strings are used as-is.
//
//  This mirrors sklearn's behaviour: any type that casts to a unique string
//  is treated as a distinct class.  When labels are pure integers, the
//  encoded classes_ array holds int values; for strings, string values.
// ═══════════════════════════════════════════════════════════════════════════

final class LabelEncoder
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Sorted unique labels discovered at fit() time.
     * Ordering is ksort-ascending: lexicographic for strings, numeric for ints.
     * @var array<int|string>
     */
    public readonly array $classes_;

    /** Number of unique classes. */
    public readonly int $n_classes_;

    // ── Internal reverse maps ─────────────────────────────────────────────

    /**
     * Forward map: string(label) → integer code (0…n_classes-1).
     * Keys are the string-cast label values for uniform lookup.
     * @var array<string, int>
     */
    private readonly array $labelToCode_;

    /**
     * Reverse map: integer code → original label value.
     * @var array<int, int|string>
     */
    private readonly array $codeToLabel_;

    // ── Constructor ───────────────────────────────────────────────────────

    // No constructor parameters (stateless before fit, like sklearn).

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Learn the set of unique labels and their integer code assignments.
     *
     * @param array<int|string|float> $y  1-D array of raw labels.
     */
    public function fit(array $y): static
    {
        if ($y === []) {
            throw new \InvalidArgumentException('LabelEncoder::fit() received an empty label array.');
        }

        // ── Collect unique labels ──────────────────────────────────────────
        //
        // Use a hash set keyed by the original value (PHP's native array key
        // conversion: int stays int, string stays string, float → int or string).
        // Then sort the resulting key set for canonical ordering.
        $seen = [];
        foreach ($y as $label) {
            $seen[$label] = true;
        }

        // ksort: for all-integer keys PHP sorts numerically;
        //        for string keys it sorts lexicographically.
        //        Mixed-type keys follow PHP's internal comparison rules.
        ksort($seen);
        $classes = array_keys($seen);

        // Build forward and reverse maps
        $labelToCode = [];
        $codeToLabel = [];
        foreach ($classes as $code => $label) {
            $labelToCode[(string) $label] = $code;
            $codeToLabel[$code]           = $label;
        }

        $this->classes_     = $classes;
        $this->n_classes_   = count($classes);
        $this->labelToCode_ = $labelToCode;
        $this->codeToLabel_ = $codeToLabel;

        return $this;
    }

    /**
     * Encode labels to integer codes.
     *
     * Packs the code sequence into a float32 Tensor via a single
     * FFI::memcpy from a PHP pack() call — O(n) PHP plus one bulk C copy.
     *
     * @param  array<int|string|float> $y  1-D array of labels (must all be in classes_).
     * @return Tensor                       Float32 Tensor of shape [n] with integer codes.
     */
    public function transform(array $y): Tensor
    {
        $this->checkFitted();

        $n    = count($y);
        $map  = $this->labelToCode_;
        $codes = [];

        foreach ($y as $label) {
            $key = (string) $label;
            if (!isset($map[$key])) {
                throw new \RuntimeException(
                    "LabelEncoder::transform(): unseen label '{$label}'. "
                    . 'Call fit() on a corpus that contains all expected labels.'
                );
            }
            $codes[] = (float) $map[$key];
        }

        // Pack into float32 and memcpy into a Tensor in one shot
        $out  = new Tensor([$n]);
        $pack = pack('f*', ...$codes);
        \FFI::memcpy($out->buffer, $pack, $n * 4);

        return $out;
    }

    /**
     * Fit then immediately transform — convenience shortcut.
     *
     * @param  array<int|string|float> $y
     * @return Tensor                      [n] float32 code Tensor
     */
    public function fit_transform(array $y): Tensor
    {
        return $this->fit($y)->transform($y);
    }

    /**
     * Convert integer code Tensor back to original labels.
     *
     * Each float in the Tensor is rounded to the nearest int and used as
     * a code index into $this->codeToLabel_.  Codes outside [0, n_classes)
     * throw a RuntimeException.
     *
     * @param  Tensor $y  Tensor of shape [n] — float codes from predict() etc.
     * @return array<int|string>  Original label values.
     */
    public function inverse_transform(Tensor $y): array
    {
        $this->checkFitted();

        if (count($y->shape) !== 1) {
            throw new \InvalidArgumentException(
                'LabelEncoder::inverse_transform() expects a 1-D Tensor [n_samples].'
            );
        }

        $n     = $y->size;
        $map   = $this->codeToLabel_;
        $nCls  = $this->n_classes_;
        $out   = [];

        for ($i = 0; $i < $n; $i++) {
            $code = (int) round((float) $y->buffer[$i]);
            if ($code < 0 || $code >= $nCls) {
                throw new \RuntimeException(
                    "LabelEncoder::inverse_transform(): code {$code} is out of range "
                    . "[0, {$nCls})."
                );
            }
            $out[] = $map[$code];
        }

        return $out;
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private function checkFitted(): void
    {
        if (!isset($this->classes_)) {
            throw new \RuntimeException(
                'LabelEncoder is not fitted. Call fit() first.'
            );
        }
    }
}
