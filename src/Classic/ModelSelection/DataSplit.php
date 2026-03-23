<?php

declare(strict_types=1);

namespace Pml\Classic\ModelSelection;

use Pml\{Tensor, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  DataSplit — sklearn.model_selection.train_test_split
//
//  Randomly shuffle indices and partition the dataset into a training set
//  and a test set without ever copying data to PHP arrays.
//
//  Algorithm:
//    1. Generate indices [0, n_samples) and shuffle them with mt_rand.
//    2. Split at floor(n_samples * (1 − test_size)).
//    3. Gather rows from X using cblas_scopy (O(d) per row), O(1) per y element.
//
//  Memory:
//    Four new Tensors are allocated (X_train, X_test, y_train, y_test).
//    The original X and y are never modified.
//
//  API note:
//    In sklearn this is a module-level function.  In PHP it lives as a static
//    method on a utility class — identical call semantics once imported.
// ═══════════════════════════════════════════════════════════════════════════

final class DataSplit
{
    /**
     * Split arrays into random train and test subsets.
     *
     * @param Tensor $X           Feature matrix [n_samples, n_features]
     * @param Tensor $y           Target vector   [n_samples]
     * @param float  $test_size   Fraction of samples for the test set (0 < test_size < 1).
     *                            Default 0.25 → 25% test, 75% train.
     * @param int    $random_state mt_srand() seed.  0 = do not seed (non-deterministic).
     *
     * @return array{0:Tensor, 1:Tensor, 2:Tensor, 3:Tensor}
     *   [X_train, X_test, y_train, y_test]
     *
     * @throws \InvalidArgumentException When X is not 2-D or test_size is out of (0,1).
     */
    public static function train_test_split(
        Tensor $X,
        Tensor $y,
        float  $test_size    = 0.25,
        int    $random_state = 0,
    ): array {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException(
                'train_test_split: X must be 2-D [n_samples, n_features], '
                . 'got shape [' . implode(', ', $X->shape) . '].'
            );
        }

        if ($test_size <= 0.0 || $test_size >= 1.0) {
            throw new \InvalidArgumentException(
                "train_test_split: test_size must be in (0, 1), got {$test_size}."
            );
        }

        [$n, $d] = $X->shape;
        $blas    = BlasEngine::get()->ffi;

        if ($random_state !== 0) {
            mt_srand($random_state);
        }

        // ── 1. Build and shuffle index array ──────────────────────────────
        //
        // array_keys(array_fill(...)) creates an integer-indexed PHP array.
        // shuffle() uses mt_rand internally — seeded above if requested.
        $indices = range(0, $n - 1);
        shuffle($indices);

        // ── 2. Compute split boundary ─────────────────────────────────────
        //
        // n_test = max(1, round(n * test_size))  — at least 1 test sample
        // n_train = n − n_test
        $n_test  = max(1, (int) round($n * $test_size));
        $n_train = $n - $n_test;

        $trainIdx = array_slice($indices, 0, $n_train);
        $testIdx  = array_slice($indices, $n_train);

        // ── 3. Allocate output Tensors ────────────────────────────────────
        $X_train = new Tensor([$n_train, $d]);
        $X_test  = new Tensor([$n_test,  $d]);
        $y_train = new Tensor([$n_train]);
        $y_test  = new Tensor([$n_test]);

        // ── 4. Gather rows from X via cblas_scopy ─────────────────────────
        //
        // cblas_scopy(d, src_ptr, 1, dst_ptr, 1) copies one row of d floats.
        // FFI::cast('float*', FFI::addr(buf[offset])) gives a pointer to the
        // start of row $src in the flat 1-D C buffer.
        foreach ($trainIdx as $out => $src) {
            $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$src * $d]));
            $dstPtr = \FFI::cast('float*', \FFI::addr($X_train->buffer[$out * $d]));
            $blas->cblas_scopy($d, $srcPtr, 1, $dstPtr, 1);
            $y_train->buffer[$out] = $y->buffer[$src];
        }

        foreach ($testIdx as $out => $src) {
            $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$src * $d]));
            $dstPtr = \FFI::cast('float*', \FFI::addr($X_test->buffer[$out * $d]));
            $blas->cblas_scopy($d, $srcPtr, 1, $dstPtr, 1);
            $y_test->buffer[$out] = $y->buffer[$src];
        }

        return [$X_train, $X_test, $y_train, $y_test];
    }
}
