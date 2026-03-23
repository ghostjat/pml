<?php

declare(strict_types=1);

namespace Pml\Classic\Ensemble;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};
use Pml\Classic\Tree\DecisionTreeClassifier;

// ═══════════════════════════════════════════════════════════════════════════
//  RandomForestClassifier — sklearn.ensemble.RandomForestClassifier
//
//  Bootstrap aggregation (bagging) of DecisionTreeClassifier instances.
//  Each tree is fitted on a bootstrap sample of the training data and uses
//  a random subset of features at each split (controlled by max_features).
//
//  ── Bootstrap Sampling ───────────────────────────────────────────────────
//
//  For each of the n_estimators trees:
//    1. Draw n_samples row indices uniformly at random WITH replacement
//       using PHP's mt_rand().
//    2. Build the bootstrap feature matrix X_boot [n, d] and label vector
//       y_boot [n] by copying rows from X and y.
//    3. Row copy is done via cblas_scopy(d, srcPtr, 1, dstPtr, 1) —
//       one BLAS-1 call per bootstrap row, zero PHP element-level loops.
//    4. Label copy is a direct buffer index assignment (scalar, no BLAS).
//
//  Each tree receives a distinct random_state seed (base + tree index) so
//  its internal feature-subset selection is independently seeded.
//
//  ── Probability Aggregation ──────────────────────────────────────────────
//
//  predict_proba():
//    1. Call predict_proba(X) on every tree → [n_samples, tree_n_classes] Tensor.
//    2. Because bootstrap samples may not contain every class, individual trees
//       may have a different (subset) classes_ than the forest.  Probabilities
//       are scattered into the forest-level class positions using an O(1)
//       forest_class_pos lookup built from the forest's classes_ array.
//    3. After accumulating all trees' contributions, cblas_sscal divides
//       the accumulated matrix by n_estimators in a single BLAS-1 call.
//
//  predict():
//    Argmax of predict_proba() — the class index with the highest average
//    probability across all trees.
//
//  ── Class Alignment ──────────────────────────────────────────────────────
//
//  The forest's classes_ is computed from the FULL training set (not from
//  individual bootstrap trees), ensuring all labels are covered even if a
//  particular bootstrap sample omits a rare class.  Each tree's predict_proba
//  output is mapped into the forest's wider class space by scattering
//  tree->classes_[c] → forestClassPos[label] before accumulation.
// ═══════════════════════════════════════════════════════════════════════════

final class RandomForestClassifier implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** @var DecisionTreeClassifier[]  The fitted ensemble members. */
    public readonly array $estimators_;

    /** Unique class labels from the FULL training set, sorted ascending. @var int[] */
    public readonly array $classes_;

    public readonly int $n_classes_;
    public readonly int $n_features_in_;

    /**
     * @param int               $n_estimators       Number of trees to grow.
     * @param int|null          $max_depth          Maximum depth of each tree (null = unlimited).
     * @param int               $min_samples_split  Minimum samples to attempt a split.
     * @param int|string|null   $max_features       Features per split in each tree:
     *                                              'sqrt' (default) = ceil(√n_features),
     *                                              'log2'           = ceil(log₂(n_features)),
     *                                              int              = exact count,
     *                                              null             = all features.
     * @param int               $random_state       Base RNG seed for bootstrap + tree seeds.
     */
    public function __construct(
        private readonly int             $n_estimators      = 100,
        private readonly ?int            $max_depth         = null,
        private readonly int             $min_samples_split = 2,
        private readonly int|string|null $max_features      = 'sqrt',
        private readonly int             $random_state      = 0,
    ) {
        if ($n_estimators < 1) {
            throw new \InvalidArgumentException(
                'RandomForestClassifier: n_estimators must be ≥ 1.'
            );
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Fit n_estimators trees on bootstrap samples of the training data.
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Integer class labels [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException(
                'RandomForestClassifier::fit() requires target $y.'
            );
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException(
                'RandomForestClassifier::fit() requires a 2-D feature matrix X.'
            );
        }

        [$n, $d] = $X->shape;
        $blas    = BlasEngine::get()->ffi;

        // ── Discover all classes from the FULL training set ────────────────
        //
        // This is done BEFORE bootstrap so that the forest's classes_ covers
        // rare labels even if some bootstrap samples omit them entirely.
        $seen = [];
        for ($i = 0; $i < $n; $i++) {
            $seen[(int) round((float) $y->buffer[$i])] = true;
        }
        ksort($seen);
        $allClasses  = array_keys($seen);
        $nAllClasses = count($allClasses);

        // ── Seed the global RNG from random_state ─────────────────────────
        mt_srand($this->random_state);

        // ── Grow n_estimators trees ────────────────────────────────────────
        $estimators = [];

        for ($t = 0; $t < $this->n_estimators; $t++) {
            // ── Bootstrap: draw n row indices with replacement ─────────────
            //
            // mt_rand(0, n-1) gives uniform discrete sampling over [0, n).
            // Each tree sees approximately 63.2% unique samples on average
            // (1 − 1/e), with duplicates providing variance reduction.
            $bootIdx = [];
            for ($i = 0; $i < $n; $i++) {
                $bootIdx[] = mt_rand(0, $n - 1);
            }

            // ── Build bootstrap tensors X_boot [n, d] and y_boot [n] ──────
            //
            // cblas_scopy(d, srcRowPtr, 1, dstRowPtr, 1) copies one row of d
            // floats from X into X_boot — one BLAS-1 call per bootstrap row.
            // Direct buffer index assignment handles the scalar label copy.
            $Xboot = new Tensor([$n, $d]);
            $yBoot = new Tensor([$n]);

            for ($i = 0; $i < $n; $i++) {
                $src    = $bootIdx[$i];

                // Source pointer: start of row $src in X (row-major: offset $src * $d)
                $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$src * $d]));
                // Destination pointer: start of row $i in X_boot
                $dstPtr = \FFI::cast('float*', \FFI::addr($Xboot->buffer[$i * $d]));

                $blas->cblas_scopy($d, $srcPtr, 1, $dstPtr, 1);

                // Scalar label copy — no BLAS primitive for a single element
                $yBoot->buffer[$i] = $y->buffer[$src];
            }

            // ── Fit one DecisionTree with a tree-specific seed ─────────────
            //
            // Each tree gets random_state = base + t so their internal
            // feature sub-sampling (Fisher-Yates) is independently seeded.
            $tree = new DecisionTreeClassifier(
                max_depth:          $this->max_depth,
                min_samples_split:  $this->min_samples_split,
                max_features:       $this->max_features,
                random_state:       $this->random_state + $t,
            );
            $tree->fit($Xboot, $yBoot);
            $estimators[] = $tree;
        }

        $this->estimators_    = $estimators;
        $this->classes_       = $allClasses;
        $this->n_classes_     = $nAllClasses;
        $this->n_features_in_ = $d;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict class labels: argmax of averaged predict_proba().
     *
     * @param Tensor $X  Feature matrix [n_samples, n_features]
     * @return Tensor    Predicted labels [n_samples]
     */
    public function predict(Tensor $X): Tensor
    {
        $proba = $this->predict_proba($X);
        $m     = $X->shape[0];
        $nC    = $this->n_classes_;
        $out   = new Tensor([$m]);

        for ($i = 0; $i < $m; $i++) {
            $base    = $i * $nC;
            $bestPos = 0;
            $bestVal = (float) $proba->buffer[$base];

            for ($c = 1; $c < $nC; $c++) {
                $v = (float) $proba->buffer[$base + $c];
                if ($v > $bestVal) {
                    $bestVal = $v;
                    $bestPos = $c;
                }
            }

            $out->buffer[$i] = (float) $this->classes_[$bestPos];
        }

        return $out;
    }

    /**
     * Predict averaged class probability distributions across all trees.
     *
     * Each tree contributes predict_proba(X) → [n_samples, tree_n_classes].
     * Probabilities are scattered into the forest's n_classes-wide class space
     * before accumulation, then normalised by cblas_sscal(1 / n_estimators).
     *
     * Memory layout of returned Tensor:
     *   out[i * n_classes + c] = P(forest_class c | X[i])   (row-major Float32)
     *
     * @param Tensor $X  Feature matrix [n_samples, n_features]
     * @return Tensor    Averaged probability matrix [n_samples, n_classes]
     */
    public function predict_proba(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "RandomForestClassifier::predict_proba() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        $m    = $X->shape[0];
        $nC   = $this->n_classes_;
        $blas = BlasEngine::get()->ffi;

        // Accumulator: sum of per-tree probability matrices over all estimators.
        // Zeroed by FFI::new; cblas_sscal scales to average at the end.
        $acc = new Tensor([$m, $nC]);   // [n_samples, n_classes_forest]

        // Forest-level reverse map: class label (int) → position in $this->classes_
        // Used to scatter a tree's narrower proba into the forest-wide acc buffer.
        $forestClassPos = array_flip($this->classes_);

        foreach ($this->estimators_ as $tree) {
            // predict_proba returns [n_samples, tree_n_classes] as a flat 1D Tensor.
            $proba    = $tree->predict_proba($X);
            $treeNC   = $tree->n_classes_;
            $treeCls  = $tree->classes_;   // sorted int[] — may be a subset of forest classes_

            // ── Scatter tree probabilities into forest-wide accumulator ─────
            //
            // For each sample i and each tree-class index tc:
            //   acc[i, forestClassPos[treeCls[tc]]] += proba[i, tc]
            //
            // If the bootstrap sample omitted a forest class entirely,
            // that tree simply contributes 0.0 for that class (buffer stays
            // at its current accumulated value — correct behaviour).
            //
            // Fast path: if the tree has exactly the same classes_ as the forest,
            // cblas_saxpy can accumulate the entire [m * nC] vector in one call.
            if ($treeNC === $nC) {
                // Same class count — classes_ are guaranteed to be identical
                // (both sorted from same full label set or identical bootstrap).
                $blas->cblas_saxpy($m * $nC, 1.0, $proba->buffer, 1, $acc->buffer, 1);
            } else {
                // Subset of classes — scatter per sample, per tree-class.
                for ($i = 0; $i < $m; $i++) {
                    $treeBase   = $i * $treeNC;
                    $forestBase = $i * $nC;

                    for ($tc = 0; $tc < $treeNC; $tc++) {
                        $classLabel = $treeCls[$tc];
                        $fp         = $forestClassPos[$classLabel] ?? null;
                        if ($fp !== null) {
                            $acc->buffer[$forestBase + $fp] =
                                (float) $acc->buffer[$forestBase + $fp]
                                + (float) $proba->buffer[$treeBase + $tc];
                        }
                    }
                }
            }
        }

        // ── Normalise: divide by n_estimators via a single BLAS-1 call ─────
        //
        // cblas_sscal(n, alpha, x, incx): x[i] *= alpha for all i.
        // This replaces every element in acc with acc[i] / n_estimators.
        $blas->cblas_sscal($m * $nC, 1.0 / $this->n_estimators, $acc->buffer, 1);

        return $acc;
    }

    /**
     * Accuracy score on test data.
     * Mirrors sklearn's ClassifierMixin.score().
     */
    public function score(Tensor $X, Tensor $y): float
    {
        $pred = $this->predict($X);
        $n    = $y->size;
        $ok   = 0;
        for ($i = 0; $i < $n; $i++) {
            if ((int) round((float) $y->buffer[$i]) === (int) round((float) $pred->buffer[$i])) {
                $ok++;
            }
        }
        return $ok / $n;
    }

    private function checkFitted(): void
    {
        if (!isset($this->estimators_)) {
            throw new \RuntimeException(
                'RandomForestClassifier is not fitted. Call fit() first.'
            );
        }
    }
}
