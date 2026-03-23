<?php

declare(strict_types=1);

namespace Pml\Classic\Ensemble;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};
use Pml\Classic\Tree\DecisionTreeRegressor;

// ═══════════════════════════════════════════════════════════════════════════
//  RandomForestRegressor — sklearn.ensemble.RandomForestRegressor
//
//  Bootstrap aggregation (bagging) of DecisionTreeRegressor instances.
//  Each tree is fitted on a bootstrap sample and uses a random feature
//  subset at each split.  Predictions are averaged across all trees.
//
//  ── Bootstrap Sampling ───────────────────────────────────────────────────
//
//  For each of the n_estimators trees:
//    1. Draw n_samples row indices uniformly at random WITH replacement
//       using PHP's mt_rand() seeded from random_state.
//    2. Build bootstrap tensors X_boot [n, d] and y_boot [n] by copying
//       rows from X and y.
//    3. Row copy: cblas_scopy(d, srcPtr, 1, dstPtr, 1) — one BLAS-1 call
//       per bootstrap row, no element-level PHP loops over feature vectors.
//    4. Scalar label copy: direct buffer assignment.
//
//  Each tree receives a distinct seed (base + tree index) so its internal
//  feature sub-sampling is independently randomised.
//
//  ── Prediction Averaging ─────────────────────────────────────────────────
//
//  For regression, ensemble aggregation is the arithmetic mean of all tree
//  predictions — no voting, no class alignment needed.
//
//  predict():
//    1. Allocate zero-initialised accumulator Tensor acc [n_samples].
//    2. For each tree: acc += tree.predict(X)  via cblas_saxpy(n, 1.0, …).
//    3. acc /= n_estimators                    via cblas_sscal(n, 1/T, …).
//
//  Both BLAS calls operate on the entire [n_samples] vector in a single C
//  call — no PHP element-level loops after the initial accumulation.
//
//  ── max_features ─────────────────────────────────────────────────────────
//
//  'auto' maps to 'sqrt' (ceil(√n_features)), matching sklearn ≥ 1.1
//  for RandomForestRegressor.  This was sklearn's historical default for
//  regressors before 'auto' was deprecated.
//  Pass max_features=null to use all features (bagging without feature
//  subsampling).
// ═══════════════════════════════════════════════════════════════════════════

final class RandomForestRegressor implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** @var DecisionTreeRegressor[]  The fitted ensemble members. */
    public readonly array $estimators_;

    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int               $n_estimators      Number of regression trees to grow.
     * @param int|null          $max_depth         Maximum depth of each tree.
     *                                             null = unlimited (can overfit; use with bootstrap).
     * @param int               $min_samples_split Minimum samples to attempt a split.
     * @param int|string|null   $max_features      Features considered per split:
     *                                             'auto'  → 'sqrt' (ceil(√n_features)),
     *                                             'sqrt'  → ceil(√n_features),
     *                                             'log2'  → ceil(log₂(n_features)),
     *                                             int     → exact count,
     *                                             null    → all features.
     * @param bool              $bootstrap         If true, each tree trains on a bootstrap
     *                                             sample drawn with replacement.
     *                                             If false, all trees see the full dataset
     *                                             (Pasting/ExtraTrees-style).
     * @param int|null          $random_state      Base RNG seed.  null → PHP's current state.
     */
    public function __construct(
        private readonly int             $n_estimators      = 100,
        private readonly ?int            $max_depth         = null,
        private readonly int             $min_samples_split = 2,
        private readonly int|string|null $max_features      = 'auto',
        private readonly bool            $bootstrap         = true,
        private readonly ?int            $random_state      = null,
    ) {
        if ($n_estimators < 1) {
            throw new \InvalidArgumentException(
                'RandomForestRegressor: n_estimators must be ≥ 1.'
            );
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Fit n_estimators regression trees on (optionally bootstrap-sampled) training data.
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Continuous target values [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException(
                'RandomForestRegressor::fit() requires target $y.'
            );
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException(
                'RandomForestRegressor::fit() requires a 2-D feature matrix X.'
            );
        }

        [$n, $d] = $X->shape;
        $blas    = BlasEngine::get()->ffi;

        // ── Seed the global RNG ────────────────────────────────────────────
        if ($this->random_state !== null) {
            mt_srand($this->random_state);
        }

        // ── Resolve max_features for tree construction ─────────────────────
        //
        // 'auto' → 'sqrt' (see class docblock).  Pass the resolved selector
        // string/int/null directly to each DecisionTreeRegressor constructor.
        $treeFeat = ($this->max_features === 'auto') ? 'sqrt' : $this->max_features;

        // ── Grow n_estimators trees ────────────────────────────────────────
        $estimators = [];

        for ($t = 0; $t < $this->n_estimators; $t++) {
            if ($this->bootstrap) {
                // ── Bootstrap: draw n row indices with replacement ─────────
                //
                // Each tree sees ~63.2% unique samples on average (1 − 1/e).
                // Duplicates act as implicit sample weighting and are the key
                // source of variance reduction in the ensemble.
                $bootIdx = [];
                for ($i = 0; $i < $n; $i++) {
                    $bootIdx[] = mt_rand(0, $n - 1);
                }

                // ── Build bootstrap Tensors X_boot [n, d] and y_boot [n] ──
                //
                // cblas_scopy(d, srcRowPtr, 1, dstRowPtr, 1) copies one row of
                // d floats from X into X_boot without a PHP element-level loop.
                $Xboot = new Tensor([$n, $d]);
                $yBoot = new Tensor([$n]);

                for ($i = 0; $i < $n; $i++) {
                    $src    = $bootIdx[$i];
                    $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$src * $d]));
                    $dstPtr = \FFI::cast('float*', \FFI::addr($Xboot->buffer[$i * $d]));
                    $blas->cblas_scopy($d, $srcPtr, 1, $dstPtr, 1);
                    $yBoot->buffer[$i] = $y->buffer[$src];
                }

                $Xtrain = $Xboot;
                $ytrain = $yBoot;
            } else {
                // ── No bootstrap: all trees see the full dataset ───────────
                $Xtrain = $X;
                $ytrain = $y;
            }

            // ── Fit one DecisionTreeRegressor with a tree-specific seed ────
            //
            // Seed = base + t ensures independent feature sub-sampling per tree.
            $tree = new DecisionTreeRegressor(
                max_depth:         $this->max_depth,
                min_samples_split: $this->min_samples_split,
                max_features:      $treeFeat,
                random_state:      ($this->random_state ?? 0) + $t,
            );
            $tree->fit($Xtrain, $ytrain);
            $estimators[] = $tree;
        }

        $this->estimators_    = $estimators;
        $this->n_features_in_ = $d;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict by averaging all tree predictions.
     *
     * Aggregation:
     *   1. Allocate zero-initialised acc [n_samples].
     *   2. For each tree: cblas_saxpy(n, 1.0, tree_pred, 1, acc, 1)
     *                     — accumulates tree output into acc in one BLAS-1 call.
     *   3. cblas_sscal(n, 1/T, acc, 1) — divides by n_estimators in one BLAS-1 call.
     *
     * @param Tensor $X  Feature matrix [n_samples, n_features]
     * @return Tensor    Averaged continuous predictions [n_samples]
     */
    public function predict(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "RandomForestRegressor::predict() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        $m    = $X->shape[0];
        $blas = BlasEngine::get()->ffi;

        // Zero-initialised accumulator for sum of tree predictions
        $acc = new Tensor([$m]);

        foreach ($this->estimators_ as $tree) {
            $pred = $tree->predict($X);   // [n_samples]

            // acc += pred  (BLAS saxpy: y = alpha*x + y,  alpha=1.0)
            $blas->cblas_saxpy($m, 1.0, $pred->buffer, 1, $acc->buffer, 1);
        }

        // acc /= n_estimators  (BLAS sscal: x *= alpha)
        $blas->cblas_sscal($m, 1.0 / $this->n_estimators, $acc->buffer, 1);

        return $acc;
    }

    /**
     * R² score on test data.
     * Mirrors sklearn's RegressorMixin.score().
     */
    public function score(Tensor $X, Tensor $y): float
    {
        $pred  = $this->predict($X);
        $n     = $y->size;

        $yMean = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $yMean += (float) $y->buffer[$i];
        }
        $yMean /= $n;

        $ssTot = 0.0;
        $ssRes = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $ssTot += ((float) $y->buffer[$i] - $yMean) ** 2;
            $ssRes += ((float) $y->buffer[$i] - (float) $pred->buffer[$i]) ** 2;
        }

        return ($ssTot === 0.0) ? 1.0 : 1.0 - $ssRes / $ssTot;
    }

    private function checkFitted(): void
    {
        if (!isset($this->estimators_)) {
            throw new \RuntimeException(
                'RandomForestRegressor is not fitted. Call fit() first.'
            );
        }
    }
}
