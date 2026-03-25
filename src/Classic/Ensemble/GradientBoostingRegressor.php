<?php

declare(strict_types=1);

namespace Pml\Classic\Ensemble;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};
use Pml\Classic\Tree\DecisionTreeRegressor;

// ═══════════════════════════════════════════════════════════════════════════
//  GradientBoostingRegressor — sklearn.ensemble.GradientBoostingRegressor
//
//  Friedman's Gradient Boosting Machine for regression.
//  Minimises the squared-error (L2) loss by fitting a sequence of shallow
//  DecisionTreeRegressor models to the pseudo-residuals.
//
//  ── Algorithm (Friedman 2001, §3) ─────────────────────────────────────────
//
//    1. Initialise:   F_0(x) = ȳ  (mean of training targets)
//
//    2. For t = 1 … T:
//       a. Pseudo-residuals:  r_i = y_i − F_{t-1}(x_i)
//          r is the negative gradient of ½‖y−F‖² w.r.t. F — i.e. the
//          direction of steepest descent in function space.
//       b. Optional stochastic subsample (subsample < 1.0):
//          draw ⌈n·subsample⌉ indices without replacement via a partial
//          Fisher-Yates shuffle; fit the tree on this sub-matrix.
//       c. Fit DecisionTreeRegressor(max_depth) to (X_sub, r_sub).
//       d. F_t(x) = F_{t-1}(x) + η · tree_t.predict(x)
//          The update always uses the FULL feature matrix X so that F
//          stays consistent across all training samples.
//
//    3. Predict:  F_T(x) = F_0 + η Σ_t tree_t(x)
//
//  ── BLAS operations ───────────────────────────────────────────────────────
//
//  Pseudo-residual computation (each stage, O(n)):
//    r ← y.clone()
//    cblas_saxpy(n, −1.0, F, 1, r, 1)    →  r = y − F
//
//  F update (each stage, O(n)):
//    cblas_saxpy(n, η, tree_pred, 1, F, 1)  →  F += η · tree_pred
//
//  Both collapse O(n) work into a single BLAS-1 C call, avoiding PHP loops
//  over the training set for these hot-path operations.
//
//  Subsample row copy:
//    cblas_scopy(d, X[src*d], 1, Xsub[i*d], 1)  — O(1) BLAS per row
//
//  ── Stochastic GBM ────────────────────────────────────────────────────────
//
//  When subsample < 1.0 (sklearn default = 1.0), each tree is fit on a
//  random fraction of the training data sampled WITHOUT replacement.
//  This reduces variance and acts as implicit regularisation.
//  The F update, however, is always applied over ALL n training samples
//  (using the full X) so the running prediction F remains consistent.
//
//  ── max_features ─────────────────────────────────────────────────────────
//
//  Passed directly to DecisionTreeRegressor.  null = all features (sklearn
//  default for GBM).  'sqrt', 'log2', or an int are also accepted.
// ═══════════════════════════════════════════════════════════════════════════

final class GradientBoostingRegressor implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * One fitted tree per boosting stage.
     * @var DecisionTreeRegressor[]
     */
    public readonly array $estimators_;

    /** F_0: constant initial prediction = mean(y_train). */
    public readonly float $init_val_;

    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int               $n_estimators       Number of boosting stages (trees).
     * @param float             $learning_rate       Shrinkage factor η applied to each tree.
     *                                               Lower values need more trees but generalise better.
     * @param int               $max_depth           Maximum depth of each regression tree.
     *                                               Shallow trees (3–5) work best for GBM.
     * @param int               $min_samples_split   Minimum node size to attempt a split.
     * @param float             $subsample           Fraction of training samples per tree (0,1].
     *                                               < 1.0 enables Stochastic Gradient Boosting.
     * @param int|string|null   $max_features        Features to consider per split in each tree.
     *                                               null = all (sklearn default for GBM).
     * @param ?int              $random_state        RNG seed for bootstrap and tree seeds.
     */
    public function __construct(
        private readonly int             $n_estimators      = 100,
        private readonly float           $learning_rate     = 0.1,
        private readonly int             $max_depth         = 3,
        private readonly int             $min_samples_split = 2,
        private readonly float           $subsample         = 1.0,
        private readonly int|string|null $max_features      = null,
        private readonly ?int            $random_state      = null,
    ) {
        if ($n_estimators < 1) {
            throw new \InvalidArgumentException('GradientBoostingRegressor: n_estimators must be ≥ 1.');
        }
        if ($learning_rate <= 0.0) {
            throw new \InvalidArgumentException('GradientBoostingRegressor: learning_rate must be > 0.');
        }
        if ($subsample <= 0.0 || $subsample > 1.0) {
            throw new \InvalidArgumentException('GradientBoostingRegressor: subsample must be in (0, 1].');
        }
        if ($max_depth < 1) {
            throw new \InvalidArgumentException('GradientBoostingRegressor: max_depth must be ≥ 1.');
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Build the gradient boosting ensemble on training data.
     *
     * Workflow:
     *   1. Initialise F_0 = mean(y_train).
     *   2. For each stage t: compute r = y − F; (optionally subsample);
     *      fit tree to (X_sub, r_sub); update F += η · tree.predict(X_full).
     *   3. Store all fitted trees and F_0.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  Continuous targets [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('GradientBoostingRegressor: y must be provided.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('GradientBoostingRegressor: X must be 2-D [n_samples, n_features].');
        }

        [$n, $d] = $X->shape;
        $blas    = BlasEngine::get()->ffi;

        if ($this->random_state !== null) {
            mt_srand($this->random_state);
        }

        // ── Step 1: Initialise F_0 = mean(y) ──────────────────────────────
        $sumY = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $sumY += (float) $y->buffer[$i];
        }
        $initVal = $sumY / $n;

        // Running prediction F [n]: starts at F_0 everywhere
        $F = Tensor::full([$n], $initVal);

        // ── Step 2: Boosting loop ─────────────────────────────────────────
        $estimators = [];
        $nSub       = ($this->subsample < 1.0)
                      ? max(1, (int) ceil($n * $this->subsample))
                      : $n;
        $stochastic = ($this->subsample < 1.0);

        for ($t = 0; $t < $this->n_estimators; $t++) {

            // ── Pseudo-residuals: r = y − F ──────────────────────────────
            //
            // Clone y into r, then saxpy(−1, F, r):
            //   r[i] ← y[i] − F[i]   (all n elements in one BLAS call)
            $r = $y->clone();
            $blas->cblas_saxpy($n, -1.0, $F->buffer, 1, $r->buffer, 1);

            // ── Optional stochastic subsample ────────────────────────────
            if ($stochastic) {
                // Partial Fisher-Yates: shuffle indices [0..n-1], take first nSub
                $allIdx = range(0, $n - 1);
                for ($i = 0; $i < $nSub; $i++) {
                    $j          = mt_rand($i, $n - 1);
                    [$allIdx[$i], $allIdx[$j]] = [$allIdx[$j], $allIdx[$i]];
                }
                $subIdx = array_slice($allIdx, 0, $nSub);

                $Xsub = new Tensor([$nSub, $d]);
                $rsub = new Tensor([$nSub]);

                for ($i = 0; $i < $nSub; $i++) {
                    $src    = $subIdx[$i];
                    $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$src * $d]));
                    $dstPtr = \FFI::cast('float*', \FFI::addr($Xsub->buffer[$i * $d]));
                    $blas->cblas_scopy($d, $srcPtr, 1, $dstPtr, 1);
                    $rsub->buffer[$i] = $r->buffer[$src];
                }

                $Xtrain = $Xsub;
                $rtrain = $rsub;
            } else {
                $Xtrain = $X;
                $rtrain = $r;
            }

            // ── Fit tree on (X_train, r_train) ───────────────────────────
            $tree = new DecisionTreeRegressor(
                max_depth:         $this->max_depth,
                min_samples_split: $this->min_samples_split,
                max_features:      $this->max_features,
                random_state:      ($this->random_state ?? 0) + $t,
            );
            $tree->fit($Xtrain, $rtrain);
            $estimators[] = $tree;

            // ── F += η · tree.predict(X_full) ─────────────────────────────
            //
            // Update uses FULL X so every training sample's F stays consistent,
            // even when the tree was trained on a subsample.
            $pred = $tree->predict($X);
            $blas->cblas_saxpy($n, $this->learning_rate, $pred->buffer, 1, $F->buffer, 1);
        }

        $this->estimators_    = $estimators;
        $this->init_val_      = $initVal;
        $this->n_features_in_ = $d;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict continuous targets.
     *
     * F(x) = F_0 + η Σ_t tree_t.predict(x)
     *
     * Accumulation via cblas_saxpy: O(m · n_estimators) in BLAS-1 C calls.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    Predicted values [n_samples]
     */
    public function predict(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "GradientBoostingRegressor::predict() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        $m    = $X->shape[0];
        $blas = BlasEngine::get()->ffi;

        // Start from the constant baseline F_0
        $F = Tensor::full([$m], $this->init_val_);

        // Accumulate: F += η · tree_t.predict(X)  for each stage t
        foreach ($this->estimators_ as $tree) {
            $pred = $tree->predict($X);
            $blas->cblas_saxpy($m, $this->learning_rate, $pred->buffer, 1, $F->buffer, 1);
        }

        return $F;
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

    // ── Helpers ───────────────────────────────────────────────────────────

    private function checkFitted(): void
    {
        if (!isset($this->estimators_)) {
            throw new \RuntimeException(
                'GradientBoostingRegressor is not fitted. Call fit() first.'
            );
        }
    }
}
