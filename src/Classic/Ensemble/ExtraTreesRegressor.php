<?php

declare(strict_types=1);

namespace Pml\Classic\Ensemble;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  ExtraTreesRegressor — sklearn.ensemble.ExtraTreesRegressor
//
//  Extremely Randomized Trees (Geurts et al., 2006) for regression.
//
//  Identical to ExtraTreesClassifier in structure and randomisation strategy,
//  but uses Variance Reduction (MSE criterion) for split evaluation instead
//  of Gini impurity, and predicts the mean of y values at each leaf.
//
//  ── Split criterion: Variance Reduction ────────────────────────────────────
//
//  For node with n samples (sum = Σy, baseline = sum²/n):
//
//    For a random threshold t on feature j:
//      score = sum_L² / n_L + sum_R² / n_R
//
//    Baseline (no split) = sum² / n.
//    Accept if score > baseline (VR > 0).
//
//  This is mathematically equivalent to the MSE criterion in
//  DecisionTreeRegressor but evaluated at a single random threshold per
//  feature rather than scanning all possible thresholds.
//
//  ── Leaf prediction ─────────────────────────────────────────────────────
//
//  Each leaf stores value[leaf] = mean(y_i for i in leaf).
//  predict() traverses each sample to its leaf and returns value[leaf].
//
//  ── BLAS in predict() aggregation ─────────────────────────────────────────
//
//  predict() accumulates per-tree predictions into a running sum:
//    cblas_saxpy(m, 1.0, tree_pred, 1, acc, 1)   — O(m) in C per tree
//  cblas_sscal(m, 1/T, acc, 1) normalises to mean across all trees.
//
//  Bootstrap row-copy (when bootstrap=true):
//    cblas_scopy(d, src, 1, dst, 1)               — one BLAS call per row
// ═══════════════════════════════════════════════════════════════════════════

final class ExtraTreesRegressor implements Estimator, Predictor
{
    private const TREE_LEAF      = -1;
    private const TREE_UNDEFINED = -2;

    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Array of internal tree representations.
     * Each element is an associative array: cl, cr, feat, thresh, value.
     * @var array[]
     */
    public readonly array $estimators_;

    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int               $n_estimators       Number of trees.
     * @param int|null          $max_depth          Max tree depth (null = unlimited).
     * @param int               $min_samples_split  Min samples to attempt a split.
     * @param int|string|null   $max_features       Features considered per split:
     *                                              'auto'/'sqrt' = ⌈√n_features⌉ (default),
     *                                              'log2'        = ⌈log₂(n_features)⌉,
     *                                              int           = exact count,
     *                                              null          = all features.
     * @param bool              $bootstrap          Draw bootstrap samples (default false for ExtraTrees).
     * @param ?int              $random_state       RNG seed.
     */
    public function __construct(
        private readonly int             $n_estimators      = 100,
        private readonly ?int            $max_depth         = null,
        private readonly int             $min_samples_split = 2,
        private readonly int|string|null $max_features      = 'auto',
        private readonly bool            $bootstrap         = false,
        private readonly ?int            $random_state      = null,
    ) {
        if ($n_estimators < 1) {
            throw new \InvalidArgumentException('ExtraTreesRegressor: n_estimators must be ≥ 1.');
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Grow n_estimators ExtraTrees on training data.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  Continuous target values [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('ExtraTreesRegressor::fit() requires $y.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('ExtraTreesRegressor::fit() requires 2-D X.');
        }

        [$n, $d] = $X->shape;
        $blas    = BlasEngine::get()->ffi;

        if ($this->random_state !== null) {
            mt_srand($this->random_state);
        }

        // Convert y buffer → plain float[] once (avoids per-node FFI reads)
        $yArr = [];
        for ($i = 0; $i < $n; $i++) {
            $yArr[$i] = (float) $y->buffer[$i];
        }

        $maxFeat  = $this->resolveMaxFeatures($d);
        $maxDepth = $this->max_depth ?? PHP_INT_MAX;

        $estimators = [];

        for ($t = 0; $t < $this->n_estimators; $t++) {
            // ── Optional bootstrap sampling ────────────────────────────────
            if ($this->bootstrap) {
                $bootIdx = [];
                for ($i = 0; $i < $n; $i++) {
                    $bootIdx[] = mt_rand(0, $n - 1);
                }
                $Xboot = new Tensor([$n, $d]);
                $yBoot = [];
                for ($i = 0; $i < $n; $i++) {
                    $src    = $bootIdx[$i];
                    $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$src * $d]));
                    $dstPtr = \FFI::cast('float*', \FFI::addr($Xboot->buffer[$i * $d]));
                    $blas->cblas_scopy($d, $srcPtr, 1, $dstPtr, 1);
                    $yBoot[$i] = $yArr[$src];
                }
                $Xtrain  = $Xboot;
                $yTrain  = $yBoot;
                $indices = range(0, $n - 1);
            } else {
                $Xtrain  = $X;
                $yTrain  = $yArr;
                $indices = range(0, $n - 1);
            }

            $estimators[] = $this->buildTree(
                $Xtrain, $yTrain, $indices, $maxFeat, $maxDepth,
            );
        }

        $this->estimators_    = $estimators;
        $this->n_features_in_ = $d;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict by averaging all tree predictions.
     *
     * Accumulation:
     *   1. acc = Tensor::zeros([m])
     *   2. For each tree: cblas_saxpy(m, 1.0, pred, acc)
     *   3. cblas_sscal(m, 1/T, acc)
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    Averaged predictions [n_samples]
     */
    public function predict(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "ExtraTreesRegressor::predict() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        $m    = $X->shape[0];
        $blas = BlasEngine::get()->ffi;
        $acc  = Tensor::zeros([$m]);

        foreach ($this->estimators_ as $tree) {
            $pred = $this->predictTree($tree, $X, $m);
            $blas->cblas_saxpy($m, 1.0, $pred->buffer, 1, $acc->buffer, 1);
        }
        $blas->cblas_sscal($m, 1.0 / $this->n_estimators, $acc->buffer, 1);

        return $acc;
    }

    /**
     * R² score on test data.
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

    // ── Tree building ─────────────────────────────────────────────────────

    /**
     * Build one ExtraTree (random-threshold regression tree).
     *
     * For each node:
     *   1. Compute sum and sum² of y values (needed for baseline and leaf value).
     *   2. Select max_features features randomly.
     *   3. For each candidate feature j:
     *      a. Find min/max of X[:, j] in this node's samples.
     *      b. Draw random threshold t ∈ [min_j, max_j].
     *      c. Evaluate score = sum_L²/n_L + sum_R²/n_R.
     *   4. Best score > baseline → split; else make leaf.
     *
     * @param  Tensor  $X        Feature matrix
     * @param  float[] $yArr     Target values (plain PHP float[])
     * @param  int[]   $indices  Root sample indices
     * @param  int     $maxFeat  Number of candidate features per split
     * @param  int     $maxDepth Maximum tree depth
     * @return array             Flat tree associative array
     */
    private function buildTree(
        Tensor $X,
        array  $yArr,
        array  $indices,
        int    $maxFeat,
        int    $maxDepth,
    ): array {
        [, $d] = $X->shape;

        $cl     = [];
        $cr     = [];
        $feat   = [];
        $thresh = [];
        $value  = [];

        $nodeCount = 0;
        $rootId    = $nodeCount++;
        $stack     = [[$rootId, $indices, 0]];

        while ($stack !== []) {
            [$nodeId, $idxs, $depth] = array_pop($stack);
            $nNode = count($idxs);

            // ── Node statistics ────────────────────────────────────────────
            $sumNode = 0.0;
            foreach ($idxs as $idx) {
                $sumNode += $yArr[$idx];
            }
            $value[$nodeId] = $sumNode / $nNode;

            // ── Leaf conditions ────────────────────────────────────────────
            // Check variance: ss = Σy² − sum²/n; pure if ss < ε
            $sumSq = 0.0;
            foreach ($idxs as $idx) {
                $v = $yArr[$idx];
                $sumSq += $v * $v;
            }
            $ss   = $sumSq - $sumNode * $sumNode / $nNode;
            $pure = ($ss < 1e-14);

            if ($depth >= $maxDepth || $nNode < $this->min_samples_split || $pure) {
                $cl[$nodeId]     = self::TREE_LEAF;
                $cr[$nodeId]     = self::TREE_LEAF;
                $feat[$nodeId]   = self::TREE_UNDEFINED;
                $thresh[$nodeId] = 0.0;
                continue;
            }

            // ── Select candidate features ─────────────────────────────────
            $allFeats = range(0, $d - 1);
            if ($maxFeat < $d) {
                for ($fi = 0; $fi < $maxFeat; $fi++) {
                    $rj = mt_rand($fi, $d - 1);
                    [$allFeats[$fi], $allFeats[$rj]] = [$allFeats[$rj], $allFeats[$fi]];
                }
                $selectedFeats = array_slice($allFeats, 0, $maxFeat);
            } else {
                $selectedFeats = $allFeats;
            }

            // Baseline: sum² / n (best achievable with no split)
            $baseline  = $sumNode * $sumNode / $nNode;
            $bestScore = $baseline;
            $bestFeat  = self::TREE_UNDEFINED;
            $bestThresh = 0.0;

            // ── Random threshold search ───────────────────────────────────
            foreach ($selectedFeats as $fj) {
                // Find min/max for this feature in this node
                $minVal =  PHP_FLOAT_MAX;
                $maxVal = -PHP_FLOAT_MAX;
                foreach ($idxs as $idx) {
                    $v = (float) $X->buffer[$idx * $d + $fj];
                    if ($v < $minVal) { $minVal = $v; }
                    if ($v > $maxVal) { $maxVal = $v; }
                }
                if ($maxVal - $minVal < 1e-9) {
                    continue;  // constant feature — no valid split
                }

                // Draw random threshold in (min_val, max_val)
                $t = $minVal + ($maxVal - $minVal) * mt_rand() / mt_getrandmax();

                // Compute left sum and count
                $sumL = 0.0;
                $nL   = 0;
                foreach ($idxs as $idx) {
                    if ((float) $X->buffer[$idx * $d + $fj] <= $t) {
                        $sumL += $yArr[$idx];
                        $nL++;
                    }
                }
                $nR = $nNode - $nL;
                if ($nL === 0 || $nR === 0) {
                    continue;
                }

                $sumR  = $sumNode - $sumL;
                $score = ($sumL * $sumL / $nL) + ($sumR * $sumR / $nR);

                if ($score > $bestScore) {
                    $bestScore  = $score;
                    $bestFeat   = $fj;
                    $bestThresh = $t;
                }
            }

            // ── Make leaf if no improving split was found ─────────────────
            if ($bestFeat === self::TREE_UNDEFINED) {
                $cl[$nodeId]     = self::TREE_LEAF;
                $cr[$nodeId]     = self::TREE_LEAF;
                $feat[$nodeId]   = self::TREE_UNDEFINED;
                $thresh[$nodeId] = 0.0;
                continue;
            }

            // ── Partition on best split ───────────────────────────────────
            $leftIdx  = [];
            $rightIdx = [];
            foreach ($idxs as $idx) {
                if ((float) $X->buffer[$idx * $d + $bestFeat] <= $bestThresh) {
                    $leftIdx[]  = $idx;
                } else {
                    $rightIdx[] = $idx;
                }
            }

            $leftId  = $nodeCount++;
            $rightId = $nodeCount++;

            $cl[$nodeId]     = $leftId;
            $cr[$nodeId]     = $rightId;
            $feat[$nodeId]   = $bestFeat;
            $thresh[$nodeId] = $bestThresh;

            $stack[] = [$rightId, $rightIdx, $depth + 1];
            $stack[] = [$leftId,  $leftIdx,  $depth + 1];
        }

        return ['cl' => $cl, 'cr' => $cr, 'feat' => $feat, 'thresh' => $thresh, 'value' => $value];
    }

    // ── Tree traversal ───────────────────────────────────────────────────

    /**
     * Predict continuous values by traversing the ExtraTree for each sample.
     *
     * @return Tensor [m] predicted values
     */
    private function predictTree(array $tree, Tensor $X, int $m): Tensor
    {
        $cl    = $tree['cl'];
        $cr    = $tree['cr'];
        $feat  = $tree['feat'];
        $thr   = $tree['thresh'];
        $value = $tree['value'];
        $d     = $X->shape[1];

        $out = new Tensor([$m]);

        for ($i = 0; $i < $m; $i++) {
            $node = 0;
            while ($cl[$node] !== self::TREE_LEAF) {
                $node = ((float) $X->buffer[$i * $d + $feat[$node]] <= $thr[$node])
                    ? $cl[$node]
                    : $cr[$node];
            }
            $out->buffer[$i] = $value[$node];
        }

        return $out;
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private function resolveMaxFeatures(int $d): int
    {
        return match (true) {
            is_int($this->max_features)                             => min($this->max_features, $d),
            $this->max_features === 'sqrt'                          => max(1, (int) ceil(sqrt($d))),
            $this->max_features === 'auto'                          => max(1, (int) ceil(sqrt($d))),
            $this->max_features === 'log2'                          => max(1, (int) ceil(log($d, 2))),
            $this->max_features === null                            => $d,
            default                                                 => $d,
        };
    }

    private function checkFitted(): void
    {
        if (!isset($this->estimators_)) {
            throw new \RuntimeException(
                'ExtraTreesRegressor is not fitted. Call fit() first.'
            );
        }
    }
}
