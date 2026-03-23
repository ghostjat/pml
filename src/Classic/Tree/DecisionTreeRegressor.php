<?php

declare(strict_types=1);

namespace Pml\Classic\Tree;

use Pml\Tensor;
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  DecisionTreeRegressor — sklearn.tree.DecisionTreeRegressor
//
//  CART regression tree.  Uses Variance Reduction (equivalent to MSE gain)
//  to select splits; stores the compiled tree as five flat PHP arrays
//  mirroring sklearn's internal sklearn.tree._tree.Tree structure.
//
//  ── Flat Tree Storage ────────────────────────────────────────────────────
//
//  All nodes are identified by a sequential integer node_id (root = 0).
//  Five parallel arrays, each indexed by node_id:
//
//    children_left [id]  → left child id if X[feat] ≤ threshold, else TREE_LEAF (-1)
//    children_right[id]  → right child id if X[feat] > threshold, else TREE_LEAF (-1)
//    feature       [id]  → feature index used for the split, or TREE_UNDEFINED (-2)
//    threshold     [id]  → split threshold (float), or TREE_UNDEFINED (-2.0)
//    value         [id]  → float mean of y for samples reaching this node
//
//  A leaf node is identified by: children_left[id] === TREE_LEAF.
//  At prediction time, the leaf's value is the output — no argmax or
//  normalisation is needed.
//
//  ── CART MSE / Variance Reduction Split Criterion ────────────────────────
//
//  For a node containing n samples with target values y_1, …, y_n, define:
//
//    sum   = Σ y_i
//    ȳ     = sum / n
//    SS    = Σ(y_i − ȳ)²   = Σ y_i² − sum²/n   (sum of squared deviations)
//
//  Variance reduction for a split at boundary p (left n_L, right n_R = n − n_L):
//
//    VR = SS_parent − (SS_left + SS_right)
//       = (Σy² − sum²/n) − (Σy²_L − sum_L²/n_L) − (Σy²_R − sum_R²/n_R)
//       = sum_L²/n_L + sum_R²/n_R − sum²/n          (Σy² terms cancel)
//
//  Since sum²/n is constant per node, maximising VR ⟺ maximising the criterion:
//
//    criterion = sum_L² / n_L  +  sum_R² / n_R
//
//  Baseline (no split) = sum² / n.  A split is accepted iff criterion > baseline.
//
//  ── Incremental O(n) Scan ────────────────────────────────────────────────
//
//  After sorting sample indices by X[:, feat_j], we scan left-to-right:
//    - Maintain sum_L (initially 0), n_L (initially 0).
//    - sum_R = sum_total − sum_L,  n_R = n − n_L.
//    - Moving sorted[sp] from R → L:  sum_L += y[sp],  n_L++.
//    - Evaluate: criterion = sum_L²/n_L + sum_R²/n_R.
//
//  O(n log n) sort + O(n) scan per feature — identical cost structure to the
//  Gini criterion in DecisionTreeClassifier.
//
//  ── Leaf Prediction ──────────────────────────────────────────────────────
//
//  Each leaf stores the MEAN of the y values that reach it:
//    value[leaf] = sum_y / n_node
//
//  predict() traverses each sample to its leaf and returns value[leaf] directly.
//
//  ── Tree Construction ────────────────────────────────────────────────────
//
//  Iterative (stack-based) DFS — identical to DecisionTreeClassifier.
//  Left children are pushed after right so they are processed first (pre-order),
//  matching sklearn's node-id ordering.
// ═══════════════════════════════════════════════════════════════════════════

final class DecisionTreeRegressor implements Estimator, Predictor
{
    // sklearn Tree sentinel constants (mirrored from DecisionTreeClassifier)
    private const TREE_LEAF      = -1;
    private const TREE_UNDEFINED = -2;

    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Flat tree arrays (sklearn Tree object equivalent).
     *
     * @var array{
     *   children_left:  int[],
     *   children_right: int[],
     *   feature:        int[],
     *   threshold:      float[],
     *   value:          float[]
     * }
     */
    public readonly array $tree_;

    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int|null          $max_depth         Maximum tree depth (null = unlimited).
     * @param int               $min_samples_split Minimum samples to consider splitting.
     * @param int|string|null   $max_features      Features considered per split:
     *                                             null   = all features,
     *                                             int    = exact count,
     *                                             'sqrt' = ceil(√n_features),
     *                                             'log2' = ceil(log₂(n_features)).
     * @param int               $random_state      RNG seed for feature sub-sampling.
     */
    public function __construct(
        private readonly ?int            $max_depth         = null,
        private readonly int             $min_samples_split = 2,
        private readonly int|string|null $max_features      = null,
        private readonly int             $random_state      = 0,
    ) {
        if ($min_samples_split < 2) {
            throw new \InvalidArgumentException(
                'DecisionTreeRegressor: min_samples_split must be ≥ 2.'
            );
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Build the CART regression tree on training data.
     *
     * The MSE/variance-reduction criterion is evaluated incrementally for each
     * feature, maintaining running sums left-to-right.  Only splits that produce
     * a strictly positive variance reduction are accepted.
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Continuous target values [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException(
                'DecisionTreeRegressor::fit() requires target $y.'
            );
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException(
                'DecisionTreeRegressor::fit() requires a 2-D feature matrix X.'
            );
        }

        [$n, $d] = $X->shape;

        // ── Resolve max_features count ─────────────────────────────────────
        $maxFeat = match (true) {
            is_int($this->max_features)    => min($this->max_features, $d),
            $this->max_features === 'sqrt' => max(1, (int) ceil(sqrt($d))),
            $this->max_features === 'log2' => max(1, (int) ceil(log($d, 2))),
            default                        => $d,   // null → all features
        };

        $maxDepth = $this->max_depth ?? PHP_INT_MAX;

        mt_srand($this->random_state);

        // ── Flat tree arrays ───────────────────────────────────────────────
        $childrenLeft  = [];
        $childrenRight = [];
        $featureArr    = [];
        $thresholdArr  = [];
        $valueArr      = [];   // float[] — mean y per node

        $nodeCount = 0;

        // ── Iterative pre-order DFS ────────────────────────────────────────
        $rootId = $nodeCount++;
        $stack  = [[$rootId, range(0, $n - 1), 0]];

        while ($stack !== []) {
            [$nodeId, $indices, $depth] = array_pop($stack);
            $nNode = count($indices);

            // ── Compute node statistics ────────────────────────────────────
            //
            // sum_node: Σ y_i for all i in $indices.
            // sum_sq_node: Σ y_i² (used to compute SS efficiently).
            $sumNode   = 0.0;
            $sumSqNode = 0.0;
            for ($ii = 0; $ii < $nNode; $ii++) {
                $v         = (float) $y->buffer[$indices[$ii]];
                $sumNode   += $v;
                $sumSqNode += $v * $v;
            }
            $meanNode = $sumNode / $nNode;

            // Every node stores its mean — this is the leaf prediction value
            // AND is stored at internal nodes for potential pruning / importance.
            $valueArr[$nodeId] = $meanNode;

            // ── Leaf conditions ────────────────────────────────────────────
            //   1. Maximum depth reached.
            //   2. Too few samples to split.
            //   3. Node is already pure (all y values identical → SS = 0).
            $ssNode = $sumSqNode - $sumNode * $sumNode / $nNode;
            $isPure = ($ssNode < 1e-14);

            if ($depth >= $maxDepth
                || $nNode < $this->min_samples_split
                || $isPure
            ) {
                $childrenLeft[$nodeId]  = self::TREE_LEAF;
                $childrenRight[$nodeId] = self::TREE_LEAF;
                $featureArr[$nodeId]    = self::TREE_UNDEFINED;
                $thresholdArr[$nodeId]  = (float) self::TREE_UNDEFINED;
                continue;
            }

            // ── Select feature subset via partial Fisher-Yates shuffle ─────
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

            // ── Baseline criterion: sum_total² / n (= no-split value) ──────
            //
            // Any real split must exceed this to be accepted (strict improvement).
            $bestCriterion = $sumNode * $sumNode / $nNode;

            $bestFeat   = self::TREE_UNDEFINED;
            $bestThresh = (float) self::TREE_UNDEFINED;

            // ── Scan each selected feature for the best variance-reduction split
            foreach ($selectedFeats as $feat_j) {
                // ── Sort sample indices by X[:, feat_j] ascending ──────────
                $sorted = $indices;
                usort($sorted, static function (int $a, int $b) use ($X, $d, $feat_j): int {
                    return (float) $X->buffer[$a * $d + $feat_j]
                        <=> (float) $X->buffer[$b * $d + $feat_j];
                });

                // ── Incremental left→right scan ────────────────────────────
                //
                // Start: all n samples on RIGHT (sum_L = 0, n_L = 0).
                // Move sorted[$sp] from R → L one at a time.
                //
                // criterion = sum_L² / n_L + sum_R² / n_R
                //
                // This equals sum_total² / n + VR, so maximising criterion
                // is equivalent to maximising variance reduction.
                $sumL  = 0.0;
                $sumR  = $sumNode;

                for ($sp = 0; $sp < $nNode - 1; $sp++) {
                    $idx = $sorted[$sp];
                    $v   = (float) $y->buffer[$idx];

                    // Move sample $sp from R → L
                    $sumL += $v;
                    $sumR -= $v;
                    $nL    = $sp + 1;
                    $nR    = $nNode - $nL;

                    // ── Skip equal adjacent feature values ─────────────────
                    //
                    // A threshold must lie strictly between two distinct values.
                    $valCurr = (float) $X->buffer[$idx * $d + $feat_j];
                    $valNext = (float) $X->buffer[$sorted[$sp + 1] * $d + $feat_j];
                    if ($valNext - $valCurr < 1e-9) {
                        continue;
                    }

                    // ── Evaluate variance-reduction criterion ──────────────
                    $criterion = ($sumL * $sumL / $nL) + ($sumR * $sumR / $nR);

                    if ($criterion > $bestCriterion) {
                        $bestCriterion = $criterion;
                        $bestFeat      = $feat_j;
                        $bestThresh    = ($valCurr + $valNext) * 0.5;
                    }
                }
            }

            // ── If no split improved the criterion, make this a leaf ───────
            if ($bestFeat === self::TREE_UNDEFINED) {
                $childrenLeft[$nodeId]  = self::TREE_LEAF;
                $childrenRight[$nodeId] = self::TREE_LEAF;
                $featureArr[$nodeId]    = self::TREE_UNDEFINED;
                $thresholdArr[$nodeId]  = (float) self::TREE_UNDEFINED;
                continue;
            }

            // ── Partition sample indices on the winning split ──────────────
            $leftIdx  = [];
            $rightIdx = [];
            foreach ($indices as $idx) {
                if ((float) $X->buffer[$idx * $d + $bestFeat] <= $bestThresh) {
                    $leftIdx[]  = $idx;
                } else {
                    $rightIdx[] = $idx;
                }
            }

            // ── Allocate child node IDs and record split ───────────────────
            $leftId  = $nodeCount++;
            $rightId = $nodeCount++;

            $childrenLeft[$nodeId]  = $leftId;
            $childrenRight[$nodeId] = $rightId;
            $featureArr[$nodeId]    = $bestFeat;
            $thresholdArr[$nodeId]  = $bestThresh;

            // Push right first so left is popped first (pre-order DFS)
            $stack[] = [$rightId, $rightIdx, $depth + 1];
            $stack[] = [$leftId,  $leftIdx,  $depth + 1];
        }

        // ── Store fitted attributes ────────────────────────────────────────
        $this->tree_ = [
            'children_left'  => $childrenLeft,
            'children_right' => $childrenRight,
            'feature'        => $featureArr,
            'threshold'      => $thresholdArr,
            'value'          => $valueArr,   // float[] — mean y per node
        ];

        $this->n_features_in_ = $d;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict continuous values by traversing each sample to its leaf.
     *
     * At each internal node: X[i, feature[node]] ≤ threshold[node] → left, else right.
     * At the leaf: return value[leaf] (the mean of training y values at that leaf).
     *
     * @param Tensor $X  Feature matrix [n_samples, n_features]
     * @return Tensor    Predicted values [n_samples]
     */
    public function predict(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "DecisionTreeRegressor::predict() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        [$m, $d] = $X->shape;
        $out     = new Tensor([$m]);

        // Cache tree arrays to avoid repeated property lookups per sample
        $childrenLeft  = $this->tree_['children_left'];
        $childrenRight = $this->tree_['children_right'];
        $featArr       = $this->tree_['feature'];
        $threshArr     = $this->tree_['threshold'];
        $valueArr      = $this->tree_['value'];

        for ($i = 0; $i < $m; $i++) {
            // ── Tree traversal ─────────────────────────────────────────────
            //
            // Follow left (≤ threshold) or right (> threshold) until leaf.
            // TREE_LEAF sentinel: children_left[$node] === -1.
            $node = 0;
            while ($childrenLeft[$node] !== self::TREE_LEAF) {
                $node = ((float) $X->buffer[$i * $d + $featArr[$node]] <= $threshArr[$node])
                    ? $childrenLeft[$node]
                    : $childrenRight[$node];
            }

            // Leaf value is the mean y of training samples at this node
            $out->buffer[$i] = $valueArr[$node];
        }

        return $out;
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
        if (!isset($this->tree_)) {
            throw new \RuntimeException(
                'DecisionTreeRegressor is not fitted. Call fit() first.'
            );
        }
    }
}
