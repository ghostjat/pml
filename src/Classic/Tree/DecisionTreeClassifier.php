<?php

declare(strict_types=1);

namespace Pml\Classic\Tree;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  DecisionTreeClassifier — sklearn.tree.DecisionTreeClassifier
//
//  CART (Classification and Regression Trees) for classification.
//  Uses Gini impurity to select splits; stores the compiled tree as five
//  flat PHP arrays — NO per-node objects — mirroring sklearn's internal
//  sklearn.tree._tree.Tree data structure.
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
//    value         [id]  → int[] class counts [n_classes] at this node
//
//  A leaf node is identified by: children_left[id] === TREE_LEAF.
//  Only leaf values are used for prediction; internal node values are stored
//  for potential feature importance computation.
//
//  ── CART Gini Split Selection ────────────────────────────────────────────
//
//  For a node containing n samples with class counts c_k, the Gini impurity is:
//
//    Gini(node) = 1 − Σ_k (c_k / n)²
//
//  For a split at position p (after sorting by feature f):
//    Left  set L has n_L = p + 1 samples, right set R has n_R = n − n_L.
//
//    Gini_split = (n_L/n) · Gini(L) + (n_R/n) · Gini(R)
//               = 1 − (Σ c²_k_L / n_L  +  Σ c²_k_R / n_R) / n
//
//  Minimising Gini_split ⟺ maximising the criterion:
//
//    criterion = Σ c²_k_L / n_L  +  Σ c²_k_R / n_R
//
//  The sums-of-squares are maintained incrementally as we scan left-to-right
//  through the sorted samples.  Moving one sample of class k from R → L:
//
//    Δ(Σ c²_R) = (c_R − 1)² − c_R² = −2·c_R + 1
//    Δ(Σ c²_L) = (c_L + 1)² − c_L² = +2·c_L + 1
//
//  This gives an O(n) scan per feature after O(n log n) sort.
//
//  ── Tree Construction ────────────────────────────────────────────────────
//
//  An iterative (stack-based) DFS is used rather than PHP recursion to avoid
//  stack-overflow on deep trees.  Each frame holds (node_id, sample_indices[],
//  depth).  Left children are pushed after right so they are processed first
//  (pre-order traversal, matching sklearn's node ordering).
//
//  ── predict / predict_proba ──────────────────────────────────────────────
//
//  For each test sample, tree traversal is a while-loop that reads
//  children_left / children_right until hitting TREE_LEAF, then reads the
//  class counts from value[leaf_id] and normalises to probabilities.
//  predict() returns argmax of the probability vector.
// ═══════════════════════════════════════════════════════════════════════════

final class DecisionTreeClassifier implements Estimator, Predictor
{
    // sklearn Tree sentinel constants
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
     *   value:          array<int, int[]>
     * }
     */
    public readonly array $tree_;

    /** Unique class labels discovered in fit(), sorted ascending. @var int[] */
    public readonly array $classes_;

    public readonly int $n_classes_;
    public readonly int $n_features_in_;

    /**
     * @param int|null          $max_depth          Maximum tree depth (null = unlimited).
     * @param int               $min_samples_split  Minimum samples to attempt a split.
     * @param int|string|null   $max_features       Features per split: null = all features,
     *                                              int = exact count,
     *                                              'sqrt' = ceil(√n_features),
     *                                              'log2' = ceil(log₂(n_features)).
     * @param int               $random_state       RNG seed for feature sub-sampling.
     */
    public function __construct(
        private readonly ?int            $max_depth         = null,
        private readonly int             $min_samples_split = 2,
        private readonly int|string|null $max_features      = null,
        private readonly int             $random_state      = 0,
    ) {
        if ($min_samples_split < 2) {
            throw new \InvalidArgumentException(
                'DecisionTreeClassifier: min_samples_split must be ≥ 2.'
            );
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Build the CART tree on training data.
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Integer class labels [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException(
                'DecisionTreeClassifier::fit() requires target $y.'
            );
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException(
                'DecisionTreeClassifier::fit() requires a 2-D feature matrix X.'
            );
        }

        [$n, $d] = $X->shape;

        // ── Discover unique class labels ───────────────────────────────────
        $seen = [];
        for ($i = 0; $i < $n; $i++) {
            $seen[(int) round((float) $y->buffer[$i])] = true;
        }
        ksort($seen);
        $classes    = array_keys($seen);    // sorted int[]
        $nClasses   = count($classes);
        // Reverse map: class label (int) → position index in $classes
        $classToPos = array_flip($classes);

        // ── Resolve max_features count ─────────────────────────────────────
        $maxFeat = match (true) {
            is_int($this->max_features)    => min($this->max_features, $d),
            $this->max_features === 'sqrt' => max(1, (int) ceil(sqrt($d))),
            $this->max_features === 'log2' => max(1, (int) ceil(log($d, 2))),
            default                        => $d,   // null → all features
        };

        $maxDepth = $this->max_depth ?? PHP_INT_MAX;

        mt_srand($this->random_state);

        // ── Flat tree arrays (all empty; filled during DFS) ────────────────
        $childrenLeft  = [];
        $childrenRight = [];
        $featureArr    = [];
        $thresholdArr  = [];
        $valueArr      = [];   // int[][] — class counts per node

        $nodeCount = 0;   // sequential node ID allocator (root = 0)

        // ── Iterative pre-order DFS ────────────────────────────────────────
        //
        // Stack entries: [node_id (int), sample_indices (int[]), depth (int)]
        // We push right child before left so left is popped first (pre-order).
        $rootId = $nodeCount++;
        $stack  = [[$rootId, range(0, $n - 1), 0]];

        while ($stack !== []) {
            [$nodeId, $indices, $depth] = array_pop($stack);
            $nNode = count($indices);

            // ── Compute class counts at this node ──────────────────────────
            //
            // Counts are always stored (needed for predict_proba at leaves and
            // optionally for feature importance at internal nodes).
            $counts = array_fill(0, $nClasses, 0);
            for ($ii = 0; $ii < $nNode; $ii++) {
                $lbl = (int) round((float) $y->buffer[$indices[$ii]]);
                $counts[$classToPos[$lbl]]++;
            }
            $valueArr[$nodeId] = $counts;

            // ── Leaf conditions ────────────────────────────────────────────
            //  1. Maximum depth reached.
            //  2. Too few samples to attempt another split.
            //  3. Node is already pure (only one class present).
            $isPure = (max($counts) === $nNode);

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
            //
            // Swap the first $maxFeat positions with randomly chosen positions
            // in [fi, d-1].  This avoids a full array shuffle when maxFeat << d.
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

            // ── Parent sum-of-squares (for the no-split baseline) ──────────
            //
            // criterion = Σ c²_k / n  when everything is on one side.
            // Any real split must achieve a strictly higher criterion.
            $sumSqParent = 0.0;
            foreach ($counts as $c) {
                $sumSqParent += (float) ($c * $c);
            }
            $bestCriterion = $sumSqParent / $nNode;  // baseline (no gain)

            $bestFeat   = self::TREE_UNDEFINED;
            $bestThresh = (float) self::TREE_UNDEFINED;

            // ── Scan each selected feature for the best Gini split ─────────
            foreach ($selectedFeats as $feat_j) {
                // ── Sort sample indices by X[:, feat_j] ascending ──────────
                //
                // PHP usort uses a comparison closure that reads directly from
                // the FFI Float32 buffer.  The sort is O(n log n).
                $sorted = $indices;   // value-copy (PHP array semantics)
                usort($sorted, static function (int $a, int $b) use ($X, $d, $feat_j): int {
                    return (float) $X->buffer[$a * $d + $feat_j]
                        <=> (float) $X->buffer[$b * $d + $feat_j];
                });

                // ── Incremental left→right scan ────────────────────────────
                //
                // Initialise: all n samples are on the RIGHT, none on the LEFT.
                // We slide the split boundary one position at a time, moving
                // sorted[sp] from R → L and updating the sum-of-squares in O(1).
                $leftCounts  = array_fill(0, $nClasses, 0);
                $rightCounts = $counts;       // fresh copy per feature
                $sumSqLeft   = 0.0;
                $sumSqRight  = $sumSqParent;

                for ($sp = 0; $sp < $nNode - 1; $sp++) {
                    $idx = $sorted[$sp];
                    $cls = $classToPos[(int) round((float) $y->buffer[$idx])];

                    // ── Move sorted[$sp] from RIGHT to LEFT ────────────────
                    //
                    //  Right side loses one count of class $cls:
                    //    Δ(Σ c²_R) = (c_R − 1)² − c_R²  =  −2·c_R + 1
                    $cR = $rightCounts[$cls];
                    $sumSqRight   += -2.0 * $cR + 1.0;
                    $rightCounts[$cls]--;

                    //  Left side gains one count of class $cls:
                    //    Δ(Σ c²_L) = (c_L + 1)² − c_L²  =  +2·c_L + 1
                    $cL = $leftCounts[$cls];
                    $sumSqLeft    += 2.0 * $cL + 1.0;
                    $leftCounts[$cls]++;

                    // ── Skip if adjacent values are equal ──────────────────
                    //
                    // A valid threshold must lie strictly between two distinct
                    // feature values.  If sorted[sp] and sorted[sp+1] share the
                    // same value for feat_j, no threshold can separate them.
                    $valCurr = (float) $X->buffer[$idx * $d + $feat_j];
                    $valNext = (float) $X->buffer[$sorted[$sp + 1] * $d + $feat_j];
                    if ($valNext - $valCurr < 1e-9) {
                        continue;   // duplicate feature value → no split here
                    }

                    $nLeft  = $sp + 1;
                    $nRight = $nNode - $nLeft;

                    // ── Evaluate split criterion ───────────────────────────
                    //
                    //  criterion = Σ c²_L / n_L  +  Σ c²_R / n_R
                    //
                    //  This is maximised when Gini_split is minimised.
                    //  A split is only accepted if criterion > baseline (positive Gini gain).
                    $criterion = $sumSqLeft / $nLeft + $sumSqRight / $nRight;

                    if ($criterion > $bestCriterion) {
                        $bestCriterion = $criterion;
                        $bestFeat      = $feat_j;
                        // Threshold is the midpoint of the two boundary values.
                        $bestThresh    = ($valCurr + $valNext) * 0.5;
                    }
                }
                // End of split scan for feature feat_j.
                // $rightCounts / $leftCounts are discarded; fresh copies used next iteration.
            }

            // ── If no split improved the criterion, this becomes a leaf ─────
            if ($bestFeat === self::TREE_UNDEFINED) {
                $childrenLeft[$nodeId]  = self::TREE_LEAF;
                $childrenRight[$nodeId] = self::TREE_LEAF;
                $featureArr[$nodeId]    = self::TREE_UNDEFINED;
                $thresholdArr[$nodeId]  = (float) self::TREE_UNDEFINED;
                continue;
            }

            // ── Partition sample indices on the winning split ──────────────
            //
            // O(n) scan: left if X[idx, bestFeat] ≤ bestThresh, else right.
            // This avoids storing intermediate sorted arrays from the scan above.
            $leftIdx  = [];
            $rightIdx = [];
            foreach ($indices as $idx) {
                if ((float) $X->buffer[$idx * $d + $bestFeat] <= $bestThresh) {
                    $leftIdx[]  = $idx;
                } else {
                    $rightIdx[] = $idx;
                }
            }

            // ── Allocate child node IDs and record split in flat arrays ─────
            $leftId  = $nodeCount++;
            $rightId = $nodeCount++;

            $childrenLeft[$nodeId]  = $leftId;
            $childrenRight[$nodeId] = $rightId;
            $featureArr[$nodeId]    = $bestFeat;
            $thresholdArr[$nodeId]  = $bestThresh;

            // Push right child first so left is popped first (pre-order DFS).
            $stack[] = [$rightId, $rightIdx, $depth + 1];
            $stack[] = [$leftId,  $leftIdx,  $depth + 1];
        }

        // ── Store fitted attributes ────────────────────────────────────────
        $this->tree_ = [
            'children_left'  => $childrenLeft,
            'children_right' => $childrenRight,
            'feature'        => $featureArr,
            'threshold'      => $thresholdArr,
            'value'          => $valueArr,
        ];

        $this->classes_       = $classes;
        $this->n_classes_     = $nClasses;
        $this->n_features_in_ = $d;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict class labels: argmax of predict_proba().
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
     * Predict class probability distributions.
     *
     * For each sample, traverse the tree to the leaf and normalise the leaf's
     * class-count vector to a probability distribution.
     *
     * Memory layout:
     *   out[i * n_classes + c] = P(class c | X[i])   (row-major Float32)
     *
     * @param Tensor $X  Feature matrix [n_samples, n_features]
     * @return Tensor    Probability matrix [n_samples, n_classes] (flat 1D)
     */
    public function predict_proba(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "DecisionTreeClassifier::predict_proba() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        [$m, $d] = $X->shape;
        $nC      = $this->n_classes_;
        $out     = new Tensor([$m, $nC]);   // zero-initialised by FFI::new

        // Cache tree arrays in local variables to avoid repeated property lookups
        $childrenLeft  = $this->tree_['children_left'];
        $childrenRight = $this->tree_['children_right'];
        $featArr       = $this->tree_['feature'];
        $threshArr     = $this->tree_['threshold'];
        $valueArr      = $this->tree_['value'];

        for ($i = 0; $i < $m; $i++) {
            // ── Tree traversal for sample i ───────────────────────────────
            //
            // Start at root (node 0).  At each internal node compare
            // X[i, feature[node]] against threshold[node]:
            //   ≤ threshold → go left   (children_left[node])
            //   > threshold → go right  (children_right[node])
            // Stop when children_left[node] === TREE_LEAF.
            $node = 0;
            while ($childrenLeft[$node] !== self::TREE_LEAF) {
                $node = ((float) $X->buffer[$i * $d + $featArr[$node]] <= $threshArr[$node])
                    ? $childrenLeft[$node]
                    : $childrenRight[$node];
            }

            // ── Normalise leaf class counts → probabilities ───────────────
            $leafCounts = $valueArr[$node];   // int[]
            $total      = (float) array_sum($leafCounts);
            $base       = $i * $nC;

            if ($total > 0.0) {
                for ($c = 0; $c < $nC; $c++) {
                    $out->buffer[$base + $c] = (float) $leafCounts[$c] / $total;
                }
            }
            // Empty leaf (pathological) → row stays all-zeros.
        }

        return $out;
    }

    /**
     * Accuracy score: fraction of samples correctly classified.
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
        if (!isset($this->tree_)) {
            throw new \RuntimeException(
                'DecisionTreeClassifier is not fitted. Call fit() first.'
            );
        }
    }
}
