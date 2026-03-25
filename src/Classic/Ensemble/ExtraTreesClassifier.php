<?php

declare(strict_types=1);

namespace Pml\Classic\Ensemble;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  ExtraTreesClassifier — sklearn.ensemble.ExtraTreesClassifier
//
//  Extremely Randomized Trees (Geurts et al., 2006) for classification.
//
//  Differences from RandomForestClassifier:
//
//    1. No bootstrap by default (bootstrap=false): each tree is trained on
//       the FULL training set.  Without bootstrap variance comes from the
//       random split selection alone, not sampling.
//
//    2. Random thresholds: for each candidate feature, a single threshold t
//       is drawn uniformly from [min_feat, max_feat] in the current node,
//       instead of scanning all possible split points.  The best (feature, t)
//       pair among all max_features candidates is selected.
//
//  These two changes together:
//    • Eliminate the O(n log n) sort per feature (→ O(n) per feature).
//    • Eliminate bootstrap overhead.
//    • Increase bias slightly but significantly reduce variance.
//    • Make tree construction faster than standard Random Forest.
//
//  ── Split criterion: Gini ──────────────────────────────────────────────────
//
//  For a node with n samples and K classes (count_k samples of class k):
//
//    Gini purity  = Σ_k (n_k / n)²        (higher = purer node)
//
//  For a random split threshold t on feature j, producing left/right subsets:
//
//    score = Σ_k n_{k,L}² / n_L + Σ_k n_{k,R}² / n_R
//
//  Baseline (no split) = Σ_k n_k² / n.  Accept if score > baseline.
//
//  ── Internal tree storage (flat PHP arrays) ────────────────────────────────
//
//  Each tree is stored as an associative PHP array matching the layout of
//  Pml\Classic\Tree\DecisionTreeClassifier:
//
//    cl[id]     → left child id  (−1 = leaf)
//    cr[id]     → right child id (−1 = leaf)
//    feat[id]   → feature index used for split (−2 = leaf)
//    thresh[id] → split threshold (float)
//    proba[id]  → float[K]: normalised class frequencies at node id
//
//  Leaf nodes have cl[id] = −1.  predict() and predict_proba() traverse
//  the tree using the (feat, thresh) arrays until a leaf is reached, then
//  return proba[leaf].
//
//  ── BLAS in aggregation ───────────────────────────────────────────────────
//
//  predict_proba() accumulates per-tree [n_samples, K] matrices.
//  Same BLAS strategy as RandomForestClassifier:
//    • Fast path (all trees same K): cblas_saxpy(m*K, 1.0, treeProba, acc).
//    • Slow path (rare subset-class trees): scattered PHP loop.
//    • Normalise: cblas_sscal(m*K, 1/T, acc).
//
//  Bootstrap row-copy (when bootstrap=true):
//    cblas_scopy(d, src, 1, dst, 1) — one BLAS-1 call per bootstrap row.
// ═══════════════════════════════════════════════════════════════════════════

final class ExtraTreesClassifier implements Estimator, Predictor
{
    private const TREE_LEAF      = -1;
    private const TREE_UNDEFINED = -2;

    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Array of internal tree representations.
     * Each element is an associative array with keys: cl, cr, feat, thresh, proba, n_classes.
     * @var array[]
     */
    public readonly array $estimators_;

    /** @var int[] Sorted unique class labels from the FULL training set. */
    public readonly array $classes_;

    public readonly int $n_classes_;
    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int               $n_estimators       Number of trees.
     * @param int|null          $max_depth          Max tree depth (null = unlimited).
     * @param int               $min_samples_split  Min samples to attempt a split.
     * @param int|string|null   $max_features       Features considered per split:
     *                                              'sqrt' (default) = ⌈√n_features⌉,
     *                                              'log2'           = ⌈log₂(n_features)⌉,
     *                                              int              = exact count,
     *                                              null             = all features.
     * @param bool              $bootstrap          Draw bootstrap samples (default false for ExtraTrees).
     * @param ?int              $random_state       RNG seed.
     */
    public function __construct(
        private readonly int             $n_estimators      = 100,
        private readonly ?int            $max_depth         = null,
        private readonly int             $min_samples_split = 2,
        private readonly int|string|null $max_features      = 'sqrt',
        private readonly bool            $bootstrap         = false,
        private readonly ?int            $random_state      = null,
    ) {
        if ($n_estimators < 1) {
            throw new \InvalidArgumentException('ExtraTreesClassifier: n_estimators must be ≥ 1.');
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Grow n_estimators ExtraTrees on training data.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  Integer class labels [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('ExtraTreesClassifier::fit() requires $y.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('ExtraTreesClassifier::fit() requires 2-D X.');
        }

        [$n, $d] = $X->shape;
        $blas    = BlasEngine::get()->ffi;

        if ($this->random_state !== null) {
            mt_srand($this->random_state);
        }

        // ── Discover all classes from the FULL dataset ─────────────────────
        $seen = [];
        for ($i = 0; $i < $n; $i++) {
            $seen[(int) round((float) $y->buffer[$i])] = true;
        }
        ksort($seen);
        $allClasses  = array_keys($seen);
        $K           = count($allClasses);
        $classPos    = array_flip($allClasses);

        // Convert y buffer → plain int[] once (avoids per-node FFI read)
        $yInt = [];
        for ($i = 0; $i < $n; $i++) {
            $yInt[$i] = (int) round((float) $y->buffer[$i]);
        }

        // Resolve max_features count
        $maxFeat = $this->resolveMaxFeatures($d);

        $maxDepth = $this->max_depth ?? PHP_INT_MAX;
        $estimators = [];

        for ($t = 0; $t < $this->n_estimators; $t++) {
            // ── Optional bootstrap sampling ────────────────────────────────
            if ($this->bootstrap) {
                $bootIdx = [];
                for ($i = 0; $i < $n; $i++) {
                    $bootIdx[] = mt_rand(0, $n - 1);
                }
                $Xboot  = new Tensor([$n, $d]);
                $yBoot  = [];
                for ($i = 0; $i < $n; $i++) {
                    $src    = $bootIdx[$i];
                    $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$src * $d]));
                    $dstPtr = \FFI::cast('float*', \FFI::addr($Xboot->buffer[$i * $d]));
                    $blas->cblas_scopy($d, $srcPtr, 1, $dstPtr, 1);
                    $yBoot[$i] = $yInt[$src];
                }
                $Xtrain  = $Xboot;
                $yTrain  = $yBoot;
                $nTrain  = $n;
                $indices = range(0, $n - 1);
            } else {
                $Xtrain  = $X;
                $yTrain  = $yInt;
                $nTrain  = $n;
                $indices = range(0, $n - 1);
            }

            $estimators[] = $this->buildTree(
                $Xtrain, $yTrain, $K, $allClasses, $classPos,
                $indices, $maxFeat, $maxDepth,
            );
        }

        $this->estimators_    = $estimators;
        $this->classes_       = $allClasses;
        $this->n_classes_     = $K;
        $this->n_features_in_ = $d;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict class labels: argmax of averaged predict_proba().
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    Integer class labels [n_samples]
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
                if ($v > $bestVal) { $bestVal = $v; $bestPos = $c; }
            }
            $out->buffer[$i] = (float) $this->classes_[$bestPos];
        }

        return $out;
    }

    /**
     * Predict averaged class probability distributions across all trees.
     *
     * Accumulation uses the same BLAS fast-path as RandomForestClassifier:
     *   cblas_saxpy(m*K, 1.0, treeProba, acc)  when all trees share K classes.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples, n_classes] probability matrix
     */
    public function predict_proba(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "ExtraTreesClassifier::predict_proba() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        $m    = $X->shape[0];
        $nC   = $this->n_classes_;
        $blas = BlasEngine::get()->ffi;

        $acc            = Tensor::zeros([$m, $nC]);
        $forestClassPos = array_flip($this->classes_);

        foreach ($this->estimators_ as $tree) {
            $treeProba = $this->predictTreeProba($tree, $X, $m);
            $treeNC    = $tree['n_classes'];

            if ($treeNC === $nC) {
                // Fast path: same class layout — single BLAS saxpy
                $blas->cblas_saxpy($m * $nC, 1.0, $treeProba->buffer, 1, $acc->buffer, 1);
            } else {
                // Slow path: scatter into forest-wide class positions
                $treeCls = $tree['classes'];
                for ($i = 0; $i < $m; $i++) {
                    $treeBase   = $i * $treeNC;
                    $forestBase = $i * $nC;
                    for ($tc = 0; $tc < $treeNC; $tc++) {
                        $fp = $forestClassPos[$treeCls[$tc]] ?? null;
                        if ($fp !== null) {
                            $acc->buffer[$forestBase + $fp] =
                                (float) $acc->buffer[$forestBase + $fp]
                                + (float) $treeProba->buffer[$treeBase + $tc];
                        }
                    }
                }
            }
        }

        // Normalise: divide by n_estimators
        $blas->cblas_sscal($m * $nC, 1.0 / $this->n_estimators, $acc->buffer, 1);

        return $acc;
    }

    /**
     * Accuracy score on test data.
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

    // ── Tree building ─────────────────────────────────────────────────────

    /**
     * Build one ExtraTree (random-threshold classification tree).
     *
     * For each node:
     *   1. Select max_features features randomly (partial Fisher-Yates).
     *   2. For each candidate feature j:
     *      a. Find min/max of X[:, j] in this node's samples.
     *      b. Draw a random threshold t ∈ [min_j, max_j].
     *      c. Evaluate Gini-purity score = Σ n_{k,L}²/n_L + Σ n_{k,R}²/n_R.
     *   3. Accept the best-scoring (feature, threshold); make leaf if none improves.
     *
     * Time per node: O(max_features × n_node) — no sort required.
     *
     * @param  Tensor  $X        Training feature matrix
     * @param  int[]   $yInt     Integer class labels (plain PHP array)
     * @param  int     $K        Total number of classes
     * @param  int[]   $classes  Sorted class labels
     * @param  array   $classPos label → index mapping
     * @param  int[]   $indices  Sample indices for root node
     * @return array             Flat tree associative array
     */
    private function buildTree(
        Tensor $X,
        array  $yInt,
        int    $K,
        array  $classes,
        array  $classPos,
        array  $indices,
        int    $maxFeat,
        int    $maxDepth,
    ): array {
        [$nFull, $d] = $X->shape;

        $cl     = [];  // children_left[node_id]
        $cr     = [];  // children_right[node_id]
        $feat   = [];  // feature[node_id]
        $thresh = [];  // threshold[node_id]
        $proba  = [];  // proba[node_id] = float[K]

        $nodeCount = 0;

        // Iterative DFS stack: [node_id, indices[], depth]
        $rootId = $nodeCount++;
        $stack  = [[$rootId, $indices, 0]];

        while ($stack !== []) {
            [$nodeId, $idxs, $depth] = array_pop($stack);
            $nNode = count($idxs);

            // ── Compute class counts ───────────────────────────────────────
            $counts = array_fill(0, $K, 0);
            foreach ($idxs as $idx) {
                $counts[$classPos[$yInt[$idx]]]++;
            }

            // Store normalised class frequencies at this node (for predict_proba)
            $nodeProba = [];
            for ($k = 0; $k < $K; $k++) {
                $nodeProba[$k] = $counts[$k] / $nNode;
            }
            $proba[$nodeId] = $nodeProba;

            // ── Leaf conditions ────────────────────────────────────────────
            $nNonZeroClasses = count(array_filter($counts));
            if ($depth >= $maxDepth
                || $nNode < $this->min_samples_split
                || $nNonZeroClasses <= 1
            ) {
                $cl[$nodeId]     = self::TREE_LEAF;
                $cr[$nodeId]     = self::TREE_LEAF;
                $feat[$nodeId]   = self::TREE_UNDEFINED;
                $thresh[$nodeId] = 0.0;
                continue;
            }

            // ── Select candidate features (partial Fisher-Yates) ──────────
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

            // ── Baseline Gini purity score (no split) ────────────────────
            $baseline = 0.0;
            for ($k = 0; $k < $K; $k++) {
                $baseline += $counts[$k] * $counts[$k];
            }
            $baseline /= $nNode;

            $bestScore = $baseline;
            $bestFeat  = self::TREE_UNDEFINED;
            $bestThresh = 0.0;

            // ── Random threshold search ───────────────────────────────────
            foreach ($selectedFeats as $fj) {
                // Find min/max of feature fj in this node
                $minVal =  PHP_FLOAT_MAX;
                $maxVal = -PHP_FLOAT_MAX;
                foreach ($idxs as $idx) {
                    $v = (float) $X->buffer[$idx * $d + $fj];
                    if ($v < $minVal) { $minVal = $v; }
                    if ($v > $maxVal) { $maxVal = $v; }
                }
                if ($maxVal - $minVal < 1e-9) {
                    continue;  // constant feature in this node — skip
                }

                // Draw a random threshold uniformly in (min_val, max_val)
                $t = $minVal + ($maxVal - $minVal) * mt_rand() / mt_getrandmax();

                // Compute left class counts and sizes in O(n_node)
                $countsL = array_fill(0, $K, 0);
                $nL      = 0;
                foreach ($idxs as $idx) {
                    if ((float) $X->buffer[$idx * $d + $fj] <= $t) {
                        $countsL[$classPos[$yInt[$idx]]]++;
                        $nL++;
                    }
                }
                $nR = $nNode - $nL;
                if ($nL === 0 || $nR === 0) {
                    continue;  // degenerate split — skip
                }

                // score = Σ n_{k,L}² / n_L + Σ n_{k,R}² / n_R
                $scoreL = 0.0;
                $scoreR = 0.0;
                for ($k = 0; $k < $K; $k++) {
                    $ckL = $countsL[$k];
                    $ckR = $counts[$k] - $ckL;
                    $scoreL += $ckL * $ckL;
                    $scoreR += $ckR * $ckR;
                }
                $score = $scoreL / $nL + $scoreR / $nR;

                if ($score > $bestScore) {
                    $bestScore  = $score;
                    $bestFeat   = $fj;
                    $bestThresh = $t;
                }
            }

            // ── Make leaf if no split improved ────────────────────────────
            if ($bestFeat === self::TREE_UNDEFINED) {
                $cl[$nodeId]     = self::TREE_LEAF;
                $cr[$nodeId]     = self::TREE_LEAF;
                $feat[$nodeId]   = self::TREE_UNDEFINED;
                $thresh[$nodeId] = 0.0;
                continue;
            }

            // ── Partition samples on the best split ───────────────────────
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

            // Push right first so left is popped first (pre-order)
            $stack[] = [$rightId, $rightIdx, $depth + 1];
            $stack[] = [$leftId,  $leftIdx,  $depth + 1];
        }

        return [
            'cl'        => $cl,
            'cr'        => $cr,
            'feat'      => $feat,
            'thresh'    => $thresh,
            'proba'     => $proba,
            'n_classes' => $K,
            'classes'   => $classes,
        ];
    }

    // ── Tree traversal ───────────────────────────────────────────────────

    /**
     * Predict per-sample class probability from one internal ExtraTree.
     * Returns a Tensor [n_samples, n_classes_tree] (flat row-major).
     *
     * @return Tensor [m, K_tree]
     */
    private function predictTreeProba(array $tree, Tensor $X, int $m): Tensor
    {
        $K    = $tree['n_classes'];
        $cl   = $tree['cl'];
        $cr   = $tree['cr'];
        $feat = $tree['feat'];
        $thr  = $tree['thresh'];
        $prob = $tree['proba'];
        $d    = $X->shape[1];

        $out = new Tensor([$m, $K]);

        for ($i = 0; $i < $m; $i++) {
            // Traverse to leaf
            $node = 0;
            while ($cl[$node] !== self::TREE_LEAF) {
                $node = ((float) $X->buffer[$i * $d + $feat[$node]] <= $thr[$node])
                    ? $cl[$node]
                    : $cr[$node];
            }
            // Copy leaf probabilities into output
            $base = $i * $K;
            foreach ($prob[$node] as $k => $p) {
                $out->buffer[$base + $k] = $p;
            }
        }

        return $out;
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private function resolveMaxFeatures(int $d): int
    {
        return match (true) {
            is_int($this->max_features)        => min($this->max_features, $d),
            $this->max_features === 'sqrt'      => max(1, (int) ceil(sqrt($d))),
            $this->max_features === 'log2'      => max(1, (int) ceil(log($d, 2))),
            $this->max_features === null        => $d,
            default                            => $d,
        };
    }

    private function checkFitted(): void
    {
        if (!isset($this->estimators_)) {
            throw new \RuntimeException(
                'ExtraTreesClassifier is not fitted. Call fit() first.'
            );
        }
    }
}
