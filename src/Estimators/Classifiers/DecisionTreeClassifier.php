<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Classification and Regression Tree (CART).
 * * JIT & Memory Optimized:
 * - Uses AVX2 C-tensors to compute Gini impurities simultaneously.
 * - Extracts 1D boolean masks to PHP to rapidly route tree topology via JIT.
 * - Employs Randomized Split searching to bypass exhaustive O(N^2) loops.
 */
final class DecisionTreeClassifier implements Learner
{
    private int $maxDepth;
    private int $minSamplesSplit;
    private ?int $maxFeatures;

    private ?array $tree = null;
    private int $nFeatures = 0;

    /** Flat C-side HardwareNode array for zero-PHP-loop prediction. */
    private ?\FFI\CData $hardwareNodes   = null;
    private int          $numHardwareNodes = 0;

    public function __construct(int $maxDepth = 10, int $minSamplesSplit = 2, ?int $maxFeatures = null)
    {
        $this->maxDepth = $maxDepth;
        $this->minSamplesSplit = $minSamplesSplit;
        $this->maxFeatures = $maxFeatures;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $y = $dataset->labels();

        if ($y === null) {
            throw new \InvalidArgumentException("Decision Tree requires labeled data.");
        }

        $this->nFeatures = $x->shape()[1];
        $this->tree      = $this->buildTree($x, $y, 0);
        $this->buildHardwareNodes();
    }

    /**
     * Recursively builds the tree topology.
     */

private function buildTree(Tensor $x, Tensor $y, int $depth): array
{
    $n = $y->size();

    // Explicit temp variable so the bincount C-buffer is freed the instant
    // toFlatArray() returns — not deferred to end-of-scope GC.
    $bincountT     = $y->bincount();
    $counts        = $bincountT->toFlatArray();
    unset($bincountT);

    $maxCount      = -1;
    $majorityClass = 0;
    foreach ($counts as $class => $count) {
        if ($count > $maxCount) {
            $maxCount      = $count;
            $majorityClass = $class;
        }
    }
    unset($counts);

    // Stopping criteria: max depth, minimum samples, or pure node.
    if ($depth >= $this->maxDepth || $n < $this->minSamplesSplit || $maxCount === $n) {
        return ['class' => $majorityClass];
    }

    $split = $this->findBestSplit($x, $y);

    if ($split === null) {
        return ['class' => $majorityClass];
    }

    // Extract the C boolean mask to a PHP array once — avoids repeated FFI
    // round-trips during left/right routing (rule 2: no booleanIndex on 2D $x).
    $maskArray = $split['mask']->toFlatArray();
    unset($split['mask']);          // Immediately return mask C-memory to pool.

    $leftIdx  = [];
    $rightIdx = [];
    foreach ($maskArray as $i => $val) {
        if ($val > 0.5) {
            $leftIdx[]  = $i;
        } else {
            $rightIdx[] = $i;
        }
    }
    unset($maskArray);

    $node = [
        'feature'   => $split['feature'],
        'threshold' => $split['threshold'],
    ];

    // FLOAT32 index tensors (rule 3: no INT32 cast; fromArray defaults to F32).
    // take() on the 2D matrix is the only safe 2D routing path (rule 2).
    $leftT  = Tensor::fromArray($leftIdx);
    unset($leftIdx);
    $xLeft  = $x->take($leftT, 0);
    $yLeft  = $y->take($leftT, 0);
    unset($leftT);                  // Pool $leftT immediately after both takes.

    $rightT = Tensor::fromArray($rightIdx);
    unset($rightIdx);
    $xRight = $x->take($rightT, 0);
    $yRight = $y->take($rightT, 0);
    unset($rightT);                 // Pool $rightT immediately after both takes.

    // Recurse left; free branch data the instant recursion returns.
    $node['left'] = $this->buildTree($xLeft, $yLeft, $depth + 1);
    unset($xLeft, $yLeft);

    // Recurse right; same pattern.
    $node['right'] = $this->buildTree($xRight, $yRight, $depth + 1);
    unset($xRight, $yRight);

    return $node;
}

private function findBestSplit(Tensor $x, Tensor $y): ?array
{
    $bestGini  = INF;
    $bestSplit = null;
    $n         = $y->size();

    // Feature bagging (Random Forest integration).
    $features = range(0, $this->nFeatures - 1);
    if ($this->maxFeatures !== null) {
        shuffle($features);
        $features = array_slice($features, 0, $this->maxFeatures);
    }

    foreach ($features as $feature) {
        // Zero-copy column view — no C allocation, points into existing buffer.
        $col = $x->col($feature);
        $min = $col->min();
        $max = $col->max();

        if ($min >= $max) {
            unset($col);
            continue;
        }

        // Pure-PHP threshold arithmetic: zero C allocations (rule 1).
        // 8 interior points uniformly spaced between min and max (excludes endpoints).
        $step = ($max - $min) / 9.0;

        for ($t = 1; $t <= 8; $t++) {
            $threshold = $min + $step * $t;

            // O(1) scalar comparison — no threshold Tensor allocated (rule 1).
            $mask  = $col->lessScalar($threshold);
            $leftY = $y->booleanIndex($mask);   // Safe: $y is 1D labels.
            $nLeft = $leftY->size();

            if ($nLeft === 0 || $nLeft === $n) {
                unset($mask, $leftY);
                continue;
            }

            // Avoid a second C call for $nRight — pure PHP arithmetic.
            $nRight    = $n - $nLeft;
            $rightMask = $mask->logicalNot();
            $rightY    = $y->booleanIndex($rightMask);
            unset($rightMask);              // Pool immediately; not needed again.

            $gini = ($nLeft  / $n) * $this->gini($leftY)
                  + ($nRight / $n) * $this->gini($rightY);
            unset($leftY, $rightY);         // Pool label slices the instant Gini is done.

            if ($gini < $bestGini) {
                // Explicitly free the PREVIOUS best mask before overwriting
                // (rule 4: zero leak — PHP won't GC the old entry until later otherwise).
                if ($bestSplit !== null) {
                    unset($bestSplit['mask']);
                }
                $bestGini  = $gini;
                $bestSplit = [
                    'feature'   => $feature,
                    'threshold' => $threshold,
                    'mask'      => $mask,   // Transfer ownership; do NOT unset here.
                ];
            } else {
                unset($mask);               // Not the best — return C-memory to pool now.
            }
        }

        unset($col);                        // Pool column view after all thresholds tried.
    }

    return $bestSplit;
}

    /**
     * Gini Impurity: 1.0 - sum( (count_i / N)^2 )
     * Calculated entirely in C-memory without loops.
     */
    private function gini(Tensor $y): float
    {
        $n = $y->size();
        if ($n === 0) return 0.0;
        
        $counts = $y->bincount();
        $sumSq = $counts->square()->sum(); // C-Level reduction
        
        return 1.0 - ($sumSq / ($n * $n));
    }

    /**
     * Serialize the PHP tree into a flat BFS-ordered HardwareNode C array.
     * Called once after training; enables zero-PHP-loop prediction via C kernel.
     */
    private function buildHardwareNodes(): void
    {
        $ffi = \Pml\Lib\TensorEngine::get();

        // BFS traversal — collect flat node descriptors, back-fill child indices.
        $nodeData = [];   // [feature_idx, threshold, left_idx, right_idx, value]
        $queue    = [[$this->tree, -1, false]];
        $head     = 0;
        $nextIdx  = 0;

        while ($head < count($queue)) {
            [$phpNode, $parentIdx, $isLeft] = $queue[$head++];
            $myIdx = $nextIdx++;

            if ($parentIdx >= 0) {
                if ($isLeft) {
                    $nodeData[$parentIdx][2] = $myIdx;   // parent.left_idx
                } else {
                    $nodeData[$parentIdx][3] = $myIdx;   // parent.right_idx
                }
            }

            if (isset($phpNode['feature'])) {
                $nodeData[$myIdx] = [$phpNode['feature'], (float)$phpNode['threshold'], -1, -1, 0.0];
                $queue[] = [$phpNode['left'],  $myIdx, true];
                $queue[] = [$phpNode['right'], $myIdx, false];
            } else {
                $nodeData[$myIdx] = [-1, 0.0, -1, -1, (float)($phpNode['class'] ?? 0)];
            }
        }

        $count = count($nodeData);
        $this->numHardwareNodes = $count;
        $this->hardwareNodes    = $ffi->new("HardwareNode[$count]");

        foreach ($nodeData as $i => [$feat, $thresh, $left, $right, $val]) {
            $this->hardwareNodes[$i]->feature_idx = $feat;
            $this->hardwareNodes[$i]->threshold   = $thresh;
            $this->hardwareNodes[$i]->left_idx    = $left;
            $this->hardwareNodes[$i]->right_idx   = $right;
            $this->hardwareNodes[$i]->value        = $val;
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Decision Tree is not trained.");
        }

        $ffi   = \Pml\Lib\TensorEngine::get();
        $rows  = $dataset->numRows();

        // Pre-allocate output [N] tensor — C writes directly into it.
        $shape    = $ffi->new('int[1]');
        $shape[0] = $rows;
        $out = Tensor::wrap($ffi->tensor_create_dtype(1, $ffi->cast('int*', $shape), Tensor::DTYPE_FLOAT32));

        // Single C call — no PHP loop, no toFlatArray().
        $ffi->tensor_hardware_tree_predict(
            $dataset->samples()->ptr,
            $ffi->cast('HardwareNode*', $this->hardwareNodes),
            $out->ptr
        );

        return $out;
    }

    public function trained(): bool
    {
        return $this->tree !== null;
    }

    /**
     * Export the pure-PHP tree structure for external persistence.
     * The returned array contains no FFI\CData — safe to JSON-encode.
     */
    public function exportPhpTree(): array
    {
        return [
            'tree'            => $this->tree,
            'nFeatures'       => $this->nFeatures,
            'maxDepth'        => $this->maxDepth,
            'minSamplesSplit' => $this->minSamplesSplit,
            'maxFeatures'     => $this->maxFeatures,
        ];
    }

    /**
     * Reconstruct a trained instance from an exported PHP tree.
     * Rebuilds the FFI HardwareNode array internally.
     */
    public static function fromPhpTree(array $data): self
    {
        $instance = new self(
            (int) $data['maxDepth'],
            (int) $data['minSamplesSplit'],
            isset($data['maxFeatures']) ? (int) $data['maxFeatures'] : null
        );
        $instance->nFeatures = (int) $data['nFeatures'];
        $instance->tree      = $data['tree'];
        $instance->buildHardwareNodes();

        return $instance;
    }
}