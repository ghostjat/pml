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
        
        // Fast C-level class counting
        $counts = $y->bincount()->toFlatArray();
        
        $maxCount = -1;
        $majorityClass = 0;
        foreach ($counts as $class => $count) {
            if ($count > $maxCount) {
                $maxCount = $count;
                $majorityClass = $class;
            }
        }

        // Stopping Criteria: Max Depth, Min Samples, or Pure Node (all one class)
        if ($depth >= $this->maxDepth || $n < $this->minSamplesSplit || $maxCount == $n) {
            return ['class' => $majorityClass];
        }

        $split = $this->findBestSplit($x, $y);
        
        if (!$split) {
            return ['class' => $majorityClass];
        }

        // Unpack the C-mask to physically route the indices in PHP
        $maskArray = $split['mask']->toFlatArray();
        $leftIdx = [];
        $rightIdx = [];
        
        foreach ($maskArray as $i => $val) {
            if ($val > 0.5) $leftIdx[] = $i; 
            else $rightIdx[] = $i;
        }

        // Clean up the mask pointer
        unset($split['mask']);

        // Generate C-pointers for the next depth layer
        $leftT = Tensor::fromArray($leftIdx);
        $rightT = Tensor::fromArray($rightIdx);

        $node = [
            'feature'   => $split['feature'],
            'threshold' => $split['threshold']
        ];

        // Zero-copy view extraction via tensor_take
        $node['left']  = $this->buildTree($x->take($leftT, 0), $y->take($leftT, 0), $depth + 1);
        $node['right'] = $this->buildTree($x->take($rightT, 0), $y->take($rightT, 0), $depth + 1);

        return $node;
    }

    /**
     * Searches for the optimal feature split using C-Accelerated Gini Impurity.
     */
    private function findBestSplit(Tensor $x, Tensor $y): ?array
    {
        $bestGini = INF;
        $bestSplit = null;
        $n = $y->size();

        // Feature Bagging (crucial for Random Forests)
        $features = range(0, $this->nFeatures - 1);
        if ($this->maxFeatures !== null) {
            shuffle($features);
            $features = array_slice($features, 0, $this->maxFeatures);
        }

        foreach ($features as $feature) {
            // Zero-copy column slice
            $col = $x->col($feature);
            $min = $col->min();
            $max = $col->max();

            if ($min === $max) continue;

            // Randomized Split Search: Evaluate 10 evenly spaced thresholds natively in C.
            // This bypasses the O(N^2) pure-PHP sorting bottleneck.
            $thresholds = Tensor::linspace($min, $max, 10)->toFlatArray();
            array_pop($thresholds); // Exclude absolute max
            array_shift($thresholds); // Exclude absolute min

            foreach ($thresholds as $threshold) {
                // Generate threshold tensor and AVX2 Boolean Mask
                $threshT = Tensor::zeros($n)->addScalarInplace($threshold);
                $mask = $col->less($threshT);

                // Zero-copy label extraction
                $leftY = $y->booleanIndex($mask);
                $nLeft = $leftY->size();

                if ($nLeft === 0 || $nLeft === $n) continue;

                $rightMask = $mask->logicalNot();
                $rightY = $y->booleanIndex($rightMask);
                $nRight = $rightY->size();

                // Compute Gini heavily parallelized in C
                $giniLeft = $this->gini($leftY);
                $giniRight = $this->gini($rightY);

                $gini = ($nLeft / $n) * $giniLeft + ($nRight / $n) * $giniRight;

                if ($gini < $bestGini) {
                    $bestGini = $gini;
                    $bestSplit = [
                        'feature'   => $feature,
                        'threshold' => $threshold,
                        'mask'      => $mask
                    ];
                }
            }
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
}