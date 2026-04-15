<?php

declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Classification and Regression Tree (CART) for Continuous Targets.
 * The core weak-learner for Gradient Boosting.
 * * JIT & Memory Optimized:
 * - Employs C-Level Variance reductions to evaluate splits without PHP loops.
 * - Hardware Split Routing avoids N-sized PHP array evaluations.
 * - Scalar comparison kernels avoid temporary tensor allocations.
 */
final class DecisionTreeRegressor implements Learner
{
    private int $maxDepth;
    private int $minSamplesSplit;
    private ?int $maxFeatures;
    
    private ?array $tree = null;
    private int $nFeatures = 0;

    /** Flat C-side HardwareNode array for zero-PHP-loop prediction. */
    private ?\FFI\CData $hardwareNodes    = null;
    private int          $numHardwareNodes = 0;

    public function __construct(int $maxDepth = 3, int $minSamplesSplit = 2, ?int $maxFeatures = null)
    {
        // GBM weak learners are typically very shallow (depth 3-5)
        $this->maxDepth = $maxDepth;
        $this->minSamplesSplit = $minSamplesSplit;
        $this->maxFeatures = $maxFeatures;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $y = $dataset->labels();

        if ($y === null) {
            throw new \InvalidArgumentException("Decision Tree Regressor requires labeled continuous data.");
        }

        $this->nFeatures = $x->shape()[1];
        $this->tree      = $this->buildTree($x, $y, 0);
        $this->buildHardwareNodes();
    }

    private function buildTree(Tensor $x, Tensor $y, int $depth): array
    {
        $n = $y->size();
        
        // Fast C-level average for the leaf node value
        $leafValue = $y->mean();

        // Stopping Criteria
        if ($depth >= $this->maxDepth || $n < $this->minSamplesSplit) {
            return ['value' => $leafValue];
        }

        $split = $this->findBestSplit($x, $y);
        
        // If no split improves the variance, become a leaf
        if (!$split) {
            return ['value' => $leafValue];
        }

        // --- CRITICAL OPTIMIZATION: Hardware Split Routing ---
        // Replaces the O(N) PHP foreach loop with an AVX2 C-kernel.
        // Directly returns the Left and Right INT32 index tensors.
        [$leftT, $rightT] = $split['mask']->splitIndices();
        
        // Eagerly clean up the mask pointer to free memory pool block
        unset($split['mask']); 

        $node = [
            'feature'   => $split['feature'],
            'threshold' => $split['threshold']
        ];

        // Zero-copy view extraction via tensor_take
        $node['left']  = $this->buildTree($x->take($leftT, 0), $y->take($leftT, 0), $depth + 1);
        $node['right'] = $this->buildTree($x->take($rightT, 0), $y->take($rightT, 0), $depth + 1);

        return $node;
    }

    private function findBestSplit(Tensor $x, Tensor $y): ?array
    {
        $bestCost = INF;
        $bestSplit = null;
        $n = $y->size();

        // Feature Bagging (Colsample_bytree)
        $features = range(0, $this->nFeatures - 1);
        if ($this->maxFeatures !== null) {
            shuffle($features);
            $features = array_slice($features, 0, $this->maxFeatures);
        }

        foreach ($features as $feature) {
            $col = $x->col($feature);
            $min = $col->min();
            $max = $col->max();

            if ($min === $max) {
                unset($col);
                continue;
            }

            // Randomized Split Search: Evaluate evenly spaced thresholds natively in C.
            $thresholds = Tensor::linspace($min, $max, 10)->toFlatArray();
            $numThresholds = count($thresholds);

            // Skip min/max boundaries
            for ($i = 1; $i < $numThresholds - 1; $i++) {
                $threshold = $thresholds[$i];

                // --- CRITICAL OPTIMIZATION: Scalar SIMD Comparison ---
                // Replaces Tensor::zeros($n)->addScalarInplace() with an O(1) alloc scalar kernel.
                $mask = $col->lessScalar($threshold);

                $leftY = $y->booleanIndex($mask);
                $nLeft = $leftY->size();

                if ($nLeft === 0 || $nLeft === $n) {
                    unset($mask, $leftY);
                    continue;
                }

                $rightMask = $mask->logicalNot();
                $rightY = $y->booleanIndex($rightMask);
                $nRight = $rightY->size();

                // Cost is measured by the weighted Mean Squared Error (Variance) of the child nodes
                // ->variance() executes entirely in OpenBLAS C-memory
                $varLeft = $leftY->variance();
                $varRight = $rightY->variance();

                $cost = ($nLeft / $n) * $varLeft + ($nRight / $n) * $varRight;

                if ($cost < $bestCost) {
                    $bestCost = $cost;
                    $bestSplit = [
                        'feature'   => $feature,
                        'threshold' => $threshold,
                        'mask'      => $mask // Keep only the best mask
                    ];
                } else {
                    // Eager garbage collection to prevent memory ballooning
                    unset($mask);
                }
                
                unset($leftY, $rightMask, $rightY);
            }
            unset($col);
        }

        return $bestSplit;
    }

    /**
     * Serialize the PHP tree into a flat BFS-ordered HardwareNode C array.
     * Called once after training; enables zero-PHP-loop prediction via C kernel.
     */
    private function buildHardwareNodes(): void
    {
        $ffi = \Pml\Lib\TensorEngine::get();

        $nodeData = [];
        $queue    = [[$this->tree, -1, false]];
        $head     = 0;
        $nextIdx  = 0;

        while ($head < count($queue)) {
            [$phpNode, $parentIdx, $isLeft] = $queue[$head++];
            $myIdx = $nextIdx++;

            if ($parentIdx >= 0) {
                if ($isLeft) {
                    $nodeData[$parentIdx][2] = $myIdx;
                } else {
                    $nodeData[$parentIdx][3] = $myIdx;
                }
            }

            if (isset($phpNode['feature'])) {
                $nodeData[$myIdx] = [$phpNode['feature'], (float)$phpNode['threshold'], -1, -1, 0.0];
                $queue[] = [$phpNode['left'],  $myIdx, true];
                $queue[] = [$phpNode['right'], $myIdx, false];
            } else {
                $nodeData[$myIdx] = [-1, 0.0, -1, -1, (float)($phpNode['value'] ?? 0.0)];
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
            throw new RuntimeException("Decision Tree Regressor is not trained.");
        }

        $ffi      = \Pml\Lib\TensorEngine::get();
        $rows     = $dataset->numRows();

        $shape    = $ffi->new('int[1]');
        $shape[0] = $rows;
        $out = Tensor::wrap($ffi->tensor_create_dtype(1, $ffi->cast('int*', $shape), Tensor::DTYPE_FLOAT32));

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