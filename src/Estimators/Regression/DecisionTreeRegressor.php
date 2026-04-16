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

        // Fast C-level mean for the leaf node value.
        $leafValue = $y->mean();

        // Stopping criteria: max depth or minimum samples.
        if ($depth >= $this->maxDepth || $n < $this->minSamplesSplit) {
            return ['value' => $leafValue];
        }

        $split = $this->findBestSplit($x, $y);

        if ($split === null) {
            return ['value' => $leafValue];
        }

        // Extract the C boolean mask to a PHP array once — avoids repeated FFI
        // round-trips during routing; booleanIndex is unsafe on 2D $x matrices.
        $maskArray = $split['mask']->toFlatArray();
        unset($split['mask']);          // Return mask C-memory to pool immediately.

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

        // FLOAT32 index tensors (fromArray defaults to F32 — required by tensor_take).
        $leftT  = Tensor::fromArray($leftIdx);
        unset($leftIdx);
        $xLeft  = $x->take($leftT, 0);
        $yLeft  = $y->take($leftT, 0);
        unset($leftT);

        $rightT = Tensor::fromArray($rightIdx);
        unset($rightIdx);
        $xRight = $x->take($rightT, 0);
        $yRight = $y->take($rightT, 0);
        unset($rightT);

        // Recurse left; free branch data the instant recursion returns.
        $node['left'] = $this->buildTree($xLeft, $yLeft, $depth + 1);
        unset($xLeft, $yLeft);

        $node['right'] = $this->buildTree($xRight, $yRight, $depth + 1);
        unset($xRight, $yRight);

        return $node;
    }

    private function findBestSplit(Tensor $x, Tensor $y): ?array
    {
        $bestCost  = INF;
        $bestSplit = null;
        $n         = $y->size();

        // Feature bagging (colsample_bytree for GBM integration).
        $features = range(0, $this->nFeatures - 1);
        if ($this->maxFeatures !== null) {
            shuffle($features);
            $features = array_slice($features, 0, $this->maxFeatures);
        }

        foreach ($features as $feature) {
            // Zero-copy column view — points into existing C buffer, no allocation.
            $col = $x->col($feature);
            $min = $col->min();
            $max = $col->max();

            if ($min >= $max) {
                unset($col);
                continue;
            }

            // Pure-PHP threshold arithmetic: zero C allocations for threshold values.
            // 8 interior points uniformly spaced between min and max (excludes endpoints).
            $step = ($max - $min) / 9.0;

            for ($t = 1; $t <= 8; $t++) {
                $threshold = $min + $step * $t;

                // O(1) scalar comparison kernel — no threshold Tensor allocated.
                $mask  = $col->lessScalar($threshold);
                $leftY = $y->booleanIndex($mask);   // Safe: $y is 1D targets.
                $nLeft = $leftY->size();

                if ($nLeft === 0 || $nLeft === $n) {
                    unset($mask, $leftY);
                    continue;
                }

                // Avoid a second FFI call for $nRight — pure PHP arithmetic.
                $nRight    = $n - $nLeft;
                $rightMask = $mask->logicalNot();
                $rightY    = $y->booleanIndex($rightMask);
                unset($rightMask);              // Pool immediately; not needed again.

                // Weighted variance reduction — both variance() calls run in C/OpenBLAS.
                $cost = ($nLeft  / $n) * $leftY->variance()
                      + ($nRight / $n) * $rightY->variance();
                unset($leftY, $rightY);         // Pool label slices the instant cost is done.

                if ($cost < $bestCost) {
                    // Free the PREVIOUS best mask before overwriting to avoid leaking
                    // its C buffer until PHP's deferred GC runs.
                    if ($bestSplit !== null) {
                        unset($bestSplit['mask']);
                    }
                    $bestCost  = $cost;
                    $bestSplit = [
                        'feature'   => $feature,
                        'threshold' => $threshold,
                        'mask'      => $mask,   // Transfer ownership; do NOT unset here.
                    ];
                } else {
                    unset($mask);               // Not the best — return C-memory to pool now.
                }
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