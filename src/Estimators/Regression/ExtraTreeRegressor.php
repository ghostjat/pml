<?php
declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Extra-Tree Regressor — Extremely Randomized Tree for regression.
 * Splits are chosen at random thresholds instead of optimal thresholds,
 * giving O(N) training per node instead of O(N log N).
 *
 * JIT & Memory Optimized:
 * - Random threshold selection avoids expensive C-level sort on every feature.
 * - All label reductions (mean) are single C calls.
 * - Tree traversal at predict time uses JIT-compiled PHP array indexing.
 */
final class ExtraTreeRegressor implements Learner
{
    private ?array $tree    = null;
    private int    $nFeatures = 0;

    public function __construct(
        private readonly int  $maxDepth       = 10,
        private readonly int  $minSamplesSplit = 2,
        private readonly ?int $maxFeatures     = null
    ) {}

    public function train(Dataset $dataset): void
    {
        if ($dataset->labels() === null) {
            throw new \InvalidArgumentException("ExtraTreeRegressor requires labeled data.");
        }
        $this->nFeatures = $dataset->numColumns();
        $this->tree      = $this->buildTree($dataset->samples(), $dataset->labels(), 0);
    }

    private function buildTree(Tensor $x, Tensor $y, int $depth): array
    {
        $n    = $y->size();
        $mean = $y->mean();                                    // single C call

        if ($depth >= $this->maxDepth || $n < $this->minSamplesSplit) {
            return ['value' => $mean];
        }

        $split = $this->findRandomSplit($x, $y);
        if (!$split) {
            return ['value' => $mean];
        }

        $maskArray = $split['mask']->toFlatArray();
        $leftIdx   = [];
        $rightIdx  = [];
        foreach ($maskArray as $i => $v) {
            if ($v > 0.5) $leftIdx[] = $i;
            else $rightIdx[] = $i;
        }
        unset($split['mask']);

        if (empty($leftIdx) || empty($rightIdx)) {
            return ['value' => $mean];
        }

        $lT = Tensor::fromArray($leftIdx);
        $rT = Tensor::fromArray($rightIdx);

        return [
            'feature'   => $split['feature'],
            'threshold' => $split['threshold'],
            'left'      => $this->buildTree($x->take($lT, 0), $y->take($lT, 0), $depth + 1),
            'right'     => $this->buildTree($x->take($rT, 0), $y->take($rT, 0), $depth + 1),
        ];
    }

    private function findRandomSplit(Tensor $x, Tensor $y): ?array
    {
        $n        = $y->size();
        $features = range(0, $this->nFeatures - 1);

        if ($this->maxFeatures !== null) {
            shuffle($features);
            $features = array_slice($features, 0, $this->maxFeatures);
        }

        $bestVar  = INF;
        $bestSplit = null;

        foreach ($features as $feature) {
            $col = $x->col($feature);
            $min = $col->min();
            $max = $col->max();
            if ($min >= $max) continue;

            // Single random threshold — the "Extra" in Extra-Trees
            $threshold = $min + mt_rand() / mt_getrandmax() * ($max - $min);
            $threshT   = Tensor::zeros($n)->addScalarInplace($threshold);
            $mask      = $col->less($threshT);

            $leftY  = $y->booleanIndex($mask);
            $nLeft  = $leftY->size();
            if ($nLeft === 0 || $nLeft === $n) continue;

            $rightMask = $mask->logicalNot();
            $rightY    = $y->booleanIndex($rightMask);
            $nRight    = $rightY->size();

            // Weighted variance reduction
            $varLeft  = $nLeft  > 1 ? $leftY->variance()  : 0.0;
            $varRight = $nRight > 1 ? $rightY->variance() : 0.0;
            $wVar     = ($nLeft / $n) * $varLeft + ($nRight / $n) * $varRight;

            if ($wVar < $bestVar) {
                $bestVar   = $wVar;
                $bestSplit = ['feature' => $feature, 'threshold' => $threshold, 'mask' => $mask];
            }
        }

        return $bestSplit;
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("ExtraTreeRegressor is not trained.");
        }
        $flat  = $dataset->samples()->toFlatArray();
        $rows  = $dataset->numRows();
        $cols  = $dataset->numColumns();
        $preds = [];

        for ($i = 0; $i < $rows; $i++) {
            $node   = $this->tree;
            $offset = $i * $cols;
            while (isset($node['feature'])) {
                $node = $flat[$offset + $node['feature']] < $node['threshold']
                    ? $node['left'] : $node['right'];
            }
            $preds[] = $node['value'];
        }

        return Tensor::fromArray($preds);
    }

    public function trained(): bool
    {
        return $this->tree !== null;
    }
}
