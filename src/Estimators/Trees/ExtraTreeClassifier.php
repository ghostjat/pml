<?php

declare(strict_types=1);

namespace Pml\Estimators\Trees;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Extremely Randomized Classification Tree (ExtraTree).
 * * JIT & Memory Optimized:
 * - Skips exhaustive threshold searching. Evaluates exactly ONE random threshold per feature.
 * - Builds tree topologies exponentially faster than standard CART algorithms.
 */
final class ExtraTreeClassifier implements Learner, Persistable
{
    private int $maxDepth;
    private int $minSamplesSplit;
    private ?int $maxFeatures;
    
    private ?array $tree = null;
    private int $nFeatures = 0;

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
            throw new \InvalidArgumentException("ExtraTree requires labeled data.");
        }

        $this->nFeatures = $x->shape()[1];
        $this->tree = $this->buildTree($x, $y, 0);
    }

    private function buildTree(Tensor $x, Tensor $y, int $depth): array
    {
        $n = $y->size();
        $counts = $y->bincount()->toFlatArray();
        
        $maxCount = -1;
        $majorityClass = 0;
        foreach ($counts as $class => $count) {
            if ($count > $maxCount) {
                $maxCount = $count;
                $majorityClass = $class;
            }
        }

        if ($depth >= $this->maxDepth || $n < $this->minSamplesSplit || $maxCount == $n) {
            return ['class' => $majorityClass];
        }

        $split = $this->findRandomSplit($x, $y);
        
        if (!$split) {
            return ['class' => $majorityClass];
        }

        $maskArray = $split['mask']->toFlatArray();
        $leftIdx = [];
        $rightIdx = [];
        
        foreach ($maskArray as $i => $val) {
            if ($val > 0.5) $leftIdx[] = $i; 
            else $rightIdx[] = $i;
        }
        
        unset($split['mask']); // Clean C-pointer early

        $leftT = Tensor::fromArray($leftIdx);
        $rightT = Tensor::fromArray($rightIdx);

        return [
            'feature'   => $split['feature'],
            'threshold' => $split['threshold'],
            'left'      => $this->buildTree($x->take($leftT, 0), $y->take($leftT, 0), $depth + 1),
            'right'     => $this->buildTree($x->take($rightT, 0), $y->take($rightT, 0), $depth + 1)
        ];
    }

    /**
     * ExtraTrees Core: Selects completely random thresholds without exhaustive search.
     */
    private function findRandomSplit(Tensor $x, Tensor $y): ?array
    {
        $bestGini = INF;
        $bestSplit = null;
        $n = $y->size();

        $features = range(0, $this->nFeatures - 1);
        if ($this->maxFeatures !== null) {
            shuffle($features);
            $features = array_slice($features, 0, $this->maxFeatures);
        }

        foreach ($features as $feature) {
            $col = $x->col($feature);
            $min = $col->min();
            $max = $col->max();

            if ($min === $max) continue;

            // Generate exactly ONE random threshold
            $threshold = $min + (lcg_value() * ($max - $min));

            $threshT = Tensor::zeros($n)->addScalarInplace($threshold);
            $mask = $col->less($threshT);

            $leftY = $y->booleanIndex($mask);
            $nLeft = $leftY->size();

            if ($nLeft === 0 || $nLeft === $n) continue;

            $rightMask = $mask->logicalNot();
            $rightY = $y->booleanIndex($rightMask);
            $nRight = $rightY->size();

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

        return $bestSplit;
    }

    private function gini(Tensor $y): float
    {
        $n = $y->size();
        if ($n === 0) return 0.0;
        
        $sumSq = $y->bincount()->square()->sum();
        return 1.0 - ($sumSq / ($n * $n));
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("ExtraTree is not trained.");
        }

        $flatX = $dataset->samples()->toFlatArray();
        $rows = $dataset->numRows();
        $cols = $dataset->numColumns();
        
        $preds = [];

        for ($i = 0; $i < $rows; $i++) {
            $node = $this->tree;
            $rowOffset = $i * $cols;
            
            while (isset($node['feature'])) {
                $val = $flatX[$rowOffset + $node['feature']];
                $node = ($val < $node['threshold']) ? $node['left'] : $node['right'];
            }
            $preds[] = $node['class'];
        }

        return Tensor::fromArray($preds);
    }

    public function trained(): bool
    {
        return $this->tree !== null;
    }

    public function exportPhpTree(): array
    {
        return ['tree' => $this->tree, 'nFeatures' => $this->nFeatures, 'maxDepth' => $this->maxDepth, 'minSamplesSplit' => $this->minSamplesSplit, 'maxFeatures' => $this->maxFeatures];
    }

    public static function fromPhpTree(array $data): self
    {
        $i = new self((int) $data['maxDepth'], (int) $data['minSamplesSplit'], isset($data['maxFeatures']) ? (int) $data['maxFeatures'] : null);
        $i->nFeatures = (int) $data['nFeatures'];
        $i->tree = $data['tree'];
        return $i;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/tree.json', json_encode($this->exportPhpTree()));
    }

    public static function load(string $dir): self
    {
        return self::fromPhpTree(json_decode(file_get_contents($dir . '/tree.json'), true));
    }
}