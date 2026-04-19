<?php
declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Histogram-based Gradient Boosted Decision Tree Regressor (LightGBM-style).
 *
 * Uses MSE loss. All per-sample work runs in C. PHP manages the O(T * 2^depth)
 * BFS node loop only.
 */
final class GBDTRegressor implements Learner, Persistable
{
    private ?Tensor $boundaries  = null;
    private ?Tensor $treeFeats   = null;
    private ?Tensor $treeThresh  = null;
    private ?Tensor $treeLefts   = null;
    private ?Tensor $treeRights  = null;
    private ?Tensor $treeSizes   = null;
    private float   $baseScore   = 0.0;

    public function __construct(
        private readonly int   $nEstimators = 100,
        private readonly int   $maxDepth    = 4,
        private readonly int   $numBins     = 255,
        private readonly float $lr          = 0.1,
        private readonly float $lambda      = 1.0,
        private readonly float $gamma       = 0.0,
        private readonly float $minChildW   = 1.0
    ) {}

    public function train(Dataset $dataset): void
    {
        $y = $dataset->labels();
        if ($y === null) {
            throw new \InvalidArgumentException("GBDTRegressor requires labeled data.");
        }

        $X   = $dataset->samples();
        $N   = $X->shape()[0];
        $Q   = $this->numBins;

        $this->boundaries = Tensor::gbdtComputeBoundaries($X, $Q);
        $bins             = Tensor::gbdtBinSamples($X, $this->boundaries, $Q);

        $this->baseScore  = $y->sum() / $N;
        $preds            = Tensor::zeros($N)->addScalarInplace($this->baseScore);

        $maxNodes        = (1 << $this->maxDepth) * 2;
        $T               = $this->nEstimators;

        $featsArr  = array_fill(0, $T * $maxNodes, -1.0);
        $threshArr = array_fill(0, $T * $maxNodes, 0.0);
        $leftsArr  = array_fill(0, $T * $maxNodes, -1.0);
        $rightsArr = array_fill(0, $T * $maxNodes, -1.0);
        $sizesArr  = array_fill(0, $T, 0.0);

        for ($t = 0; $t < $T; $t++) {
            [$g, $h] = Tensor::gbdtMseGradHess($preds, $y);

            $nodeIdx = 0;
            $offset  = $t * $maxNodes;
            $queue   = [[Tensor::ones($N), 0, 0]];

            while (!empty($queue)) {
                [$mask, $nodeId, $depth] = array_shift($queue);

                $sumG  = $mask->mul($g)->sum();
                $sumH  = $mask->mul($h)->sum();
                $nodeN = (int)$mask->sum();

                if ($depth >= $this->maxDepth || $nodeN < (int)(2 * $this->minChildW)) {
                    $leaf = -$sumG / ($sumH + $this->lambda);
                    $featsArr[$offset + $nodeId]  = -1.0;
                    $threshArr[$offset + $nodeId] = $leaf;
                    $nodeIdx = max($nodeIdx, $nodeId + 1);
                    continue;
                }

                [$histG, $histH] = Tensor::gbdtHistogram($bins, $g, $h, $mask, $Q);
                [$feat, $bin, $gain] = Tensor::gbdtBestSplit(
                    $histG, $histH, $Q, $sumG, $sumH, $nodeN, $this->lambda, $this->gamma
                );

                if ($feat < 0 || $gain <= 0.0) {
                    $leaf = -$sumG / ($sumH + $this->lambda);
                    $featsArr[$offset + $nodeId]  = -1.0;
                    $threshArr[$offset + $nodeId] = $leaf;
                    $nodeIdx = max($nodeIdx, $nodeId + 1);
                    continue;
                }

                $leftId  = $nodeIdx + 1;
                $rightId = $nodeIdx + 2;
                $nodeIdx += 2;

                $featsArr[$offset + $nodeId]  = (float)$feat;
                $threshArr[$offset + $nodeId] = (float)$bin;
                $leftsArr[$offset + $nodeId]  = (float)$leftId;
                $rightsArr[$offset + $nodeId] = (float)$rightId;

                [$leftMask, $rightMask] = Tensor::gbdtSplitNode($bins, $mask, $feat, $bin);
                $queue[] = [$leftMask,  $leftId,  $depth + 1];
                $queue[] = [$rightMask, $rightId, $depth + 1];
            }

            $sizesArr[$t] = (float)($nodeIdx + 1);

            // Recompute preds for next iteration
            $tF2 = Tensor::fromArray($featsArr)->reshape($T, $maxNodes);
            $tT2 = Tensor::fromArray($threshArr)->reshape($T, $maxNodes);
            $tL2 = Tensor::fromArray($leftsArr)->reshape($T, $maxNodes);
            $tR2 = Tensor::fromArray($rightsArr)->reshape($T, $maxNodes);
            $tS  = Tensor::fromArray($sizesArr);

            $preds = Tensor::gbdtPredictAll(
                $bins,
                $tF2->slice(0, 0, $t + 1),
                $tT2->slice(0, 0, $t + 1),
                $tL2->slice(0, 0, $t + 1),
                $tR2->slice(0, 0, $t + 1),
                $tS->slice(0, 0, $t + 1),
                $this->baseScore
            )->mulScalarInplace($this->lr);
        }

        $this->treeFeats  = Tensor::fromArray($featsArr)->reshape($T, $maxNodes);
        $this->treeThresh = Tensor::fromArray($threshArr)->reshape($T, $maxNodes);
        $this->treeLefts  = Tensor::fromArray($leftsArr)->reshape($T, $maxNodes);
        $this->treeRights = Tensor::fromArray($rightsArr)->reshape($T, $maxNodes);
        $this->treeSizes  = Tensor::fromArray($sizesArr);
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("GBDTRegressor is not trained.");
        }
        $bins = Tensor::gbdtBinSamples($dataset->samples(), $this->boundaries, $this->numBins);
        return Tensor::gbdtPredictAll(
            $bins,
            $this->treeFeats, $this->treeThresh,
            $this->treeLefts, $this->treeRights,
            $this->treeSizes, $this->baseScore
        )->mulScalarInplace($this->lr);
    }

    public function trained(): bool
    {
        return $this->treeFeats !== null;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['nEstimators'=>$this->nEstimators,'maxDepth'=>$this->maxDepth,'numBins'=>$this->numBins,'lr'=>$this->lr,'lambda'=>$this->lambda,'gamma'=>$this->gamma,'minChildW'=>$this->minChildW,'baseScore'=>$this->baseScore]));
        if ($this->treeFeats !== null) {
            SafeTensorsIO::save($dir . '/model.safetensors', [
                'boundaries'  => $this->boundaries,
                'tree_feats'  => $this->treeFeats,
                'tree_thresh' => $this->treeThresh,
                'tree_lefts'  => $this->treeLefts,
                'tree_rights' => $this->treeRights,
                'tree_sizes'  => $this->treeSizes,
            ]);
        }
    }
    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((int)$c['nEstimators'],(int)$c['maxDepth'],(int)$c['numBins'],(float)$c['lr'],(float)$c['lambda'],(float)$c['gamma'],(float)$c['minChildW']);
        $i->baseScore = (float)$c['baseScore'];
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) {
            $t = SafeTensorsIO::load($stPath);
            $i->boundaries  = $t['boundaries']  ?? null;
            $i->treeFeats   = $t['tree_feats']  ?? null;
            $i->treeThresh  = $t['tree_thresh'] ?? null;
            $i->treeLefts   = $t['tree_lefts']  ?? null;
            $i->treeRights  = $t['tree_rights'] ?? null;
            $i->treeSizes   = $t['tree_sizes']  ?? null;
        }
        return $i;
    }
}
