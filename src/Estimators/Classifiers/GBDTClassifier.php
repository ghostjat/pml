<?php
declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use Pml\Interfaces\Persistable;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Histogram-based Gradient Boosted Decision Tree Classifier (LightGBM-style).
 *
 * All per-sample work (binning, histogram building, split finding, leaf updates,
 * inference) runs in C via Section 22 kernels. PHP only manages the O(T * 2^depth)
 * node-level tree-building loop — negligible overhead.
 */
final class GBDTClassifier implements Learner, Probabilistic, Persistable
{
    private ?Tensor $boundaries  = null;
    private ?Tensor $treeFeats   = null;
    private ?Tensor $treeThresh  = null;
    private ?Tensor $treeLefts   = null;
    private ?Tensor $treeRights  = null;
    private ?Tensor $treeSizes   = null;
    private float   $baseScore   = 0.0;
    private int     $maxNodes    = 0;

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
            throw new \InvalidArgumentException("GBDTClassifier requires labeled data.");
        }

        $X   = $dataset->samples();
        $N   = $X->shape()[0];
        $Q   = $this->numBins;

        $this->boundaries = Tensor::gbdtComputeBoundaries($X, $Q);
        $bins             = Tensor::gbdtBinSamples($X, $this->boundaries, $Q);

        // Base score: log-odds of positive class
        $posCount       = $y->sum();
        $p              = max(1e-7, min(1.0 - 1e-7, $posCount / $N));
        $this->baseScore = log($p / (1.0 - $p));

        $preds = Tensor::zeros($N)->addScalarInplace($this->baseScore);

        $maxNodes        = (1 << $this->maxDepth) * 2;
        $this->maxNodes  = $maxNodes;
        $T               = $this->nEstimators;

        // Pre-allocate packed tree storage [T, maxNodes]
        $featsArr   = array_fill(0, $T * $maxNodes, -1.0);
        $threshArr  = array_fill(0, $T * $maxNodes, 0.0);
        $leftsArr   = array_fill(0, $T * $maxNodes, -1.0);
        $rightsArr  = array_fill(0, $T * $maxNodes, -1.0);
        $sizesArr   = array_fill(0, $T, 0.0);

        for ($t = 0; $t < $T; $t++) {
            [$g, $h] = Tensor::gbdtLogLossGradHess($preds, $y);

            $rootMask = Tensor::ones($N);
            $nodeIdx  = 0;
            $offset   = $t * $maxNodes;

            // BFS queue: [mask, nodeIdx]
            $queue = [[$rootMask, 0, 0]]; // [mask, nodeId, depth]

            while (!empty($queue)) {
                [$mask, $nodeId, $depth] = array_shift($queue);

                $sumG  = $mask->mul($g)->sum();
                $sumH  = $mask->mul($h)->sum();
                $nodeN = (int)$mask->sum();

                if ($depth >= $this->maxDepth || $nodeN < (int)(2 * $this->minChildW)) {
                    // Leaf
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
                    // No profitable split → leaf
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

            // Update predictions with this tree's leaf values (walk tree in C)
            $treeF = Tensor::fromArray($featsArr);
            $treeT = Tensor::fromArray($threshArr);
            $treeL = Tensor::fromArray($leftsArr);
            $treeR = Tensor::fromArray($rightsArr);
            $treeS = Tensor::fromArray($sizesArr);

            // Reshape to [T, maxNodes] for predict_all
            $tF2 = $treeF->reshape($T, $maxNodes);
            $tT2 = $treeT->reshape($T, $maxNodes);
            $tL2 = $treeL->reshape($T, $maxNodes);
            $tR2 = $treeR->reshape($T, $maxNodes);

            // Use only trees 0..t: slice first t+1 rows
            $curF = $tF2->slice(0, 0, $t + 1);
            $curT = $tT2->slice(0, 0, $t + 1);
            $curL = $tL2->slice(0, 0, $t + 1);
            $curR = $tR2->slice(0, 0, $t + 1);
            $curS = $treeS->slice(0, 0, $t + 1);

            $preds = Tensor::gbdtPredictAll($bins, $curF, $curT, $curL, $curR, $curS, $this->baseScore)
                          ->mulScalarInplace($this->lr);
        }

        // Pack final trees
        $this->treeFeats  = Tensor::fromArray($featsArr)->reshape($T, $maxNodes);
        $this->treeThresh = Tensor::fromArray($threshArr)->reshape($T, $maxNodes);
        $this->treeLefts  = Tensor::fromArray($leftsArr)->reshape($T, $maxNodes);
        $this->treeRights = Tensor::fromArray($rightsArr)->reshape($T, $maxNodes);
        $this->treeSizes  = Tensor::fromArray($sizesArr);
    }

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("GBDTClassifier is not trained.");
        }
        $bins  = Tensor::gbdtBinSamples($dataset->samples(), $this->boundaries, $this->numBins);
        $raw   = Tensor::gbdtPredictAll(
            $bins,
            $this->treeFeats, $this->treeThresh,
            $this->treeLefts, $this->treeRights,
            $this->treeSizes, $this->baseScore
        )->mulScalarInplace($this->lr);
        // Sigmoid → [N, 2] columns [P(0), P(1)]
        $p1   = $raw->copy()->sigmoidInplace();                 // [N]
        // Stack as [N, 2]: [1-p, p]
        $ones = Tensor::ones($raw->shape()[0]);
        $p0   = $ones->sub($p1);
        return Tensor::concat([$p0->expandDims(1), $p1->expandDims(1)], 1);
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("GBDTClassifier is not trained.");
        }
        $bins = Tensor::gbdtBinSamples($dataset->samples(), $this->boundaries, $this->numBins);
        $raw  = Tensor::gbdtPredictAll(
            $bins,
            $this->treeFeats, $this->treeThresh,
            $this->treeLefts, $this->treeRights,
            $this->treeSizes, $this->baseScore
        )->mulScalarInplace($this->lr);
        // threshold at 0 (equivalent to sigmoid > 0.5)
        $zeros = Tensor::zeros($raw->shape()[0]);
        return $raw->greaterEqual($zeros);
    }

    public function trained(): bool
    {
        return $this->treeFeats !== null;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['nEstimators'=>$this->nEstimators,'maxDepth'=>$this->maxDepth,'numBins'=>$this->numBins,'lr'=>$this->lr,'lambda'=>$this->lambda,'gamma'=>$this->gamma,'minChildW'=>$this->minChildW,'baseScore'=>$this->baseScore,'maxNodes'=>$this->maxNodes]));
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
        $i->baseScore = (float)$c['baseScore']; $i->maxNodes = (int)$c['maxNodes'];
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
