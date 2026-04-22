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
        private readonly float $alpha       = 0.0,
        private readonly float $gamma       = 0.0,
        private readonly float $minChildW   = 1.0
    ) {}

    public function train(Dataset $dataset): void
    {
        $y = $dataset->labels();
        if ($y === null) {
            throw new \InvalidArgumentException("GBDTClassifier requires labeled data.");
        }

        $X  = $dataset->samples();
        $N  = $X->shape()[0];
        $Q  = $this->numBins;
        $T  = $this->nEstimators;

        $this->boundaries = Tensor::gbdtComputeBoundaries($X, $Q);
        $bins             = Tensor::gbdtBinSamples($X, $this->boundaries, $Q);

        // Base score: log-odds of positive class
        $posCount        = $y->sum();
        $p               = max(1e-7, min(1.0 - 1e-7, $posCount / $N));
        $this->baseScore = log($p / (1.0 - $p));

        // preds updated in-place by each C tree call
        $preds = Tensor::zeros($N)->addScalarInplace($this->baseScore);

        $maxLeaves       = 1 << $this->maxDepth;
        $maxNodes        = $maxLeaves * 2;
        $this->maxNodes  = $maxNodes;

        // Reusable per-tree output tensors (reset before each call)
        $outFeats  = Tensor::zeros($maxNodes);
        $outThresh = Tensor::zeros($maxNodes);
        $outLefts  = Tensor::zeros($maxNodes);
        $outRights = Tensor::zeros($maxNodes);

        $featsArr  = array_fill(0, $T * $maxNodes, -1.0);
        $threshArr = array_fill(0, $T * $maxNodes,  0.0);
        $leftsArr  = array_fill(0, $T * $maxNodes, -1.0);
        $rightsArr = array_fill(0, $T * $maxNodes, -1.0);
        $sizesArr  = array_fill(0, $T, 0.0);

        for ($t = 0; $t < $T; $t++) {
            [$g, $h] = Tensor::gbdtLogLossGradHess($preds, $y);

            $outFeats->fill(-1.0);
            $outThresh->fill(0.0);
            $outLefts->fill(-1.0);
            $outRights->fill(-1.0);

            $nodesUsed = Tensor::gbdtTrainTree(
                $bins, $g, $h, $Q, $maxLeaves,
                $this->lambda, $this->alpha, $this->gamma, $this->minChildW, $this->lr,
                $preds, $outFeats, $outThresh, $outLefts, $outRights
            );
            $sizesArr[$t] = (float)$nodesUsed;

            $offset = $t * $maxNodes;
            $fBuf   = $outFeats->buffer();
            $tBuf   = $outThresh->buffer();
            $lBuf   = $outLefts->buffer();
            $rBuf   = $outRights->buffer();
            for ($i = 0; $i < $maxNodes; $i++) {
                $featsArr[$offset + $i]  = $fBuf[$i];
                $threshArr[$offset + $i] = $tBuf[$i];
                $leftsArr[$offset + $i]  = $lBuf[$i];
                $rightsArr[$offset + $i] = $rBuf[$i];
            }
            unset($g, $h);
        }

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
        // lr is already baked into the stored leaf values — no additional scaling needed.
        $raw   = Tensor::gbdtPredictAll(
            $bins,
            $this->treeFeats, $this->treeThresh,
            $this->treeLefts, $this->treeRights,
            $this->treeSizes, $this->baseScore
        );
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
        );
        // lr already baked into stored leaf values; threshold raw log-odds at 0
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
        file_put_contents($dir . '/config.json', json_encode(['nEstimators'=>$this->nEstimators,'maxDepth'=>$this->maxDepth,'numBins'=>$this->numBins,'lr'=>$this->lr,'lambda'=>$this->lambda,'alpha'=>$this->alpha,'gamma'=>$this->gamma,'minChildW'=>$this->minChildW,'baseScore'=>$this->baseScore,'maxNodes'=>$this->maxNodes]));
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
        $i = new self((int)$c['nEstimators'],(int)$c['maxDepth'],(int)$c['numBins'],(float)$c['lr'],(float)$c['lambda'],(float)($c['alpha']??0.0),(float)$c['gamma'],(float)$c['minChildW']);
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
