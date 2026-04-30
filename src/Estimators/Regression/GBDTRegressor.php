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
        private readonly float $alpha       = 0.0,
        private readonly float $gamma       = 0.0,
        private readonly float $minChildW   = 1.0
    ) {}

    public function train(Dataset $dataset): void
    {
        $y = $dataset->labels();
        if ($y === null) {
            throw new \InvalidArgumentException("GBDTRegressor requires labeled data.");
        }

        $X  = $dataset->samples();
        $N  = $X->shape()[0];
        $Q  = $this->numBins;
        $T  = $this->nEstimators;

        $this->boundaries = Tensor::gbdtComputeBoundaries($X, $Q);
        $bins             = Tensor::gbdtBinSamples($X, $this->boundaries, $Q);

        $this->baseScore = $y->sum() / $N;
        $preds           = Tensor::zeros($N)->addScalarInplace($this->baseScore);

        $maxLeaves = 1 << $this->maxDepth;
        $maxNodes  = $maxLeaves * 2;

        // Per-tree scratch tensors (reused each iteration)
        $outFeats  = Tensor::zeros($maxNodes);
        $outThresh = Tensor::zeros($maxNodes);
        $outLefts  = Tensor::zeros($maxNodes);
        $outRights = Tensor::zeros($maxNodes);

        // Flat pre-allocated storage for all trees — no PHP arrays, no fromArray() at end
        $this->treeFeats  = Tensor::zeros($T * $maxNodes)->fill(-1.0);
        $this->treeThresh = Tensor::zeros($T * $maxNodes);
        $this->treeLefts  = Tensor::zeros($T * $maxNodes)->fill(-1.0);
        $this->treeRights = Tensor::zeros($T * $maxNodes)->fill(-1.0);
        $sizesArr         = array_fill(0, $T, 0.0);

        for ($t = 0; $t < $T; $t++) {
            [$g, $h] = Tensor::gbdtMseGradHess($preds, $y);

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

            // Single C memcpy per array instead of maxNodes PHP float reads
            Tensor::gbdtCollectTree($this->treeFeats,  $t, $maxNodes, $outFeats);
            Tensor::gbdtCollectTree($this->treeThresh, $t, $maxNodes, $outThresh);
            Tensor::gbdtCollectTree($this->treeLefts,  $t, $maxNodes, $outLefts);
            Tensor::gbdtCollectTree($this->treeRights, $t, $maxNodes, $outRights);
            unset($g, $h);
        }

        $this->treeFeats  = $this->treeFeats->reshape($T, $maxNodes);
        $this->treeThresh = $this->treeThresh->reshape($T, $maxNodes);
        $this->treeLefts  = $this->treeLefts->reshape($T, $maxNodes);
        $this->treeRights = $this->treeRights->reshape($T, $maxNodes);
        $this->treeSizes  = Tensor::fromArray($sizesArr);
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("GBDTRegressor is not trained.");
        }
        $bins = Tensor::gbdtBinSamples($dataset->samples(), $this->boundaries, $this->numBins);
        // lr already baked into stored leaf values by tensor_gbdt_train_tree
        return Tensor::gbdtPredictAll(
            $bins,
            $this->treeFeats, $this->treeThresh,
            $this->treeLefts, $this->treeRights,
            $this->treeSizes, $this->baseScore
        );
    }

    public function trained(): bool
    {
        return $this->treeFeats !== null;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['nEstimators'=>$this->nEstimators,'maxDepth'=>$this->maxDepth,'numBins'=>$this->numBins,'lr'=>$this->lr,'lambda'=>$this->lambda,'alpha'=>$this->alpha,'gamma'=>$this->gamma,'minChildW'=>$this->minChildW,'baseScore'=>$this->baseScore]));
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
