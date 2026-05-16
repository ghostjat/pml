<?php
declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Interfaces\Scoring;
use Pml\Lib\SafeTensorsIO;
use Pml\Traits\GBDTCore;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Histogram-based Gradient Boosted Decision Tree Regressor (LightGBM-style).
 *
 * Uses MSE loss. All per-sample work runs in C. PHP manages the O(T * 2^depth)
 * BFS node loop only.
 */
final class GBDTRegressor implements Learner, Persistable, Scoring
{
    use GBDTCore;

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

    public function train(Dataset $dataset, mixed ...$options): void
    {
        $y = $dataset->labels();
        if ($y === null) {
            throw new \InvalidArgumentException("GBDTRegressor requires labeled data.");
        }

        $X  = $dataset->samples();
        $N  = $X->shape()[0];
        $T  = $this->nEstimators;

        $bins = $this->gbdtInitBins($X);

        $this->baseScore = $y->sum() / $N;
        $preds           = Tensor::zeros($N)->addScalarInplace($this->baseScore);

        $maxLeaves = 1 << $this->maxDepth;
        $maxNodes  = $maxLeaves * 2;

        [$outFeats, $outThresh, $outLefts, $outRights] = $this->gbdtAllocateScratch($maxNodes);
        $this->gbdtAllocateForest($T, $maxNodes);
        $sizesArr = array_fill(0, $T, 0.0);

        for ($t = 0; $t < $T; $t++) {
            [$g, $h] = Tensor::gbdtMseGradHess($preds, $y);

            $outFeats->fill(-1.0);
            $outThresh->fill(0.0);
            $outLefts->fill(-1.0);
            $outRights->fill(-1.0);

            $nodesUsed = Tensor::gbdtTrainTree(
                $bins, $g, $h, $this->numBins, $maxLeaves,
                $this->lambda, $this->alpha, $this->gamma, $this->minChildW, $this->lr,
                $preds, $outFeats, $outThresh, $outLefts, $outRights
            );
            $sizesArr[$t] = (float) $nodesUsed;

            $this->gbdtStoreTree($t, $maxNodes, $outFeats, $outThresh, $outLefts, $outRights);
            unset($g, $h);
        }

        $this->gbdtReshapeForest($T, $maxNodes, $sizesArr);
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("GBDTRegressor is not trained.");
        }
        return $this->gbdtRunForest($dataset);
    }

    /**
     * R² coefficient of determination — 1 - SS_res/SS_tot.
     * All arithmetic delegated to C via Tensor ops; one PHP float extracted at the end.
     */
    public function score(Dataset $dataset): float
    {
        $pred   = $this->predict($dataset);
        $labels = $dataset->labels();
        $res    = $pred->sub($labels);
        $ssRes  = $res->sumSquares();
        unset($res);
        $ssTot  = $labels->variance() * $labels->size();
        return $ssTot > 0.0 ? 1.0 - $ssRes / $ssTot : 1.0;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode([
            'nEstimators' => $this->nEstimators, 'maxDepth'  => $this->maxDepth,
            'numBins'     => $this->numBins,      'lr'        => $this->lr,
            'lambda'      => $this->lambda,        'alpha'     => $this->alpha,
            'gamma'       => $this->gamma,         'minChildW' => $this->minChildW,
            'baseScore'   => $this->baseScore,
        ]));
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
        $i = new self(
            (int)$c['nEstimators'], (int)$c['maxDepth'],   (int)$c['numBins'],
            (float)$c['lr'],        (float)$c['lambda'],    (float)($c['alpha'] ?? 0.0),
            (float)$c['gamma'],     (float)$c['minChildW']
        );
        $i->baseScore = (float) $c['baseScore'];
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
