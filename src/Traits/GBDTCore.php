<?php

declare(strict_types=1);

namespace Pml\Traits;

use Pml\Tensor;
use Pml\Dataset;

/**
 * Shared histogram-GBDT state and helper methods used by GBDTClassifier and GBDTRegressor.
 * Users of this trait must declare constructor properties:
 *   $nEstimators, $maxDepth, $numBins, $lr, $lambda, $alpha, $gamma, $minChildW
 */
trait GBDTCore
{
    protected ?Tensor $boundaries  = null;
    protected ?Tensor $treeFeats   = null;
    protected ?Tensor $treeThresh  = null;
    protected ?Tensor $treeLefts   = null;
    protected ?Tensor $treeRights  = null;
    protected ?Tensor $treeSizes   = null;
    protected float   $baseScore   = 0.0;

    public function trained(): bool
    {
        return $this->treeFeats !== null;
    }

    /** Compute bin boundaries and bin X — returns binned samples Tensor. */
    protected function gbdtInitBins(Tensor $X): Tensor
    {
        $this->boundaries = Tensor::gbdtComputeBoundaries($X, $this->numBins);
        return Tensor::gbdtBinSamples($X, $this->boundaries, $this->numBins);
    }

    /** Allocate per-tree scratch tensors (reused each boosting round). */
    protected function gbdtAllocateScratch(int $maxNodes): array
    {
        return [
            Tensor::zeros($maxNodes),
            Tensor::zeros($maxNodes),
            Tensor::zeros($maxNodes),
            Tensor::zeros($maxNodes),
        ];
    }

    /** Allocate flat forest storage: T*maxNodes cells, pre-filled with sentinels. */
    protected function gbdtAllocateForest(int $T, int $maxNodes): void
    {
        $this->treeFeats  = Tensor::zeros($T * $maxNodes)->fill(-1.0);
        $this->treeThresh = Tensor::zeros($T * $maxNodes);
        $this->treeLefts  = Tensor::zeros($T * $maxNodes)->fill(-1.0);
        $this->treeRights = Tensor::zeros($T * $maxNodes)->fill(-1.0);
    }

    /** Copy one tree's scratch buffers into the flat forest storage at slot $t. */
    protected function gbdtStoreTree(
        int $t, int $maxNodes,
        Tensor $outFeats, Tensor $outThresh, Tensor $outLefts, Tensor $outRights
    ): void {
        Tensor::gbdtCollectTree($this->treeFeats,  $t, $maxNodes, $outFeats);
        Tensor::gbdtCollectTree($this->treeThresh, $t, $maxNodes, $outThresh);
        Tensor::gbdtCollectTree($this->treeLefts,  $t, $maxNodes, $outLefts);
        Tensor::gbdtCollectTree($this->treeRights, $t, $maxNodes, $outRights);
    }

    /** Reshape flat forest buffers to [T, maxNodes] and materialise treeSizes. */
    protected function gbdtReshapeForest(int $T, int $maxNodes, array $sizesArr): void
    {
        $this->treeFeats  = $this->treeFeats->reshape($T, $maxNodes);
        $this->treeThresh = $this->treeThresh->reshape($T, $maxNodes);
        $this->treeLefts  = $this->treeLefts->reshape($T, $maxNodes);
        $this->treeRights = $this->treeRights->reshape($T, $maxNodes);
        $this->treeSizes  = Tensor::fromArray($sizesArr);
    }

    /** Run the stored forest on a dataset's binned samples + base score. */
    protected function gbdtRunForest(Dataset $dataset): Tensor
    {
        $bins = Tensor::gbdtBinSamples($dataset->samples(), $this->boundaries, $this->numBins);
        return Tensor::gbdtPredictAll(
            $bins,
            $this->treeFeats, $this->treeThresh,
            $this->treeLefts, $this->treeRights,
            $this->treeSizes, $this->baseScore
        );
    }
}
