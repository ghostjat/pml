<?php
declare(strict_types=1);

namespace Pml\Ensemble;

use Pml\CrossValidation\KFoldOOF;
use Pml\Dataset;
use Pml\Estimators\Regression\Ridge;
use Pml\Interfaces\Learner;
use Pml\Tensor;

/**
 * Two-level stacking regressor.
 *
 * Level 1: base learners trained via K-fold OOF to prevent leakage.
 * Level 2: Ridge meta-learner trained on the OOF predictions matrix.
 *
 * At inference time, base learners (retrained on full training set) generate
 * predictions that are fed as features to the meta-learner.
 *
 * Memory layout: OOF matrix is [N × M] where M = number of base learners.
 */
final class StackingRegressor
{
    /** @var Learner[]  Base learners retrained on full training data */
    private array $trainedBase = [];

    /** Meta-learner (Level 2) */
    private ?Ridge $meta = null;

    /**
     * @param array<callable> $baseFactories  Each fn(): Learner produces a fresh base model
     * @param int             $k              K-fold splits for OOF generation
     * @param float           $metaAlpha      Ridge alpha for meta-learner
     */
    public function __construct(
        private readonly array $baseFactories,
        private readonly int   $k           = 5,
        private readonly float $metaAlpha   = 1.0,
        private readonly int   $metaEpochs  = 1000,
        private readonly float $metaLr      = 0.005,
        private readonly int   $metaBatch   = 256,
    ) {}

    /**
     * Phase 1: generate OOF predictions for all base learners.
     * Phase 2: train meta-learner on OOF matrix.
     * Phase 3: retrain base learners on full training set.
     *
     * @return array{oof_rmse: float[][], meta_oof_rmse: float}
     */
    public function train(Dataset $dataset, ?callable $logger = null): array
    {
        $kfold   = new KFoldOOF($this->k);
        $oofCols = [];
        $oofRmseAll = [];

        // ── Phase 1: OOF predictions per base learner ─────────────────────────
        foreach ($this->baseFactories as $idx => $factory) {
            if ($logger) $logger("  base[$idx] OOF {$this->k}-fold...");
            [$oofPreds, $foldRmse] = $kfold->run($factory, $dataset);
            $oofCols[]     = $oofPreds;         // [N]
            $oofRmseAll[]  = $foldRmse;
            if ($logger) {
                $mean = array_sum($foldRmse) / count($foldRmse);
                $logger(sprintf('    base[%d] mean OOF RMSE (log): %.4f', $idx, $mean));
            }
        }

        // ── Phase 2: stack OOF columns → [N × M] meta-features ───────────────
        // Each $oofCols[i] is [N]; expand to [N,1] then concat on axis=1 → [N,M]
        $expanded  = array_map(fn(Tensor $t) => $t->expandDims(1), $oofCols);
        $oofMatrix = Tensor::concat($expanded, 1);  // [N × M]
        $metaDs    = new Dataset($oofMatrix, $dataset->labels());
        $metaDs->randomize();

        $this->meta = new Ridge(
            $this->metaAlpha,
            $this->metaEpochs,
            $this->metaLr,
            $this->metaBatch,
        );
        $this->meta->train($metaDs);

        // Meta OOF RMSE (train-set re-prediction — optimistic but useful as sanity check)
        $metaPreds   = $this->meta->predict($metaDs);
        $diff        = $metaPreds->sub($dataset->labels());
        $metaOofRmse = sqrt($diff->mul($diff)->mean());
        if ($logger) $logger(sprintf('  meta train RMSE (log): %.4f', $metaOofRmse));

        // ── Phase 3: retrain base learners on full training set ───────────────
        $this->trainedBase = [];
        foreach ($this->baseFactories as $idx => $factory) {
            if ($logger) $logger("  base[$idx] retraining on full train...");
            $model = $factory();
            $model->train($dataset);
            $this->trainedBase[] = $model;
        }

        return ['oof_rmse' => $oofRmseAll, 'meta_oof_rmse' => $metaOofRmse];
    }

    public function predict(Dataset $dataset): Tensor
    {
        if ($this->meta === null) {
            throw new \RuntimeException('StackingRegressor not trained yet.');
        }

        $cols = [];
        foreach ($this->trainedBase as $model) {
            $cols[] = $model->predict($dataset)->expandDims(1);  // [N,1]
        }

        $meta = new Dataset(Tensor::concat($cols, 1));  // [N × M]
        return $this->meta->predict($meta);
    }

    public function trained(): bool { return $this->meta !== null; }
}
