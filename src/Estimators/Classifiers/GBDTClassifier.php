<?php
declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use Pml\Interfaces\Persistable;
use Pml\Interfaces\Scoring;
use Pml\Lib\SafeTensorsIO;
use Pml\Traits\GBDTCore;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Histogram-based Gradient Boosted Decision Tree Classifier (LightGBM-style).
 *
 * Supports binary classification (log-loss) and multiclass (softmax cross-entropy).
 *
 * Binary:     1 tree per round, sigmoid output, stored as [T, maxNodes].
 * Multiclass: K trees per round (one per class), softmax output,
 *             stored as [T*K, maxNodes] round-major (tk % K == class index).
 *
 * All per-sample work (binning, histograms, splits, leaf updates, inference)
 * runs in C via Section 22/26 kernels. PHP manages the O(T * K * 2^depth)
 * node-level loop — negligible overhead. Zero memory copies across FFI.
 */
final class GBDTClassifier implements Learner, Probabilistic, Persistable, Scoring
{
    use GBDTCore;

    private int     $maxNodes    = 0;
    private int     $nClasses    = 2;
    private array   $classLabels = [];
    private ?Tensor $baseScoresMC = null;

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

    // ── Training ──────────────────────────────────────────────────────────────

    public function train(Dataset $dataset, mixed ...$options): void
    {
        $y = $dataset->labels();
        if ($y === null) {
            throw new \InvalidArgumentException("GBDTClassifier requires labeled data.");
        }

        $X = $dataset->samples();
        $N = $X->shape()[0];
        $T = $this->nEstimators;

        $bins = $this->gbdtInitBins($X);

        // Detect unique classes (sorted)
        $flatY  = $y->toFlatArray();
        $unique = array_values(array_unique($flatY));
        sort($unique);
        $K = count($unique);

        $this->nClasses    = $K;
        $this->classLabels = $unique;

        $maxLeaves      = 1 << $this->maxDepth;
        $maxNodes       = $maxLeaves * 2;
        $this->maxNodes = $maxNodes;

        [$outFeats, $outThresh, $outLefts, $outRights] = $this->gbdtAllocateScratch($maxNodes);

        if ($K === 2) {
            $this->trainBinary($bins, $y, $N, $T, $maxLeaves, $maxNodes,
                               $outFeats, $outThresh, $outLefts, $outRights);
        } else {
            $this->trainMulticlass($bins, $y, $flatY, $N, $K, $T, $maxLeaves, $maxNodes,
                                   $outFeats, $outThresh, $outLefts, $outRights);
        }
    }

    private function trainBinary(
        Tensor $bins, Tensor $y, int $N, int $T,
        int $maxLeaves, int $maxNodes,
        Tensor $outFeats, Tensor $outThresh, Tensor $outLefts, Tensor $outRights
    ): void {
        $posCount        = $y->sum();
        $p               = max(1e-7, min(1.0 - 1e-7, $posCount / $N));
        $this->baseScore = log($p / (1.0 - $p));

        $preds    = Tensor::zeros($N)->addScalarInplace($this->baseScore);
        $sizesArr = array_fill(0, $T, 0.0);

        $this->gbdtAllocateForest($T, $maxNodes);

        for ($t = 0; $t < $T; $t++) {
            [$g, $h] = Tensor::gbdtLogLossGradHess($preds, $y);

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

    private function trainMulticlass(
        Tensor $bins, Tensor $y, array $flatY,
        int $N, int $K, int $T,
        int $maxLeaves, int $maxNodes,
        Tensor $outFeats, Tensor $outThresh, Tensor $outLefts, Tensor $outRights
    ): void {
        $labelToIdx = array_flip(array_map('strval', $this->classLabels));
        $encArr     = array_map(fn($v) => (float)$labelToIdx[strval($v)], $flatY);
        $yEnc       = Tensor::fromArray($encArr);

        $counts = array_fill(0, $K, 0);
        foreach ($encArr as $idx) { $counts[(int)$idx]++; }
        $baseArr = [];
        for ($k = 0; $k < $K; $k++) {
            $baseArr[$k] = log(max(1e-7, $counts[$k] / $N));
        }
        $this->baseScoresMC = Tensor::fromArray($baseArr);

        $preds = new Tensor([$N, $K]);
        Tensor::gbdtInitPredsMC($preds, $this->baseScoresMC);

        $g  = new Tensor([$N, $K]);
        $h  = new Tensor([$N, $K]);
        $TK = $T * $K;

        $this->gbdtAllocateForest($TK, $maxNodes);
        $sizesArr = array_fill(0, $TK, 0.0);

        for ($t = 0; $t < $T; $t++) {
            Tensor::gbdtSoftmaxGradHessInto($preds, $yEnc, $g, $h);

            for ($k = 0; $k < $K; $k++) {
                $outFeats->fill(-1.0);
                $outThresh->fill(0.0);
                $outLefts->fill(-1.0);
                $outRights->fill(-1.0);

                $nodesUsed = Tensor::gbdtTrainTreeMC(
                    $bins, $g, $h, $K, $k,
                    $this->numBins, $maxLeaves,
                    $this->lambda, $this->alpha, $this->gamma, $this->minChildW, $this->lr,
                    $preds, $outFeats, $outThresh, $outLefts, $outRights
                );
                $tk = $t * $K + $k;
                $sizesArr[$tk] = (float) $nodesUsed;

                $this->gbdtStoreTree($tk, $maxNodes, $outFeats, $outThresh, $outLefts, $outRights);
            }
        }

        $this->gbdtReshapeForest($TK, $maxNodes, $sizesArr);
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("GBDTClassifier is not trained.");
        }
        $bins = Tensor::gbdtBinSamples($dataset->samples(), $this->boundaries, $this->numBins);

        return $this->nClasses === 2
            ? $this->probaBinary($bins)
            : $this->probaMulticlass($bins);
    }

    private function probaBinary(Tensor $bins): Tensor
    {
        $raw  = Tensor::gbdtPredictAll(
            $bins, $this->treeFeats, $this->treeThresh,
            $this->treeLefts, $this->treeRights, $this->treeSizes, $this->baseScore
        );
        $p1   = $raw->copy()->sigmoidInplace();
        $ones = Tensor::ones($raw->shape()[0]);
        $p0   = $ones->sub($p1);
        return Tensor::concat([$p0->expandDims(1), $p1->expandDims(1)], 1);
    }

    private function probaMulticlass(Tensor $bins): Tensor
    {
        $raw = Tensor::gbdtPredictAllMC(
            $bins, $this->treeFeats, $this->treeThresh,
            $this->treeLefts, $this->treeRights, $this->treeSizes,
            $this->baseScoresMC, $this->nClasses
        );
        $raw->rowSoftmaxInplace();
        return $raw;
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("GBDTClassifier is not trained.");
        }
        $bins = Tensor::gbdtBinSamples($dataset->samples(), $this->boundaries, $this->numBins);

        return $this->nClasses === 2
            ? $this->predictBinary($bins)
            : $this->predictMulticlass($bins);
    }

    private function predictBinary(Tensor $bins): Tensor
    {
        $raw   = Tensor::gbdtPredictAll(
            $bins, $this->treeFeats, $this->treeThresh,
            $this->treeLefts, $this->treeRights, $this->treeSizes, $this->baseScore
        );
        $zeros = Tensor::zeros($raw->shape()[0]);
        return $raw->greaterEqual($zeros);
    }

    private function predictMulticlass(Tensor $bins): Tensor
    {
        $raw = Tensor::gbdtPredictAllMC(
            $bins, $this->treeFeats, $this->treeThresh,
            $this->treeLefts, $this->treeRights, $this->treeSizes,
            $this->baseScoresMC, $this->nClasses
        );
        return $raw->argmaxAxis(1);
    }

    /** Accuracy — proportion of correctly predicted labels (computed entirely in C). */
    public function score(Dataset $dataset): float
    {
        $pred   = $this->predict($dataset);
        $labels = $dataset->labels();
        return $pred->equal($labels)->mean();
    }

    /** Return sorted class label values (index k → original label). */
    public function classes(): array
    {
        return $this->classLabels;
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode([
            'nEstimators' => $this->nEstimators, 'maxDepth'    => $this->maxDepth,
            'numBins'     => $this->numBins,      'lr'          => $this->lr,
            'lambda'      => $this->lambda,        'alpha'       => $this->alpha,
            'gamma'       => $this->gamma,         'minChildW'   => $this->minChildW,
            'maxNodes'    => $this->maxNodes,      'nClasses'    => $this->nClasses,
            'classLabels' => $this->classLabels,   'baseScore'   => $this->baseScore,
        ]));

        if ($this->treeFeats !== null) {
            $tensors = [
                'boundaries'  => $this->boundaries,
                'tree_feats'  => $this->treeFeats,
                'tree_thresh' => $this->treeThresh,
                'tree_lefts'  => $this->treeLefts,
                'tree_rights' => $this->treeRights,
                'tree_sizes'  => $this->treeSizes,
            ];
            if ($this->baseScoresMC !== null) {
                $tensors['base_scores_mc'] = $this->baseScoresMC;
            }
            SafeTensorsIO::save($dir . '/model.safetensors', $tensors);
        }
    }

    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self(
            (int)$c['nEstimators'],  (int)$c['maxDepth'],   (int)$c['numBins'],
            (float)$c['lr'],         (float)$c['lambda'],    (float)($c['alpha'] ?? 0.0),
            (float)$c['gamma'],      (float)$c['minChildW']
        );
        $i->maxNodes    = (int)($c['maxNodes']    ?? 0);
        $i->nClasses    = (int)($c['nClasses']    ?? 2);
        $i->classLabels = (array)($c['classLabels'] ?? []);
        $i->baseScore   = (float)($c['baseScore']  ?? 0.0);

        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) {
            $t = SafeTensorsIO::load($stPath);
            $i->boundaries   = $t['boundaries']     ?? null;
            $i->treeFeats    = $t['tree_feats']     ?? null;
            $i->treeThresh   = $t['tree_thresh']    ?? null;
            $i->treeLefts    = $t['tree_lefts']     ?? null;
            $i->treeRights   = $t['tree_rights']    ?? null;
            $i->treeSizes    = $t['tree_sizes']     ?? null;
            $i->baseScoresMC = $t['base_scores_mc'] ?? null;
        }
        return $i;
    }
}
