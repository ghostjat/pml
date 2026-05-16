<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Estimators\Classifiers\DecisionTreeClassifier;
use RuntimeException;

/**
 * AdaBoost (Adaptive Boosting) Classifier.
 * Sequentially trains "Decision Stumps" (Depth 1), assigning higher sample weights to misclassified points.
 * * JIT & Memory Optimized:
 * - Executes SAMME algorithm updates natively via OpenBLAS broadcasting.
 * - Extracts `weightedSample` subsets via an O(log N) binary search routing to `tensor_take()`.
 */
final class AdaBoostClassifier implements Learner, Persistable
{
    private int $nEstimators;
    private float $learningRate;

    /** @var DecisionTreeClassifier[] */
    private array $estimators = [];
    /** @var float[] */
    private array $alphas = [];

    private array $classes = [];

    // C-resident lookup tensors rebuilt after train() / load()
    private ?Tensor $classIndexTable = null;   // [maxLabel+1]: label → class index
    private ?Tensor $classLabelTable  = null;   // [K]:          class index → label

    public function __construct(int $nEstimators = 50, float $learningRate = 1.0)
    {
        $this->nEstimators = $nEstimators;
        $this->learningRate = $learningRate;
    }

    public function train(Dataset $dataset, mixed ...$options): void
    {
        $n = $dataset->numRows();
        $y = $dataset->labels();

        $this->classes = $y->unique()->sort(0)->toFlatArray();
        $k = count($this->classes);

        if ($k < 2) {
            throw new \InvalidArgumentException("AdaBoost requires at least 2 distinct classes.");
        }

        $this->buildLookupTables();

        $w = Tensor::ones($n)->mulScalarInplace(1.0 / $n);

        for ($i = 0; $i < $this->nEstimators; $i++) {

            // 1. Resample dataset proportionally to current weights
            $wFlat   = $w->toFlatArray();
            $idxT    = Tensor::fromArray($this->weightedSample($wFlat, $n));
            $subDataset = new Dataset(
                $dataset->samples()->take($idxT, 0),
                $y->take($idxT, 0)
            );
            unset($idxT);

            // 2. Train a Decision Stump (depth-1 CART)
            $stump = new DecisionTreeClassifier(maxDepth: 1);
            $stump->train($subDataset);
            unset($subDataset);

            // 3. Weighted error on full dataset
            $preds         = $stump->predict($dataset);
            $incorrectMask = $preds->notEqual($y);
            $error         = $w->mul($incorrectMask)->sum();

            if ($error <= 0.0) {
                $this->estimators[] = $stump;
                $this->alphas[]     = 1.0;
                break;
            }

            if ($error >= 1.0 - (1.0 / $k)) {
                break;
            }

            // 4. SAMME estimator weight
            $alpha = $this->learningRate * (log((1.0 - $error) / $error) + log($k - 1));

            $this->estimators[] = $stump;
            $this->alphas[]     = $alpha;

            // 5. Update sample weights in C: w *= exp(alpha * incorrect)
            $w->mulInplace($incorrectMask->mulScalar($alpha)->exp());
            $w->mulScalarInplace(1.0 / $w->sum());
            unset($preds, $incorrectMask);
        }
    }

    /** O(N log N) weighted sampling via cumulative-sum binary search. */
    private function weightedSample(array $weights, int $n): array
    {
        $cumsum = [];
        $total  = 0.0;
        foreach ($weights as $w) {
            $total   += $w;
            $cumsum[] = $total;
        }

        $indices = [];
        $maxIdx  = count($cumsum) - 1;

        for ($i = 0; $i < $n; $i++) {
            $r    = (mt_rand() / mt_getrandmax()) * $total;
            $low  = 0;
            $high = $maxIdx;
            while ($low < $high) {
                $mid = ($low + $high) >> 1;
                if ($r > $cumsum[$mid]) $low  = $mid + 1;
                else                    $high = $mid;
            }
            $indices[] = $low;
        }

        return $indices;
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("AdaBoost is not trained.");
        }

        $k      = count($this->classes);
        $n      = $dataset->numRows();
        $scores = Tensor::zeros($n, $k);   // [N, K] alpha-weighted score accumulator

        // Per estimator: one C call for predictions, one for one-hot, one for weighted add.
        // Inner N-loop is gone — all work is vectorized in C.
        foreach ($this->estimators as $i => $estimator) {
            $alpha  = $this->alphas[$i];
            $preds  = $estimator->predict($dataset);                             // [N]
            $idx    = Tensor::gatherIndices($preds, $this->classIndexTable);     // [N] → class indices
            $oneHot = Tensor::onehot($idx, $k);                                  // [N, K]
            $scores->addInplace($oneHot->mulScalarInplace($alpha));
            unset($preds, $idx, $oneHot);
        }

        // argmax index per row → gather original class label
        return Tensor::gatherIndices($scores->argmaxAxis(1), $this->classLabelTable);
    }

    public function trained(): bool
    {
        return !empty($this->estimators);
    }

    /** Build C-resident label↔index lookup tensors from $this->classes. */
    private function buildLookupTables(): void
    {
        $this->classLabelTable = Tensor::fromArray(array_map('floatval', $this->classes));  // [K]
        $maxLabel = (int) max(array_map('intval', $this->classes));
        $table    = array_fill(0, $maxLabel + 1, 0.0);
        foreach ($this->classes as $idx => $label) {
            $table[(int) $label] = (float) $idx;
        }
        $this->classIndexTable = Tensor::fromArray($table);
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode([
            'nEstimators'  => $this->nEstimators,
            'learningRate' => $this->learningRate,
            'alphas'       => $this->alphas,
            'classes'      => $this->classes,
        ]));
        $treeData = [];
        foreach ($this->estimators as $tree) { $treeData[] = $tree->exportPhpTree(); }
        file_put_contents($dir . '/trees.json', json_encode($treeData));
    }

    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((int) $c['nEstimators'], (float) $c['learningRate']);
        $i->alphas  = $c['alphas']  ?? [];
        $i->classes = $c['classes'] ?? [];
        $i->buildLookupTables();
        foreach (json_decode(file_get_contents($dir . '/trees.json'), true) as $treeData) {
            $i->estimators[] = DecisionTreeClassifier::fromPhpTree($treeData);
        }
        return $i;
    }
}