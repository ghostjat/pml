<?php
declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Estimators\Classifiers\DecisionTreeClassifier;
use RuntimeException;

/**
 * LogitBoost — gradient boosting with logistic (log-loss) base learners.
 * Each iteration fits a weak Decision Tree to the pseudo-residuals of the logistic loss.
 *
 * JIT & Memory Optimized:
 * - Pseudo-residual computation is fully in-place C arithmetic.
 * - Tree predictions are collected as Tensors; summation uses in-place addInplace.
 * - Only one toFlatArray() at the very end to materialise integer predictions.
 */
final class LogitBoost implements Learner, Probabilistic, Persistable
{
    /** @var DecisionTreeClassifier[] */
    private array $trees          = [];
    /** @var float[] per-tree shrinkage multipliers */
    private array $learningRates  = [];

    public function __construct(
        private readonly int   $estimators   = 100,
        private readonly float $learningRate = 0.1,
        private readonly int   $maxDepth     = 3,
        private readonly int   $minSamples   = 2
    ) {}

    public function train(Dataset $dataset): void
    {
        $labels = $dataset->labels();
        if ($labels === null) {
            throw new \InvalidArgumentException("LogitBoost requires labeled data.");
        }

        $n = $dataset->numRows();
        // Convert {0,1} labels to {-1,+1} in C
        $y = $labels->mulScalar(2.0)->addScalarInplace(-1.0);          // [N]

        // Initial log-odds: F_0 = 0.5 * log( p/(1-p) ), start at 0
        $F = Tensor::zeros($n);                                         // [N] raw scores

        for ($t = 0; $t < $this->estimators; $t++) {
            // 1. Compute pseudo-residuals: r = y * sigmoid(-y*F)
            $yF    = $y->mul($F)->mulScalarInplace(-1.0);               // [N]
            $r     = $y->mul($yF->sigmoid());                           // [N] — stays in C

            // 2. Fit shallow tree on (X, r) — residuals as regression target
            $residualDataset = new Dataset($dataset->samples(), $r);
            $tree = new DecisionTreeClassifier($this->maxDepth, $this->minSamples);
            $tree->train($residualDataset);

            // 3. Predict leaf values, add to F
            $delta = $tree->predict($residualDataset);                  // [N] integer predictions
            $F->addInplace($delta->mulScalar($this->learningRate));

            $this->trees[]         = $tree;
            $this->learningRates[] = $this->learningRate;
        }
    }

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("LogitBoost is not trained.");
        }

        $n = $dataset->numRows();
        $F = Tensor::zeros($n);

        foreach ($this->trees as $i => $tree) {
            $delta = $tree->predict($dataset);
            $F->addInplace($delta->mulScalar($this->learningRates[$i]));
        }

        return $F->sigmoid();   // P(y=1|x)
    }

    public function predict(Dataset $dataset): Tensor
    {
        return $this->proba($dataset)->round();
    }

    public function trained(): bool
    {
        return !empty($this->trees);
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['estimators' => $this->estimators, 'learningRate' => $this->learningRate, 'maxDepth' => $this->maxDepth, 'minSamples' => $this->minSamples, 'learningRates' => $this->learningRates]));
        $treeData = [];
        foreach ($this->trees as $tree) { $treeData[] = $tree->exportPhpTree(); }
        file_put_contents($dir . '/trees.json', json_encode($treeData));
    }

    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((int) $c['estimators'], (float) $c['learningRate'], (int) $c['maxDepth'], (int) $c['minSamples']);
        $i->learningRates = $c['learningRates'] ?? [];
        foreach (json_decode(file_get_contents($dir . '/trees.json'), true) as $treeData) {
            $i->trees[] = DecisionTreeClassifier::fromPhpTree($treeData);
        }
        return $i;
    }
}
