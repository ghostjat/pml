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
 * Random Forest Ensemble.
 * Trains multiple Decision Trees on bootstrapped datasets to prevent overfitting.
 * * JIT & Memory Optimized:
 * - Bootstrapping implemented via fast PHP-index arrays driving C-level tensor_take().
 * - Inference voting executed entirely in PHP JIT cache.
 */
final class RandomForestClassifier implements Learner, Persistable
{
    private int $nEstimators;
    private int $maxDepth;
    private int $minSamplesSplit;
    
    /** @var DecisionTreeClassifier[] */
    private array $trees = [];

    public function __construct(int $nEstimators = 100, int $maxDepth = 10, int $minSamplesSplit = 2)
    {
        $this->nEstimators = $nEstimators;
        $this->maxDepth = $maxDepth;
        $this->minSamplesSplit = $minSamplesSplit;
    }

    public function train(Dataset $dataset): void
    {
        $n = $dataset->numRows();
        $features = $dataset->numColumns();
        
        // Feature bagging: each tree only sees a random square root subset of features
        $maxFeatures = (int) max(1, sqrt($features));

        for ($i = 0; $i < $this->nEstimators; $i++) {
            
            // 1. Bootstrap Sampling (Random selection with replacement)
            $indices = [];
            for ($j = 0; $j < $n; $j++) {
                $indices[] = mt_rand(0, $n - 1);
            }
            
            // 2. Extract Bootstrap slice safely in C
            $idxT = Tensor::fromArray($indices);
            $bootX = $dataset->samples()->take($idxT, 0);
            $bootY = $dataset->labels()->take($idxT, 0);
            
            $bootDataset = new Dataset($bootX, $bootY);

            // 3. Train the sub-tree
            $tree = new DecisionTreeClassifier($this->maxDepth, $this->minSamplesSplit, $maxFeatures);
            $tree->train($bootDataset);
            
            $this->trees[] = $tree;
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Random Forest is not trained.");
        }

        $n = $dataset->numRows();
        $treePreds = [];
        
        // Gather all predictions
        foreach ($this->trees as $tree) {
            $treePreds[] = $tree->predict($dataset)->toFlatArray();
        }

        $finalPreds = [];
        
        // JIT Optimized Voting Process
        for ($i = 0; $i < $n; $i++) {
            $votes = [];
            foreach ($treePreds as $preds) {
                $v = (int) $preds[$i];
                $votes[$v] = ($votes[$v] ?? 0) + 1;
            }
            
            // Sort votes descending and pick the highest
            arsort($votes);
            $finalPreds[] = array_key_first($votes);
        }

        return Tensor::fromArray($finalPreds);
    }

    public function trained(): bool
    {
        return !empty($this->trees);
    }

    public function save(string $dir): void
    {
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }

        $treeData = array_map(
            static fn(DecisionTreeClassifier $t) => $t->exportPhpTree(),
            $this->trees
        );

        file_put_contents(
            $dir . \DIRECTORY_SEPARATOR . 'config.json',
            json_encode([
                'class'           => self::class,
                'nEstimators'     => $this->nEstimators,
                'maxDepth'        => $this->maxDepth,
                'minSamplesSplit' => $this->minSamplesSplit,
            ], \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES)
        );

        file_put_contents(
            $dir . \DIRECTORY_SEPARATOR . 'trees.json',
            json_encode($treeData, \JSON_UNESCAPED_SLASHES)
        );
    }

    public static function load(string $dir): self
    {
        $raw = file_get_contents($dir . \DIRECTORY_SEPARATOR . 'config.json');
        if ($raw === false) {
            throw new \RuntimeException("RandomForestClassifier::load — config.json missing in '$dir'.");
        }
        $config = json_decode($raw, true, 512, \JSON_THROW_ON_ERROR);

        $treesRaw = file_get_contents($dir . \DIRECTORY_SEPARATOR . 'trees.json');
        if ($treesRaw === false) {
            throw new \RuntimeException("RandomForestClassifier::load — trees.json missing in '$dir'.");
        }
        $treeData = json_decode($treesRaw, true, 512, \JSON_THROW_ON_ERROR);

        $instance = new self(
            (int) $config['nEstimators'],
            (int) $config['maxDepth'],
            (int) $config['minSamplesSplit']
        );

        foreach ($treeData as $data) {
            $instance->trees[] = DecisionTreeClassifier::fromPhpTree($data);
        }

        return $instance;
    }
}