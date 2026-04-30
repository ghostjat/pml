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
    private int $numClasses = 0;

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
        $maxFeatures = (int) max(1, sqrt($features));
        $this->numClasses = (int)($dataset->labels()->max() + 1);

        for ($i = 0; $i < $this->nEstimators; $i++) {
            // Bootstrap: one C call replaces N PHP mt_rand() calls
            $idxT  = Tensor::bootstrapIndices($n);
            $bootX = $dataset->samples()->take($idxT, 0);
            $bootY = $dataset->labels()->take($idxT, 0);
            unset($idxT);

            $tree = new DecisionTreeClassifier($this->maxDepth, $this->minSamplesSplit, $maxFeatures);
            $tree->train(new Dataset($bootX, $bootY));
            unset($bootX, $bootY);
            $this->trees[] = $tree;
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Random Forest is not trained.");
        }

        // Stack T tree predictions into [N, T] then C majority-vote — zero PHP per-sample work
        $cols = [];
        foreach ($this->trees as $tree) {
            $cols[] = $tree->predict($dataset)->expandDims(1);  // [N, 1]
        }
        $votesMatrix = Tensor::concat($cols, 1);                // [N, T]
        return Tensor::matrixVote($votesMatrix, $this->numClasses);
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
                'numClasses'      => $this->numClasses,
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
        $instance->numClasses = (int) ($config['numClasses'] ?? 2);

        foreach ($treeData as $data) {
            $instance->trees[] = DecisionTreeClassifier::fromPhpTree($data);
        }

        return $instance;
    }
}