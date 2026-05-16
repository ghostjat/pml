<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Interfaces\RanksFeatures;
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
final class RandomForestClassifier implements Learner, Persistable, RanksFeatures
{
    private int $nEstimators;
    private int $maxDepth;
    private int $minSamplesSplit;
    private int $numClasses     = 0;
    private int $nFeatures      = 0;

    /** Packed [T * maxNodes] HardwareNode flat buffer built after training (§29). */
    private ?\FFI\CData $forestNodes    = null;
    private int          $forestMaxNodes = 0;

    /** @var DecisionTreeClassifier[] */
    private array $trees = [];

    public function __construct(int $nEstimators = 100, int $maxDepth = 10, int $minSamplesSplit = 2)
    {
        $this->nEstimators = $nEstimators;
        $this->maxDepth = $maxDepth;
        $this->minSamplesSplit = $minSamplesSplit;
    }

    public function train(Dataset $dataset, mixed ...$options): void
    {
        $n = $dataset->numRows();
        $features = $dataset->numColumns();
        $this->nFeatures = $features;
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
        $this->packForestNodes();
    }

    /**
     * Pack all per-tree HardwareNode arrays into a single flat FFI buffer [T * maxNodes].
     * Done once after training; enables single-call batch predict in C (§29).
     */
    private function packForestNodes(): void
    {
        $ffi      = \Pml\Lib\TensorEngine::get();
        $T        = count($this->trees);
        $maxNodes = 0;
        foreach ($this->trees as $tree) {
            $maxNodes = max($maxNodes, $tree->numHardwareNodes());
        }
        $this->forestMaxNodes = $maxNodes;
        $this->forestNodes    = $ffi->new("HardwareNode[$T * $maxNodes]");

        // Sentinel: feature_idx = -1 marks an unused node slot
        $nodeSize = \FFI::sizeof($ffi->new('HardwareNode'));
        for ($t = 0; $t < $T; $t++) {
            $tree      = $this->trees[$t];
            $treeNodes = $tree->hardwareNodes();
            $treeCount = $tree->numHardwareNodes();
            $destPtr   = \FFI::addr($this->forestNodes[$t * $maxNodes]);
            \FFI::memcpy($destPtr, $treeNodes, $treeCount * $nodeSize);
            // Remaining slots are zero-initialized (feature_idx=0 but left_idx=-1 from calloc).
            // Sentinel: fill remaining with feature_idx = -1, left_idx = -1 so traversal stops.
            for ($j = $treeCount; $j < $maxNodes; $j++) {
                $this->forestNodes[$t * $maxNodes + $j]->feature_idx = -1;
                $this->forestNodes[$t * $maxNodes + $j]->left_idx    = -1;
                $this->forestNodes[$t * $maxNodes + $j]->right_idx   = -1;
                $this->forestNodes[$t * $maxNodes + $j]->value       = 0.0;
            }
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Random Forest is not trained.");
        }

        $ffi = \Pml\Lib\TensorEngine::get();
        $T   = count($this->trees);
        $N   = $dataset->numRows();

        // Pre-allocate [N, T] output — C writes all T tree predictions in one call (§29)
        $shape    = $ffi->new('int[2]');
        $shape[0] = $N;
        $shape[1] = $T;
        $votesMatrix = Tensor::wrap(
            $ffi->tensor_create_dtype(2, $ffi->cast('int*', $shape), Tensor::DTYPE_FLOAT32)
        );

        $ffi->tensor_rf_predict_batch(
            $dataset->samples()->ptr,
            $ffi->cast('HardwareNode*', $this->forestNodes),
            $T, $this->forestMaxNodes,
            $votesMatrix->ptr
        );

        return Tensor::matrixVote($votesMatrix, $this->numClasses);
    }

    public function trained(): bool
    {
        return !empty($this->trees);
    }

    public function featureImportances(): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Random Forest is not trained.");
        }
        // Aggregate split counts across all trees in PHP — one Tensor::fromArray() at the end.
        $agg = array_fill(0, $this->nFeatures, 0.0);
        foreach ($this->trees as $tree) {
            foreach ($tree->featureSplitCounts() as $f => $c) {
                $agg[$f] += $c;
            }
        }
        $total = array_sum($agg);
        if ($total > 0.0) {
            foreach ($agg as &$v) { $v /= $total; }
        }
        return Tensor::fromArray($agg);
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
                'nFeatures'       => $this->nFeatures,
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
        $instance->nFeatures  = (int) ($config['nFeatures']  ?? 0);

        foreach ($treeData as $data) {
            $instance->trees[] = DecisionTreeClassifier::fromPhpTree($data);
        }
        $instance->packForestNodes();

        return $instance;
    }
}