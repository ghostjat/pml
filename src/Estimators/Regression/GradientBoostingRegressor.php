<?php

declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Estimators\Regression\DecisionTreeRegressor;
use RuntimeException;

/**
 * Stochastic Gradient Boosting Regressor (GBM).
 * A highly competitive ensemble model that iteratively fits weak learners to the loss gradients.
 * * JIT & Memory Optimized:
 * - Employs purely In-Place C-tensor accumulation ($F->addInplace) to prevent OOM errors.
 * - Eagerly reuses a pre-allocated Residual buffer to eliminate N-sized allocations per tree.
 */
final class GradientBoostingRegressor implements Learner, Persistable
{
    private int $nEstimators;
    private float $learningRate;
    private int $maxDepth;
    private float $subsample;
    private ?int $maxFeatures;
    
    /** @var DecisionTreeRegressor[] */
    private array $trees = [];
    private float $initialPrediction = 0.0;

    public function __construct(
        int $nEstimators = 100, 
        float $learningRate = 0.1, 
        int $maxDepth = 3, 
        float $subsample = 1.0,
        ?int $maxFeatures = null
    ) {
        $this->nEstimators = $nEstimators;
        $this->learningRate = $learningRate;
        $this->maxDepth = $maxDepth;
        $this->subsample = $subsample;
        $this->maxFeatures = $maxFeatures;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $y = $dataset->labels();

        if ($y === null) {
            throw new \InvalidArgumentException("Gradient Boosting requires labeled continuous data.");
        }

        $n = $x->shape()[0];
        
        // 1. Initialize F_0(x) with a constant
        $F = Tensor::zeros($n)->addScalarInplace($this->initialPrediction);

        // --- CRITICAL OPTIMIZATION: Zero-Allocation Residual Buffer ---
        // Pre-allocate the memory block once for the entire training session.
        $residuals = Tensor::emptyLike($y);

        for ($m = 0; $m < $this->nEstimators; $m++) {
            
            // 2. Compute pseudo-residuals in-place: residuals = y - F(x)
            // Copies fresh targets into the buffer and subtracts predictions natively in C.
            $residuals->copyFrom($y);
            $residuals->subInplace($F);

            if ($this->subsample < 1.0) {
                // Stochastic Gradient Boosting
                $subsampleSize = (int) max(1, $n * $this->subsample);
                $indicesTensor = Tensor::range(0, $n - 1, 1)->randomPermutation()->slice(0, 0, $subsampleSize);

                // Zero-copy view extractions
                $subX = $x->take($indicesTensor, 0);
                $subY = $residuals->take($indicesTensor, 0);
                
                unset($indicesTensor);
            } else {
                $subX = $x;
                $subY = $residuals;
            }

            // 3. Fit a weak learner to the pseudo-residuals
            $trainDataset = new Dataset($subX, $subY);
            
            $tree = new DecisionTreeRegressor($this->maxDepth, 2, $this->maxFeatures);
            $tree->train($trainDataset);

            // 4. Predict on the ENTIRE training set to update cumulative predictions
            // Use the original dataset object to prevent redundant PHP object instantiations
            $h_m = $tree->predict($dataset);

            // 5. Update F_m(x) = F_{m-1}(x) + (learning_rate * h_m(x))
            $h_m->mulScalarInplace($this->learningRate);
            $F->addInplace($h_m);

            // Save the weak learner
            $this->trees[] = $tree;
            
            // --- EAGER GARBAGE COLLECTION ---
            // Instantly drop temporary views and activations to maintain a flat memory profile
            unset($h_m, $trainDataset);
            if ($this->subsample < 1.0) {
                unset($subX, $subY);
            }
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Gradient Boosting Regressor is not trained.");
        }

        $n = $dataset->numRows();
        
        // Initialize cumulative predictions array F(x)
        $F = Tensor::zeros($n)->addScalarInplace($this->initialPrediction);

        // Sequentially accumulate the scaled predictions from each tree
        foreach ($this->trees as $tree) {
            $h_m = $tree->predict($dataset);
            
            // F(x) += learning_rate * h_m(x)
            $h_m->mulScalarInplace($this->learningRate);
            $F->addInplace($h_m);
            
            // Free the intermediate prediction tensor immediately
            unset($h_m);
        }

        return $F;
    }

    public function trained(): bool
    {
        return !empty($this->trees);
    }

    public function save(string $dir): void
    {
        if (!is_dir($dir)) { mkdir($dir, 0755, true); }

        $treeData = array_map(
            static fn(DecisionTreeRegressor $t) => $t->exportPhpTree(),
            $this->trees
        );

        file_put_contents(
            $dir . \DIRECTORY_SEPARATOR . 'config.json',
            json_encode([
                'class'             => self::class,
                'nEstimators'       => $this->nEstimators,
                'learningRate'      => $this->learningRate,
                'maxDepth'          => $this->maxDepth,
                'subsample'         => $this->subsample,
                'maxFeatures'       => $this->maxFeatures,
                'initialPrediction' => $this->initialPrediction,
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
            throw new \RuntimeException("GradientBoostingRegressor::load — config.json missing in '$dir'.");
        }
        $config = json_decode($raw, true, 512, \JSON_THROW_ON_ERROR);

        $treesRaw = file_get_contents($dir . \DIRECTORY_SEPARATOR . 'trees.json');
        if ($treesRaw === false) {
            throw new \RuntimeException("GradientBoostingRegressor::load — trees.json missing in '$dir'.");
        }
        $treeData = json_decode($treesRaw, true, 512, \JSON_THROW_ON_ERROR);

        $instance = new self(
            (int)   $config['nEstimators'],
            (float) $config['learningRate'],
            (int)   $config['maxDepth'],
            (float) $config['subsample'],
            isset($config['maxFeatures']) ? (int) $config['maxFeatures'] : null
        );
        $instance->initialPrediction = (float) $config['initialPrediction'];

        foreach ($treeData as $data) {
            $instance->trees[] = DecisionTreeRegressor::fromPhpTree($data);
        }

        return $instance;
    }
}