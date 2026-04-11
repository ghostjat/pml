<?php

declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Estimators\Regression\DecisionTreeRegressor;
use RuntimeException;

/**
 * Stochastic Gradient Boosting Regressor (GBM).
 * A highly competitive ensemble model that iteratively fits weak learners to the loss gradients.
 * * JIT & Memory Optimized:
 * - Employs purely In-Place C-tensor accumulation ($F->addInplace) to prevent OOM errors.
 * - Utilizes native PHP `shuffle` driving C-level `tensor_take` for stochastic subsampling.
 */
final class GradientBoostingRegressor implements Learner
{
    private int $nEstimators;
    private float $learningRate;
    private int $maxDepth;
    private float $subsample;
    private ?int $maxFeatures;
    
    /** @var DecisionTreeRegressor[] */
    private array $trees = [];
    private float $initialPrediction = 0.0;

    /**
     * @param int $nEstimators The number of boosting stages (trees) to perform.
     * @param float $learningRate Shrinks the contribution of each tree (learning rate).
     * @param int $maxDepth Maximum depth of the individual regression estimators.
     * @param float $subsample The fraction of samples used for fitting individual trees. 
     * (Values < 1.0 result in Stochastic Gradient Boosting, reducing variance).
     * @param int|null $maxFeatures The number of features to consider when looking for the best split.
     */
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
        $n = $y->size();

        // 1. Initialize F_0(x) with the constant minimizing the loss (The Mean)
        $this->initialPrediction = $y->mean();
        
        // F holds the continuous cumulative predictions. Allocated ONCE.
        $F = Tensor::zeros($n)->addScalarInplace($this->initialPrediction);

        $subsampleCount = (int) max(1, round($n * $this->subsample));

        for ($i = 0; $i < $this->nEstimators; $i++) {
            
            // 2. Compute Pseudo-Residuals: r_m = y - F_{m-1}(x)
            // (For Mean Squared Error, the negative gradient is simply the residual)
            $residuals = $y->sub($F);

            // 3. Stochastic Subsampling (Bagging to prevent overfitting)
            if ($this->subsample < 1.0) {
                $indices = range(0, $n - 1);
                shuffle($indices);
                $indices = array_slice($indices, 0, $subsampleCount);
                
                // Zero-copy subset extraction using C-level tensor_take
                $idxT = Tensor::fromArray($indices);
                $trainX = $x->take($idxT, 0);
                $trainY = $residuals->take($idxT, 0);
                $trainDataset = new Dataset($trainX, $trainY);
            } else {
                $trainDataset = new Dataset($x, $residuals);
            }

            // 4. Fit a weak learner to the residuals
            $tree = new DecisionTreeRegressor($this->maxDepth, 2, $this->maxFeatures);
            $tree->train($trainDataset);

            // 5. Predict on the ENTIRE training set to update cumulative predictions
            // h_m(x)
            $h_m = $tree->predict(new Dataset($x));

            // 6. Update F_m(x) = F_{m-1}(x) + (learning_rate * h_m(x))
            // Executes natively in C via chained inplace mutations to prevent GC spikes
            $h_m->mulScalarInplace($this->learningRate);
            $F->addInplace($h_m);

            // Save the weak learner
            $this->trees[] = $tree;
            
            // $residuals and intermediate tensors fall out of scope here and are cleanly garbage collected.
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Gradient Boosting Regressor is not trained.");
        }

        $n = $dataset->samples()->shape()[0];
        
        // Initialize cumulative predictions array F(x)
        $F = Tensor::zeros($n)->addScalarInplace($this->initialPrediction);

        // Sequentially accumulate the scaled predictions from each tree
        foreach ($this->trees as $tree) {
            $h_m = $tree->predict($dataset);
            
            // F(x) += learning_rate * h_m(x)
            $h_m->mulScalarInplace($this->learningRate);
            $F->addInplace($h_m);
        }

        return $F;
    }

    public function trained(): bool
    {
        return !empty($this->trees);
    }
}