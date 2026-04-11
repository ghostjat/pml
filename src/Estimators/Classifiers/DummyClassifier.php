<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Dummy Classifier.
 * Acts as a baseline sanity-check by always predicting the most frequent class.
 */
final class DummyClassifier implements Learner
{
    private ?float $mode = null;

    public function train(Dataset $dataset): void
    {
        // Extract the most frequent class natively in C using bincount -> argmax
        $this->mode = (float) $dataset->labels()->bincount()->argmax();
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new \RuntimeException("DummyClassifier is not trained.");
        }

        // Return a tensor filled with the mode
        return Tensor::zeros($dataset->numRows())->addScalarInplace($this->mode);
    }

    public function trained(): bool
    {
        return $this->mode !== null;
    }
}