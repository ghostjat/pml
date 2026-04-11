<?php

declare(strict_types=1);

namespace Pml\CrossValidation;

use Pml\Tensor;
use Pml\Dataset;
use InvalidArgumentException;

/**
 * Stratified K-Fold Cross Validation.
 * Preserves the percentage of samples for each class in every fold.
 * * JIT & Memory Optimized:
 * - Resolves fold distributions natively in PHP array space to avoid matrix searches.
 * - Uses C-level `tensor_take` to extract the training and validation blocks instantly.
 * - Discarded folds are immediately garbage collected to prevent memory leakage.
 */
final class StratifiedKFold
{
    private int $k;

    public function __construct(int $k = 5)
    {
        if ($k < 2) {
            throw new InvalidArgumentException("StratifiedKFold requires at least 2 folds.");
        }
        $this->k = $k;
    }

    /**
     * Generates Stratified Folds.
     * @return \Generator<array{0: Dataset, 1: Dataset}> Yields [TrainDataset, ValidationDataset]
     */
    public function fold(Dataset $dataset): \Generator
    {
        if (!$dataset->isLabeled()) {
            throw new InvalidArgumentException("StratifiedKFold requires a labeled dataset.");
        }

        $labels = $dataset->labels()->toFlatArray();
        
        // 1. Group row indices by class to calculate strict proportions
        $classIndices = [];
        foreach ($labels as $idx => $label) {
            $classStr = (string) $label;
            if (!isset($classIndices[$classStr])) {
                $classIndices[$classStr] = [];
            }
            $classIndices[$classStr][] = $idx;
        }

        // 2. Shuffle and split each class evenly across the K folds
        $folds = array_fill(0, $this->k, []);
        foreach ($classIndices as $indices) {
            shuffle($indices);
            $count = count($indices);
            $foldSize = (int) floor($count / $this->k);
            $remainder = $count % $this->k;

            $offset = 0;
            for ($i = 0; $i < $this->k; $i++) {
                // Distribute any remainder evenly among the first few folds
                $length = $foldSize + ($i < $remainder ? 1 : 0);
                $chunk = array_slice($indices, $offset, $length);
                $folds[$i] = array_merge($folds[$i], $chunk);
                $offset += $length;
            }
        }

        // 3. Yield the folds using Zero-Copy C Pointers
        for ($i = 0; $i < $this->k; $i++) {
            $valIndices = $folds[$i];
            $trainIndices = [];

            for ($j = 0; $j < $this->k; $j++) {
                if ($i !== $j) {
                    $trainIndices = array_merge($trainIndices, $folds[$j]);
                }
            }

            // Shuffle the final fold indices to prevent sequence bias during training
            shuffle($trainIndices);
            shuffle($valIndices);

            // Pass the indices to the OpenBLAS C-Engine
            $trainT = Tensor::fromArray($trainIndices);
            $valT = Tensor::fromArray($valIndices);

            // Instantly extract the matrices using hardware-level memory extraction
            $trainSamples = $dataset->samples()->take($trainT, 0);
            $trainLabels = $dataset->labels()->take($trainT, 0);
            
            $valSamples = $dataset->samples()->take($valT, 0);
            $valLabels = $dataset->labels()->take($valT, 0);

            yield [
                new Dataset($trainSamples, $trainLabels),
                new Dataset($valSamples, $valLabels)
            ];
            
            // Memory Lifecycle: The intermediate $trainT and $valT index arrays 
            // fall out of scope here and are cleanly garbage collected by PHP!
        }
    }
}