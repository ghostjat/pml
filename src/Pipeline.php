<?php

declare(strict_types=1);

namespace Pml;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Transformer;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Machine Learning Pipeline.
 * Orchestrates a sequence of Data Transformers ending in a final Estimator/Model.
 * Prevents data leakage by ensuring transformations are fitted only on training data.
 */
final class Pipeline implements Learner, Persistable
{
    /** @var Transformer[] */
    private array $transformers;
    private Learner $estimator;

    public function __construct(array $transformers, Learner $estimator)
    {
        $this->transformers = $transformers;
        $this->estimator = $estimator;
    }

    /**
     * Passes the dataset through all transformers, fitting them if necessary, 
     * before passing the final data to the underlying estimator's train method.
     */
    public function train(Dataset $dataset, ...$args): void
    {
        $currentDataset = $dataset;

        foreach ($this->transformers as $transformer) {
            $transformer->fit($currentDataset);
            $currentDataset = $transformer->transform($currentDataset);
        }

        // Dynamically invoke the underlying estimator's train method to pass through 
        // complex arguments (like epochs, validation sets) required by Sequential networks.
        $this->estimator->train($currentDataset, ...$args);
    }

    /**
     * Transforms the inference dataset and delegates the prediction to the estimator.
     */
    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new \RuntimeException("Pipeline is not trained.");
        }

        $currentDataset = $dataset;

        foreach ($this->transformers as $transformer) {
            $currentDataset = $transformer->transform($currentDataset);
        }

        return $this->estimator->predict($currentDataset);
    }

    public function trained(): bool
    {
        return $this->estimator->trained();
    }

    // ========================================================================
    // PERSISTENCE
    // ========================================================================

    public function save(string $filepath): void
    {
        if (!is_dir($filepath)) {
            mkdir($filepath, 0777, true);
        }

        // Delegate estimator saving if it supports persistence
        if ($this->estimator instanceof Persistable) {
            $this->estimator->save($filepath . DIRECTORY_SEPARATOR . 'estimator');
        }

        // Strip C-Pointers and Serialize the Pipeline wrapper
        // Note: Production implementation would safely detach $this->min, $this->categories etc.
        $manifest = [
            'transformers' => serialize($this->transformers),
            'estimator_class' => get_class($this->estimator)
        ];

        file_put_contents($filepath . DIRECTORY_SEPARATOR . 'pipeline.json', json_encode($manifest));
    }

    public static function load(string $filepath): self
    {
        $manifestJson = file_get_contents($filepath . DIRECTORY_SEPARATOR . 'pipeline.json');
        $manifest = json_decode($manifestJson, true);

        $transformers = unserialize($manifest['transformers']);
        
        $estimatorClass = $manifest['estimator_class'];
        if (is_subclass_of($estimatorClass, Persistable::class)) {
            $estimator = $estimatorClass::load($filepath . DIRECTORY_SEPARATOR . 'estimator');
        } else {
            throw new \RuntimeException("Underlying estimator does not support loading.");
        }

        return new self($transformers, $estimator);
    }
}