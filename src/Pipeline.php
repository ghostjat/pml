<?php

declare(strict_types=1);

namespace Pml;

use Pml\Interfaces\FitTransformable;
use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Interfaces\TrainableWithOptions;
use Pml\Interfaces\Transformer;
use Pml\Lib\ModelStore;
use Pml\Lib\SafeTensorsIO;
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
    public function train(Dataset $dataset, mixed ...$args): void
    {
        $currentDataset = $dataset;

        foreach ($this->transformers as $transformer) {
            // FitTransformable fuses fit+transform into one dataset scan (O(n) vs 2×O(n)).
            if ($transformer instanceof FitTransformable) {
                $currentDataset = $transformer->fitTransform($currentDataset);
            } else {
                $transformer->fit($currentDataset);
                $currentDataset = $transformer->transform($currentDataset);
            }
        }

        // Forward variadic training options (epochs, validation, patience …) only when
        // the estimator explicitly declares support via TrainableWithOptions.
        if ($this->estimator instanceof TrainableWithOptions) {
            $this->estimator->train($currentDataset, ...$args);
        } else {
            $this->estimator->train($currentDataset);
        }
    }

    /**
     * Transforms the inference dataset and delegates the prediction to the estimator.
     * Guards each transformer with a fitted() check so a stale or mis-loaded
     * pipeline fails loudly rather than producing silently corrupted predictions.
     */
    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new \RuntimeException("Pipeline is not trained.");
        }

        $currentDataset = $dataset;

        foreach ($this->transformers as $i => $transformer) {
            if (!$transformer->fitted()) {
                throw new \RuntimeException(
                    \sprintf(
                        "Transformer %d (%s) is not fitted. Call train() or load a saved pipeline first.",
                        $i,
                        \get_class($transformer)
                    )
                );
            }
            $currentDataset = $transformer->transform($currentDataset);
        }

        return $this->estimator->predict($currentDataset);
    }

    public function trained(): bool
    {
        return $this->estimator->trained();
    }

    // ========================================================================
    // PERSISTENCE — ModelStore-based, zero serialize(), zero FFI\CData
    //
    // Saved layout:
    //   $dir/config.json               — transformer + estimator metadata
    //   $dir/transformers.safetensors  — Tensor state for all transformers
    //   $dir/estimator/                — estimator directory (ModelStore or Persistable)
    //
    // Every transformer is encoded via ModelStore::toArray() (Reflection or
    // Saveable — never serialize()).  Tensor weights are collected with a
    // per-transformer prefix and written to a single SafeTensors file.
    // ========================================================================

    public function save(string $dir): void
    {
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }

        $transformerCfg = [];
        $tensorDict     = [];

        foreach ($this->transformers as $i => $transformer) {
            $prefix = "transformer_{$i}.";

            // Collect Tensor state with prefix (Stateful or Reflection scan).
            foreach (ModelStore::extractTensors($transformer) as $key => $tensor) {
                $tensorDict[$prefix . $key] = $tensor;
            }

            $transformerCfg[] = [
                'prefix' => $prefix,
                'data'   => ModelStore::toArray($transformer),
            ];
        }

        if (!empty($tensorDict)) {
            SafeTensorsIO::save(
                $dir . \DIRECTORY_SEPARATOR . 'transformers.safetensors',
                $tensorDict
            );
        }

        // Estimator: prefer its own Persistable::save() if available, else ModelStore.
        if ($this->estimator instanceof Persistable) {
            $this->estimator->save($dir . \DIRECTORY_SEPARATOR . 'estimator');
        } else {
            ModelStore::save($this->estimator, $dir . \DIRECTORY_SEPARATOR . 'estimator');
        }

        file_put_contents(
            $dir . \DIRECTORY_SEPARATOR . 'config.json',
            json_encode([
                'class'           => self::class,
                'estimator_class' => \get_class($this->estimator),
                'estimator_mode'  => $this->estimator instanceof Persistable ? 'persistable' : 'modelstore',
                'transformers'    => $transformerCfg,
            ], \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES)
        );
    }

    public static function load(string $dir): self
    {
        $raw = file_get_contents($dir . \DIRECTORY_SEPARATOR . 'config.json');
        if ($raw === false) {
            throw new \RuntimeException("Pipeline::load — config.json missing in '$dir'.");
        }

        /** @var array<string,mixed> $config */
        $config = json_decode($raw, true, 512, \JSON_THROW_ON_ERROR);

        // Load all transformer Tensor weights once (zero-copy mmap).
        $stPath  = $dir . \DIRECTORY_SEPARATOR . 'transformers.safetensors';
        $weights = is_file($stPath) ? SafeTensorsIO::load($stPath) : [];

        $transformers = [];
        foreach ($config['transformers'] as $cfg) {
            $transformer = ModelStore::fromArray($cfg['data']);

            // Filter the weight dict to this transformer's prefix, strip prefix before inject.
            $prefix      = $cfg['prefix'];
            $localTensors = [];
            foreach ($weights as $k => $tensor) {
                if (\str_starts_with($k, $prefix)) {
                    $localTensors[\substr($k, \strlen($prefix))] = $tensor;
                }
            }
            if (!empty($localTensors)) {
                ModelStore::injectTensors($transformer, $localTensors);
            }

            $transformers[] = $transformer;
        }

        // Estimator: use Persistable::load() or ModelStore::load() to match save().
        $estimatorClass = $config['estimator_class'];
        $estimatorDir   = $dir . \DIRECTORY_SEPARATOR . 'estimator';
        $mode           = $config['estimator_mode'] ?? 'persistable';

        if ($mode === 'persistable' && \is_subclass_of($estimatorClass, Persistable::class)) {
            $estimator = $estimatorClass::load($estimatorDir);
        } else {
            $estimator = ModelStore::load($estimatorDir);
        }

        return new self($transformers, $estimator);
    }
}