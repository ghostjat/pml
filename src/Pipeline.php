<?php

declare(strict_types=1);

namespace Pml;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Interfaces\Stateful;
use Pml\Interfaces\Transformer;
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
    // PERSISTENCE — SafeTensors + JSON bundle
    //
    // Saved layout:
    //   $dir/config.json                — class names, no C-data
    //   $dir/transformers.safetensors  — Stateful transformer tensors (if any)
    //   $dir/estimator/                — estimator sub-directory (Persistable)
    //
    // Transformers are overwhelmingly pure-PHP (scalers, encoders, etc.).
    // Their PHP object shells are serialised directly; any that implement
    // Stateful have their Tensor state extracted to SafeTensors first.
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

            if ($transformer instanceof Stateful) {
                foreach ($transformer->getStateDict($prefix) as $key => $tensor) {
                    $tensorDict[$key] = $tensor;
                }
                $transformerCfg[] = [
                    'stateful' => true,
                    'prefix'   => $prefix,
                    'shell'    => base64_encode(serialize(self::stripTensors($transformer))),
                ];
            } else {
                // Strip any Tensor properties (e.g. fitted scalers hold live C-buffers)
                // before serialising — FFI\CData must never reach serialize().
                $transformerCfg[] = [
                    'stateful' => false,
                    'shell'    => base64_encode(serialize(self::stripTensors($transformer))),
                ];
            }
        }

        if (!empty($tensorDict)) {
            SafeTensorsIO::save(
                $dir . \DIRECTORY_SEPARATOR . 'transformers.safetensors',
                $tensorDict
            );
        }

        if ($this->estimator instanceof Persistable) {
            $this->estimator->save($dir . \DIRECTORY_SEPARATOR . 'estimator');
        }

        $config = [
            'class'           => self::class,
            'estimator_class' => \get_class($this->estimator),
            'transformers'    => $transformerCfg,
        ];

        file_put_contents(
            $dir . \DIRECTORY_SEPARATOR . 'config.json',
            json_encode($config, \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES)
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

        // Load Stateful transformer tensors once (zero-copy mmap).
        $stPath  = $dir . \DIRECTORY_SEPARATOR . 'transformers.safetensors';
        $weights = is_file($stPath) ? SafeTensorsIO::load($stPath) : [];

        $transformers = [];
        foreach ($config['transformers'] as $cfg) {
            $transformer = unserialize(base64_decode($cfg['shell']));
            if ($cfg['stateful'] && $transformer instanceof Stateful) {
                $transformer->loadStateDict($weights, $cfg['prefix']);
            }
            $transformers[] = $transformer;
        }

        $estimatorClass = $config['estimator_class'];
        if (!is_subclass_of($estimatorClass, Persistable::class)) {
            throw new \RuntimeException(
                "Pipeline::load — estimator '$estimatorClass' does not implement Persistable."
            );
        }
        $estimator = $estimatorClass::load($dir . \DIRECTORY_SEPARATOR . 'estimator');

        return new self($transformers, $estimator);
    }

    // -------------------------------------------------------------------------

    private static function stripTensors(object $obj): object
    {
        $clone = clone $obj;
        $class = \get_class($clone);

        foreach ((new \ReflectionClass($clone))->getProperties() as $prop) {
            $type = $prop->getType();
            if (!$type instanceof \ReflectionNamedType) {
                continue;
            }
            $typeName = $type->getName();
            if ($typeName !== Tensor::class && !is_subclass_of($typeName, Tensor::class)) {
                continue;
            }

            $prop->setAccessible(true);

            if ($type->allowsNull()) {
                $prop->setValue($clone, null);
            } else {
                $name = $prop->getName();
                \Closure::bind(
                    static function (object $o) use ($name): void { unset($o->$name); },
                    null,
                    $class
                )($clone);
            }
        }

        return $clone;
    }
}