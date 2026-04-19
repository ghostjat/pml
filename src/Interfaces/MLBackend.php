<?php

declare(strict_types=1);

namespace Pml\Interfaces;

use Pml\Dataset;
use Pml\Tensor;

/**
 * Unified interface for pluggable ML backends.
 *
 * A backend wraps a concrete model (Sequential for deep learning,
 * a classic ML estimator for Rubix-style models, etc.) and exposes
 * a common train/predict/save/load contract that higher-level
 * orchestration code (Trainer, ModelHub) can drive without caring
 * about the underlying implementation.
 *
 * Implementations:
 *   - TorchBackend  — wraps Sequential (FFI / C-native tensors)
 *   - RubixBackend  — wraps classic Learner estimators (PHP-native)
 */
interface MLBackend
{
    /**
     * Train the underlying model on $dataset, forwarding any
     * backend-specific options via the variadic $options.
     *
     * Deep-learning backends accept: epochs, batchSize, validation,
     * patience, minDelta.  Classic backends may ignore all options.
     */
    public function fit(Dataset $dataset, mixed ...$options): void;

    /**
     * Run inference on $dataset and return raw predictions as a Tensor.
     */
    public function predict(Dataset $dataset): Tensor;

    /**
     * Whether the underlying model has been trained at least once.
     */
    public function isTrained(): bool;

    /**
     * Persist the model to $dir (or a file path for single-file formats).
     */
    public function save(string $path): void;

    /**
     * Restore the model from $dir / file path.
     * Must return a new backend instance wrapping the loaded model.
     */
    public static function load(string $path): static;

    /**
     * Human-readable backend name, e.g. "torch" or "rubix".
     * Used by ModelHub to select the correct loader.
     */
    public function backendName(): string;
}
