<?php

declare(strict_types=1);

namespace Pml\Backends;

use Pml\Dataset;
use Pml\Encoding;
use Pml\Interfaces\Learner;
use Pml\Interfaces\MLBackend;
use Pml\Interfaces\Persistable;
use Pml\Serializers\RBX;
use Pml\Tensor;
use RuntimeException;

/**
 * Classic ML backend wrapping any Pml Learner estimator.
 *
 * Persistence uses the RBX format (gzip + JSON + base64).
 * Classic estimators are pure PHP and safe to PHP-serialize; no
 * C-memory pointers are involved, so RBX is appropriate here.
 *
 * Note: classic ML estimators whose predictions are PHP arrays have
 * their output automatically wrapped in a Tensor so that the MLBackend
 * contract (returning Tensor) is fulfilled.
 */
final class RubixBackend implements MLBackend
{
    private const FILE_EXTENSION = '.rbx';

    public function __construct(private Learner $estimator) {}

    // ---- MLBackend -----------------------------------------------------------

    /**
     * Train the underlying estimator.
     * Classic estimators ignore variadic $options (no epochs, validation, etc.).
     */
    public function fit(Dataset $dataset, mixed ...$options): void
    {
        $this->estimator->train($dataset);
    }

    public function predict(Dataset $dataset): Tensor
    {
        $result = $this->estimator->predict($dataset);

        // Unwrap: estimator may already return a Tensor (C-backed estimators do).
        if ($result instanceof Tensor) {
            return $result;
        }

        // PHP-array predictions → wrap in a 1-D Tensor.
        if (\is_array($result)) {
            return Tensor::fromArray(array_map('floatval', $result));
        }

        throw new RuntimeException(
            \sprintf(
                "RubixBackend: estimator %s::predict() returned an unexpected type '%s'.",
                \get_class($this->estimator),
                \get_debug_type($result)
            )
        );
    }

    public function isTrained(): bool
    {
        return $this->estimator->trained();
    }

    /**
     * Saves the estimator as an RBX file.
     * $path may be a directory (file named 'model.rbx' is created inside)
     * or an explicit .rbx file path.
     */
    public function save(string $path): void
    {
        $filePath = $this->resolveFilePath($path);

        $dir = \dirname($filePath);
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }

        if (!$this->estimator instanceof Persistable) {
            throw new RuntimeException(
                \sprintf(
                    "RubixBackend: estimator '%s' does not implement Persistable and cannot be saved.",
                    \get_class($this->estimator)
                )
            );
        }

        $rbx     = new RBX();
        $encoded = $rbx->serialize($this->estimator);
        file_put_contents($filePath, $encoded->data());
    }

    public static function load(string $path): static
    {
        $filePath = is_dir($path)
            ? $path . \DIRECTORY_SEPARATOR . 'model' . self::FILE_EXTENSION
            : $path;

        if (!is_file($filePath)) {
            throw new RuntimeException("RubixBackend::load — file not found: '$filePath'.");
        }

        $raw = file_get_contents($filePath);
        if ($raw === false) {
            throw new RuntimeException("RubixBackend::load — cannot read '$filePath'.");
        }

        $rbx       = new RBX();
        $estimator = $rbx->unserialize(Encoding::wrap($raw));

        if (!$estimator instanceof Learner) {
            throw new RuntimeException(
                "RubixBackend::load — deserialized object does not implement Learner."
            );
        }

        return new static($estimator);
    }

    public function backendName(): string { return 'rubix'; }

    // ---- Extras --------------------------------------------------------------

    public function estimator(): Learner { return $this->estimator; }

    // -------------------------------------------------------------------------

    private function resolveFilePath(string $path): string
    {
        if (is_dir($path) || (!str_ends_with($path, self::FILE_EXTENSION) && !str_contains(\basename($path), '.'))) {
            return rtrim($path, '/\\') . \DIRECTORY_SEPARATOR . 'model' . self::FILE_EXTENSION;
        }
        return $path;
    }
}
