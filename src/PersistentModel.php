<?php
declare(strict_types=1);

namespace Pml;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Persisters\Persister;
use Pml\Serializers\Serializer;
use Pml\Serializers\Native;
use RuntimeException;

/**
 * Persistent Model — wraps any Learner with automatic save/load via a Persister.
 * Delegates all train/predict calls to the wrapped estimator.
 *
 * JIT & Memory Optimized:
 * - The wrapped estimator is accessed via a direct property reference — no proxy overhead.
 * - Save/load crosses the FFI boundary exactly once per Tensor parameter (via serialize).
 */
final class PersistentModel implements Learner
{
    public function __construct(
        private readonly Learner   $estimator,
        private readonly Persister $persister,
        private readonly Serializer $serializer = new Native()
    ) {
        if (!$estimator instanceof Persistable) {
            throw new \InvalidArgumentException(
                "Wrapped estimator must implement Persistable."
            );
        }
    }

    public function train(Dataset $dataset): void
    {
        $this->estimator->train($dataset);
    }

    public function predict(Dataset $dataset): Tensor
    {
        return $this->estimator->predict($dataset);
    }

    /**
     * Serialize and persist the wrapped model to the configured storage.
     */
    public function save(): void
    {
        /** @var Persistable $model */
        $model    = $this->estimator;
        $encoding = $this->serializer->serialize($model);
        $this->persister->save($encoding);
    }

    /**
     * Load and return a model from persistent storage.
     */
    public static function load(Persister $persister, Serializer $serializer): Persistable
    {
        $encoding = $persister->load();
        return $serializer->unserialize($encoding);
    }

    public function trained(): bool
    {
        return method_exists($this->estimator, 'trained') && $this->estimator->trained();
    }
}
