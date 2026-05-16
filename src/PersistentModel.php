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
 * @deprecated Use ModelStore::save() + SafeTensorsIO instead (zero PHP serialize()).
 *
 * PersistentModel wraps a Learner with a Persister + Serializer. The Native
 * and GzipNative serializers use PHP serialize() which cannot safely round-trip
 * FFI\CData / Tensor values. This class will be removed in the next major version.
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
