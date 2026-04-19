<?php

declare(strict_types=1);

namespace Pml\Lib;

use Pml\Interfaces\Saveable;
use Pml\Interfaces\Stateful;
use Pml\Tensor;

/**
 * Universal model persistence engine — zero serialize(), zero FFI\CData exposure.
 *
 * ── Two usage modes ─────────────────────────────────────────────────────────
 *
 *  Inline  toArray(obj)  / fromArray(data)
 *    Converts any object to/from a JSON-safe PHP array.
 *    Used by Sequential to embed loss/optimizer/layers directly in config.json.
 *
 *  Disk    save(obj, $dir) / load($dir)
 *    Writes config.json (PHP state) + state.safetensors (Tensor state) to $dir.
 *    Used by Pipeline for transformers and estimators.
 *
 * ── Protocol priority ───────────────────────────────────────────────────────
 *
 *  PHP state:
 *    Saveable present  → getConfig() + getPhpState()       (explicit, safest)
 *    Saveable absent   → Reflection: constructor params + scalar/array properties
 *
 *  Tensor state:
 *    Stateful present  → getStateDict() / loadStateDict()  (zero-copy)
 *    Stateful absent   → Reflection: all ?Tensor-typed properties
 *
 * ── Guarantee ───────────────────────────────────────────────────────────────
 *   FFI\CData is NEVER written to disk.
 *   Tensor C-memory is NEVER written to disk; bytes go through SafeTensors only.
 *   No PHP serialize() / unserialize() is ever called.
 */
final class ModelStore
{
    // ══════════════════════════════════════════════════════════════ Inline API

    /**
     * Convert any object to a JSON-safe PHP array.
     *
     * Nested PHP-only objects (e.g. ActivationFunction inside Activation) are
     * recursively encoded via the same mechanism and tagged with '__object__'.
     * Tensor and FFI\CData values are silently omitted (handled separately).
     *
     * @return array{class: string, config: array, state: array}
     */
    public static function toArray(object $model): array
    {
        $ref  = new \ReflectionClass($model);
        $data = ['class' => $ref->getName()];

        if ($model instanceof Saveable) {
            $data['config'] = $model->getConfig();
            $data['state']  = $model->getPhpState();
        } else {
            [$data['config'], $data['state']] = self::reflectExtract($model, $ref);
        }

        return $data;
    }

    /**
     * Reconstruct an object from a toArray() snapshot.
     *
     * Tensor state is NOT restored here — call injectTensors() / loadStateDict()
     * afterwards for any class that implements Stateful.
     */
    public static function fromArray(array $data): object
    {
        $class  = $data['class'];
        $config = $data['config'] ?? [];
        $state  = $data['state']  ?? [];

        if (!class_exists($class)) {
            throw new \RuntimeException("ModelStore::fromArray — class '$class' not found.");
        }

        $ref = new \ReflectionClass($class);

        if (self::implements($class, Saveable::class)) {
            /** @var Saveable $model */
            $model = $class::fromConfig($config);
            $model->setPhpState($state);
        } else {
            $model = self::reflectConstruct($ref, $config);
            self::reflectSetState($model, $ref, $state);
        }

        return $model;
    }

    // ══════════════════════════════════════════════════════════════ Disk API

    /**
     * Save $model to $dir.
     *   $dir/config.json          — PHP state (class, hyperparams, scalars)
     *   $dir/state.safetensors    — Tensor weights (present only when non-empty)
     */
    public static function save(object $model, string $dir): void
    {
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }

        $tensorDict = self::extractTensors($model);
        if (!empty($tensorDict)) {
            SafeTensorsIO::save($dir . \DIRECTORY_SEPARATOR . 'state.safetensors', $tensorDict);
        }

        file_put_contents(
            $dir . \DIRECTORY_SEPARATOR . 'config.json',
            json_encode(self::toArray($model), \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES)
        );
    }

    /**
     * Load a model saved by save().
     * Returns a fully-reconstructed object with all Tensor state injected.
     */
    public static function load(string $dir): object
    {
        $raw = file_get_contents($dir . \DIRECTORY_SEPARATOR . 'config.json');
        if ($raw === false) {
            throw new \RuntimeException("ModelStore::load — config.json missing in '$dir'.");
        }

        $model = self::fromArray(json_decode($raw, true, 512, \JSON_THROW_ON_ERROR));

        $stPath = $dir . \DIRECTORY_SEPARATOR . 'state.safetensors';
        if (is_file($stPath)) {
            self::injectTensors($model, SafeTensorsIO::load($stPath));
        }

        return $model;
    }

    // ══════════════════════════════════════════════════════════ Tensor helpers

    /**
     * Extract all Tensor state from $model as a flat name → Tensor map.
     *
     * Uses Stateful::getStateDict('') when available; otherwise Reflection
     * scans for properties with a declared type of Tensor (or nullable Tensor).
     *
     * @return array<string, Tensor>
     */
    public static function extractTensors(object $model): array
    {
        if ($model instanceof Stateful) {
            return $model->getStateDict('');
        }

        // Saveable without Stateful → class has explicitly declared no Tensor state.
        if ($model instanceof Saveable) {
            return [];
        }

        return self::reflectScanTensors($model);
    }

    /**
     * Inject Tensors into $model.
     *
     * Uses Stateful::loadStateDict($tensors, '') when available; otherwise
     * Reflection injects each tensor by matching on property name.
     *
     * @param array<string, Tensor> $tensors
     */
    public static function injectTensors(object $model, array $tensors): void
    {
        if (empty($tensors)) {
            return;
        }

        if ($model instanceof Stateful) {
            $model->loadStateDict($tensors, '');
            return;
        }

        $ref = new \ReflectionClass($model);
        foreach ($tensors as $name => $tensor) {
            if (!$ref->hasProperty($name)) {
                continue;
            }
            $prop = $ref->getProperty($name);
            $prop->setAccessible(true);
            $prop->setValue($model, $tensor);
        }
    }

    // ══════════════════════════════════════════════════════════ Reflection core

    /**
     * Extract constructor-param properties → $config, all other scalar/array
     * properties → $state.  Nested PHP objects are recursively encoded.
     * Tensor and FFI\CData values are silently skipped.
     *
     * @return array{array, array}  [$config, $state]
     */
    private static function reflectExtract(object $model, \ReflectionClass $ref): array
    {
        $config         = [];
        $state          = [];
        $ctorParamNames = [];

        if ($ctor = $ref->getConstructor()) {
            foreach ($ctor->getParameters() as $p) {
                $ctorParamNames[] = $p->getName();
            }
        }

        foreach ($ref->getProperties() as $prop) {
            $prop->setAccessible(true);
            if (!$prop->isInitialized($model)) {
                continue;
            }

            $name = $prop->getName();
            $val  = $prop->getValue($model);

            // ── Skip C-memory ───────────────────────────────────────────────
            if ($val instanceof Tensor || $val instanceof \FFI\CData) {
                continue;
            }

            // ── Also skip by type hint (catches typed null properties) ──────
            $type = $prop->getType();
            if ($type instanceof \ReflectionNamedType) {
                $typeName = $type->getName();
                if ($typeName === Tensor::class
                    || is_subclass_of($typeName, Tensor::class)
                    || $typeName === \FFI\CData::class) {
                    continue;
                }
            }

            // ── Arrays must contain only plain values ────────────────────────
            if (is_array($val) && !self::isPlainArray($val)) {
                continue;
            }

            // ── Nested PHP objects — recurse ─────────────────────────────────
            if (is_object($val)) {
                $encoded = ['__object__' => true, 'data' => self::toArray($val)];
                if (in_array($name, $ctorParamNames, true)) {
                    $config[$name] = $encoded;
                } else {
                    $state[$name] = $encoded;
                }
                continue;
            }

            if (in_array($name, $ctorParamNames, true)) {
                $config[$name] = $val;
            } else {
                $state[$name] = $val;
            }
        }

        return [$config, $state];
    }

    /**
     * Reconstruct an object by calling its constructor with args from $config.
     */
    private static function reflectConstruct(\ReflectionClass $ref, array $config): object
    {
        $ctor = $ref->getConstructor();
        if ($ctor === null || empty($config)) {
            return $ref->newInstanceWithoutConstructor();
        }

        $args = [];
        foreach ($ctor->getParameters() as $p) {
            $name = $p->getName();
            if (!array_key_exists($name, $config)) {
                $args[] = $p->isOptional() ? $p->getDefaultValue() : null;
                continue;
            }
            $val    = $config[$name];
            $args[] = (is_array($val) && isset($val['__object__'])) ? self::fromArray($val['data']) : $val;
        }

        return $ref->newInstanceArgs($args);
    }

    /**
     * Set non-constructor-param state properties on a reconstructed object.
     */
    private static function reflectSetState(object $model, \ReflectionClass $ref, array $state): void
    {
        foreach ($state as $name => $val) {
            if (!$ref->hasProperty($name)) {
                continue;
            }
            $prop = $ref->getProperty($name);
            $prop->setAccessible(true);
            if (is_array($val) && isset($val['__object__'])) {
                $val = self::fromArray($val['data']);
            }
            try {
                $prop->setValue($model, $val);
            } catch (\Error) {
                // readonly property already set by constructor — skip
            }
        }
    }

    /**
     * Reflection scan for Tensor-typed properties (Stateful fallback).
     *
     * @return array<string, Tensor>
     */
    private static function reflectScanTensors(object $model): array
    {
        $dict = [];
        foreach ((new \ReflectionClass($model))->getProperties() as $prop) {
            $type = $prop->getType();
            if (!$type instanceof \ReflectionNamedType) {
                continue;
            }
            $typeName = $type->getName();
            if ($typeName !== Tensor::class && !is_subclass_of($typeName, Tensor::class)) {
                continue;
            }
            $prop->setAccessible(true);
            if (!$prop->isInitialized($model)) {
                continue;
            }
            $val = $prop->getValue($model);
            if ($val instanceof Tensor) {
                $dict[$prop->getName()] = $val;
            }
        }
        return $dict;
    }

    /**
     * Returns true iff every value in $arr is a PHP scalar, null, or a
     * recursively-plain sub-array (no Tensor, CData, or other objects).
     */
    private static function isPlainArray(array $arr): bool
    {
        foreach ($arr as $v) {
            if ($v instanceof Tensor || $v instanceof \FFI\CData || is_object($v)) {
                return false;
            }
            if (is_array($v) && !self::isPlainArray($v)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Check if $class implements $interface (handles both direct and inherited).
     */
    private static function implements(string $class, string $interface): bool
    {
        return in_array($interface, class_implements($class) ?: [], true);
    }
}
