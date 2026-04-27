# Utilities API

This section covers persistence, model wrapping, and helper classes.

## `Pml\Lib\ModelStore`

A generic persistence engine that saves PHP state and tensor weights separately.

### Methods

```php
public static function toArray(object $model): array
public static function fromArray(array $data): object
public static function save(object $model, string $dir): void
public static function load(string $dir): object
public static function extractTensors(object $model): array
public static function injectTensors(object $model, array $tensors): void
```

### What it does

- `toArray()` converts PHP-only state to a JSON-safe array.
- `extractTensors()` returns a flat map of tensor state.
- `save()` writes `config.json` and `state.safetensors`.
- `load()` reconstructs an object and restores tensor weights.

## `Pml\Lib\SafeTensorsIO`

Handles HuggingFace-compatible SafeTensors file I/O.

### Methods

```php
public static function save(string $filepath, array $tensors): void
public static function load(string $filepath): array
```

### What it does

- Writes SafeTensors headers and raw tensor bytes in C.
- Loads tensors as mmap-backed `Tensor` objects.
- Avoids copying tensor bytes into PHP heap.

## `Pml\PersistentModel`

Wraps a `Learner` and provides storage via a `Persister` and `Serializer`.

### Methods

```php
public function __construct(Learner $estimator, Persister $persister, Serializer $serializer = new Native())
public function train(Dataset $dataset): void
public function predict(Dataset $dataset): Tensor
public function save(): void
public static function load(Persister $persister, Serializer $serializer): Persistable
public function trained(): bool
```

### Persistence helpers

- `Pml\Persisters\Filesystem` — atomic disk-based persister.
- `Pml\Serializers\Native` — PHP `serialize()` wrapper used for portable persistence.
- `Pml\Encoding` — binary wrapper with base64 and compression support.

## Example usage

```php
use Pml\Persisters\Filesystem;
use Pml\Serializers\Native;
use Pml\PersistentModel;

$persister = new Filesystem('saved_models/model.bin');
$wrapper = new PersistentModel($model, $persister, new Native());
$wrapper->save();

$loaded = PersistentModel::load($persister, new Native());
```

## Common mistakes

- Using `Native` serializer for large tensor models when SafeTensors persistence is preferable.
- Passing a non-`Persistable` estimator to `PersistentModel`.
- Ignoring `ModelStore` tensor extraction semantics and expecting `FFI\CData` objects to be serialized directly.
