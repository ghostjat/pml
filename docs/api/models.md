# Models API

This section describes model interfaces, pipelines, and the main model persistence mechanisms.

## Core interfaces

### `Pml\Interfaces\Estimator`

```php
public function predict(Dataset $dataset): Tensor;
```

All estimators produce predictions from a dataset.

### `Pml\Interfaces\Learner`

```php
public function train(Dataset $dataset): void;
public function trained(): bool;
```

`Learner` extends `Estimator` and exposes training capability.

### `Pml\Interfaces\TrainableWithOptions`

```php
public function train(Dataset $dataset, mixed ...$options): void;
```

Used by deep learning models like `Pml\NeuralNetwork\Sequential` to accept optimizer and epoch options.

### `Pml\Interfaces\Persistable`

```php
public function save(string $filepath): void;
public static function load(string $filepath): self;
```

This interface is used by models and pipelines that support disk persistence without PHP `serialize()`.

### `Pml\Interfaces\Probabilistic`

```php
public function proba(Dataset $dataset): Tensor;
```

Used by classifiers that expose probability outputs.

## Pipeline

### `Pml\Pipeline`

A pipeline sequences transformers and a final estimator.

```php
public function __construct(array $transformers, Learner $estimator)
public function train(Dataset $dataset, mixed ...$args): void
public function predict(Dataset $dataset): Tensor
public function trained(): bool
public function save(string $dir): void
public static function load(string $dir): self
```

`Pipeline::train()` fits each transformer and then trains the estimator.
`Pipeline::predict()` applies fitted transformers and calls the estimator.

## Neural network container

### `Pml\NeuralNetwork\Sequential`

```php
public function __construct(array $layers, Loss $lossFn, Optimizer $optimizer)
public function add(Layers\Layer $layer): void
public function train(Dataset $dataset, mixed ...$options): void
public function predict(Dataset $dataset): Tensor
public function trained(): bool
public function save(string $dir): void
public static function load(string $dir): self
```

`Sequential` is the deep learning model container. It supports training with dataset batching, validation, early stopping, and checkpointing.

Supported training options include:

- `epochs`
- `batchSize`
- `validation`
- `patience`
- `minDelta`
- `clipGradNorm`

## Example model: GBDT Regressor

The `Pml\Estimators\Regression\GBDTRegressor` implementation trains using C-backed GBDT kernels.

```php
use Pml\Estimators\Regression\GBDTRegressor;

$model = new GBDTRegressor(nEstimators: 50, maxDepth: 4, numBins: 128);
$model->train($dataset);
$predictions = $model->predict($dataset);
```

### Persistence

`Pml\Lib\ModelStore` is the generic persistence engine.

```php
ModelStore::save($model, 'saved_models/my_model');
$model = ModelStore::load('saved_models/my_model');
```

Internally, it writes:

- `config.json` — PHP configuration and hyperparameters
- `state.safetensors` — tensor weights and state in SafeTensors format

`Pipeline` and `Sequential` use the same format.

## Example classes list

The repository includes the following estimator families:

- `Pml\Estimators\Classifiers\GBDTClassifier`, `RandomForestClassifier`, `LogisticRegression`, `KNNClassifier`, `SVC`, `SoftmaxClassifier`, `VotingClassifier`
- `Pml\Estimators\Regression\GBDTRegressor`, `LinearRegression`, `Ridge`, `Lasso`, `MLPRegressor`, `SVR`
- `Pml\Estimators\Clusterers\KMeans`, `DBSCAN`, `GaussianMixture`
- `Pml\Estimators\AnomalyDetectors\IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`

These implementations all follow the same `Learner` / `Estimator` contract.

## Common patterns

- Use `Dataset::materialize()` before calling `train()`.
- Use `Pipeline` to avoid leaking transformation state between training and inference.
- Use `Persistable` models or `Pipeline::save()` for checkpointing.

## Common mistakes

- Calling `predict()` on an untrained model.
- Saving a model with PHP `serialize()` instead of the built-in SafeTensors-based persistence.
- Building a pipeline with transformers that require ETL mode and then passing a tensor-only dataset to `predict()`.
