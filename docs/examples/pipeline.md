---
layout: default
title: Pipeline Example
---

# Pipeline Example

A pipeline assembles preprocessing and a final estimator while preserving state for inference.

## Structure

- transformers fit on training data only
- estimator trains on transformed tensors
- pipeline saves both transformer state and estimator state

## Example

```php
use Pml\Dataset;
use Pml\Pipeline;
use Pml\Transformers\StandardScaler;
use Pml\Estimators\Regression\GBDTRegressor;

$dataset = Dataset::fromCSV('datasets/housing/train.csv', labelColumn: 0);
$pipeline = new Pipeline([
    new StandardScaler(),
], new GBDTRegressor());

$pipeline->train($dataset, epochs: 5, batchSize: 64);
$pipeline->save('saved_pipeline');
```

## Internals

- `Pipeline::train()` processes transformers in order.
- `FitTransformable` transformers are fitted and transformed in one pass.
- The final estimator receives a fully transformed `Dataset`.

## Persistence

- `Pipeline::save()` writes `config.json`, transformer SafeTensors, and estimator state.
- During `Pipeline::load()`, tensors are rehydrated from SafeTensors with zero-copy semantics where possible.

## Performance notes

- Avoid transforming the same data twice.
- Fit transformers before splitting the dataset to prevent leakage.
