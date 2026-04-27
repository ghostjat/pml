---
layout: default
title: Regression Estimators
---

# Regression Estimators

Regression models in PML compute continuous targets from numeric tensor inputs. They are designed for low-latency inference and efficient training.

## Model structure

- Input features are represented as `[N × D]` float32 tensors.
- Targets are `[N]` float32 tensors.
- Regression training optimizes a differentiable loss or tree-based objective.

## Data flow

```text
Tensor samples [N × D]
   ├─ preprocessing / normalization
   └─ estimator.train() → loss / gradient computation
```

## Example API usage

```php
$dataset = Dataset::fromCSV('datasets/housing/train.csv', labelColumn: 0);
$model = new Pml\Estimators\Regression\GBDTRegressor();
$model->train($dataset);
$predictions = $model->predict($dataset);
```

## Internal behavior

- Training may use native tensor operations for matrix algebra.
- Tree-based regressors may use C-backed histograms and split scoring.
- Weights and trees are persisted separately from PHP object structure.

## Performance considerations

- Regression workloads benefit from dense tensors and contiguous layout.
- Use `Dataset::materialize()` before training to avoid repeated dataset conversion.
- Avoid broadcasting large tensors in the training loop unless required.

## When to use

- Use regression estimators for real-valued prediction targets.
- Use tree-based regressors when feature normalization is less important.

## When not to use

- Do not use regression estimators for categorical classification targets.
- Do not use dense regression models on extremely sparse high-dimensional data without prior dimensionality reduction.
