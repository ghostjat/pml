---
layout: default
title: Classification Estimators
---

# Classification Estimators

Classification models map tensor features to discrete labels. They are optimized for stable decision boundaries and efficient inference.

## Model structure

- Input tensors: `[N × D]` float32 arrays.
- Output tensors: class scores or label indices.
- Internally, classification reduces to margin-based or probabilistic objectives.

## API usage

```php
$dataset = Dataset::fromCSV('datasets/titanic/train.csv', labelColumn: 0);
$model = new Pml\Estimators\Classification\LogisticRegression();
$model->train($dataset);
$predictions = $model->predict($dataset);
```

## Internal behavior

- Logistic or softmax classifiers use fused native kernels where possible.
- Predictions are computed in a single pass over the feature tensor.
- Post-processing maps logits to label indices.

## Performance implications

- Cross-entropy and softmax require stable accumulators.
- Use batch sizes that fit the available L2 cache.
- Avoid repeated label encoding in the hot path.

## When to use

- Use classification estimators for categorical targets.
- Prefer them when inference latency matters.

## When not to use

- Do not use classification estimators on regression targets.
- Do not use them when label imbalance mandates a specialized objective unless the estimator explicitly supports it.
