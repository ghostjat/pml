---
layout: default
title: Training Lifecycle
---

# Training Lifecycle

Training in PML is a staged workflow: data ingestion, transformation, batching, optimization, and persistence.

## Stage 1: Dataset preparation

- Start with a `Dataset` in Tensor mode.
- Ensure label columns are set and tensors are contiguous.
- Use `Pipeline` to fit preprocessing on training data only.

## Stage 2: Batch execution

- Training loops consume minibatches.
- Batching is best handled at the dataset or pipeline layer.
- Avoid materializing the same batch multiple times.

## Stage 3: Optimization

- Estimators may implement native gradient steps.
- Fused Adam and fused BCE kernels reduce FFI overhead.
- Weight updates should happen in-place when safe.

## Stage 4: Validation and checkpoints

- Validation datasets should be separate and kept in Tensor mode.
- Checkpointing writes configuration and weight tensors separately.
- `Pipeline::save()` persists transformer state in SafeTensors.

## Best practices

- Use deterministic seeds for reproducibility.
- Keep the training dataset contiguous across feature dimensions.
- Avoid Python-style eager loops over rows in PHP.

## Example

```php
$pipeline = new Pml\Pipeline([
    new Pml\Transformers\StandardScaler(),
], new Pml\Estimators\Regression\GBDTRegressor());

$pipeline->train($trainDataset, epochs: 10, batchSize: 64);
```

## When to use

- Use training lifecycle patterns for production experiments.
- Use explicit validation loops for performance regression checks.

## When not to use

- Do not skip tensor-level validation for mixed-type datasets.
- Do not serialize live FFI pointers; persist state with SafeTensors and JSON metadata.
