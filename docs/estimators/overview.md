---
layout: default
title: Estimators Overview
---

# Estimators Overview

Estimators are the system component responsible for learning from tensor data. They expose `train()` and `predict()` as the core contract.

## Concept

- Estimators consume `Dataset` objects in Tensor mode.
- They may be wrapped in a `Pipeline` for feature transformation.
- Training uses native kernels where available and PHP orchestration otherwise.

## Design

- `Pipeline` ensures transformers are fitted on training data only.
- Estimators may implement `TrainableWithOptions` for hyperparameter control.
- Persistence is decoupled: `Persistable` and `ModelStore` are used instead of PHP serialization.

## Data flow

```text
Dataset (Tensor mode)
   ├─ transformers (fit/transform)
   └─ estimator.train()
         └─ native numeric kernels / optimization loop
```

## Core estimator roles

- Regression models optimize a numeric objective.
- Classification models map logits to discrete labels.
- Training controllers manage batching, metrics, and checkpoints.

## When to use

- Use estimators for supervised learning tasks.
- Use `Pipeline` when preprocessing and training should be persisted together.

## When not to use

- Do not execute training on ETL-mode datasets.
- Do not bypass `Pipeline` if your transformer state must be reused in inference.
