---
layout: default
title: Production Example
---

# Production Example

This page covers the production inference pattern with persisted model state and tensor-backed weights.

## Production flow

```text
Training -> save pipeline -> deploy inference server -> load pipeline -> predict
```

## Example

```php
use Pml\Pipeline;

$pipeline = Pipeline::load('saved_pipeline');
$dataset = Pml\Dataset::fromCSV('datasets/housing/serve.csv', labelColumn: 0);
$predictions = $pipeline->predict($dataset);
```

## Internals

- `Pipeline::load()` reconstructs transformer metadata and tensor weights.
- `SafeTensorsIO::load()` supplies native tensor buffers without PHP serialization.
- `predict()` applies the same preprocessing path used in training.

## Performance notes

- Production inference should keep the input dataset in Tensor mode.
- Reuse loaded models across requests to avoid repeated disk I/O.
- Ensure all tensors are contiguous before inference for best throughput.
