# Regression Example

Train a regression model with `GBDTRegressor` and predict on the same dataset.

```php
<?php
require 'vendor/autoload.php';

use Pml\Dataset;
use Pml\Estimators\Regression\GBDTRegressor;

$dataset = Dataset::fromArray(
    [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 6.0],
        [3.0, 4.0, 9.0],
    ],
    [6.0, 11.0, 16.0]
);

$model = new GBDTRegressor(
    nEstimators: 50,
    maxDepth: 4,
    learningRate: 0.1,
    numBins: 64,
);

$model->train($dataset);
$predictions = $model->predict($dataset);

print_r($predictions->toFlatArray());
```

## Notes

- `Dataset::fromArray()` creates a numeric dataset from PHP arrays.
- `GBDTRegressor` uses the native C-backed gradient boosting implementation.
- Use `Dataset::split()` or `Dataset::fold()` for train/validation splits.
