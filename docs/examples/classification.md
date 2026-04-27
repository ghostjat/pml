# Classification Example

This example demonstrates training a classifier on a small dataset using `GBDTClassifier`.

```php
<?php
require 'vendor/autoload.php';

use Pml\Dataset;
use Pml\Estimators\Classifiers\GBDTClassifier;

$dataset = Dataset::fromCSV('datasets/titanic/train.csv', labelColumn: 'survived', hasHeader: true)
    ->dropNans()
    ->materialize(labelCol: 'survived');

$model = new GBDTClassifier(
    nEstimators: 100,
    maxDepth: 5,
    learningRate: 0.1,
    numBins: 128,
);

$model->train($dataset);

$predictions = $model->predict($dataset);
$proba = $model->predictProba($dataset);

printf("Predicted labels: %s\n", json_encode($predictions->toFlatArray()));
printf("Probabilities shape: %s\n", json_encode($proba->shape()));
```

## Notes

- Use `dropNans()` to remove missing records before materializing.
- Call `materialize(labelCol: ...)` to convert the dataset into numeric feature and label tensors.
- `predictProba()` is available on probabilistic classifiers.
