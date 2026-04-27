# Custom Pipeline Example

Build a reusable `Pipeline` with transformers and a final estimator.

```php
<?php
require 'vendor/autoload.php';

use Pml\Dataset;
use Pml\Pipeline;
use Pml\Transformers\WordCountVectorizer;
use Pml\Estimators\Classifiers\GBDTClassifier;

$dataset = Dataset::fromCSV('datasets/sentiment/train.csv', labelColumn: 'label', hasHeader: true)
    ->dropNans()
    ->materialize(labelCol: 'label');

$vectorizer = new WordCountVectorizer(minDf: 2, maxFeatures: 2000);
$model = new GBDTClassifier(nEstimators: 80, maxDepth: 5);

$pipeline = new Pipeline([
    $vectorizer,
], $model);

$pipeline->train($dataset);

$inference = Dataset::fromArray(
    [
        ['I love this product', 'positive'],
        ['Terrible support experience', 'negative'],
    ],
    [1, 0]
)->materialize(labelCol: 1);

$predictions = $pipeline->predict($inference);
print_r($predictions->toFlatArray());
```

## Notes

- `Pipeline::train()` fits transformers and then trains the estimator.
- `Pipeline::predict()` applies fitted transformers before inference.
- Save and reload pipelines with `Pipeline::save($dir)` and `Pipeline::load($dir)`.
