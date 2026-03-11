<?php

require __DIR__.'/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\PersistentModel;

echo "Loading data ...\n";
$dataset = Labeled::fromIterator(new CSV('data/StudentPerformanceFactors.csv',true));
$dataset->transformLabels('floatval');

echo "Loaded {$dataset->numSamples()} samples \n";

echo "Building the pipeline...\n";

$estimator = new Pipeline([new NumericStringConverter()], new RegressionTree(10));

echo "Validating model with 5-Fold Cross Validation...\n";

$validator = new KFold(5);
$score = $validator->test($estimator, $dataset, new RSquared());

echo "Model R-Squared Score: " . round($score, 4) . "\n";

echo "Training the final model...\n";

$model = new PersistentModel($estimator, new Filesystem('model/student_performance.rbx'));

$model->train($dataset);
$model->save();

echo "Model successfully saved to 'student_performance.rbx'!\n";