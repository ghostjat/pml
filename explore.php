<?php

require __DIR__.'/vendor/autoload.php';

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\NumericStringConverter;

echo "Loading Dataset...\n";
$dataset = Labeled::fromIterator(new CSV('data/StudentPerformanceFactors.csv',true));
$report = $dataset->describe();
echo json_encode($report, JSON_PRETTY_PRINT);