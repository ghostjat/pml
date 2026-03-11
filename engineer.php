<?php

require __DIR__.'/vendor/autoload.php';

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Strategies\KMostFrequent;

echo "Loading Dataset...\n";
$dataset = Labeled::fromIterator(new CSV('data/StudentPerformanceFactors.csv',true));

echo "Original dataset size: " . $dataset->numSamples() . " rows.\n";

$dataset->apply(new MissingDataImputer(null, new KMostFrequent(),''));

echo "Cleaned dataset size: " . $dataset->numSamples() . " rows.\n";

// 3. Type Casting
// Now that all blanks are replaced with real text/numbers, we safely convert to math
$dataset->apply(new NumericStringConverter());
$dataset->transformLabels('floatval');

echo "Cleaned dataset size (No rows dropped!): " . $dataset->numSamples() . " rows.\n";
echo "Generating strictly typed math profile...\n";

// 4. Describe the clean data
$report = $dataset->describe();

$mathColumns = [
    'Teacher_Quality' => $report[11], // Notice the "" empty string is completely gone now!
    'Previous_Scores' => $report[6]
];

echo json_encode($mathColumns, JSON_PRETTY_PRINT) . "\n";