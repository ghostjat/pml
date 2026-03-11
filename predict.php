<?php

require __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Datasets\Unlabeled;

echo "Loading model...\n";
$estimator = PersistentModel::load(new Filesystem('model/student_performance.rbx'));

// We must provide exactly 19 features matching the exact order of the CSV (minus Exam_Score)
$newStudents = [
    [
        12, 95, 'High', 'Yes', 'No', 8, 88, 'High', 'Yes', 
        2, 'Medium', 'Good', 'Public', 'Positive', 'Moderate', 
        'No', 'College', 5, 'Female' // <-- ADDED 19th Feature (Gender)
    ],
    [
        3, 60, 'Low', 'No', 'No', 5, 55, 'Low', 'Yes', 
        0, 'Low', 'Average', 'Public', 'Negative', 'Low', 
        'Yes', 'High School', 10, 'Male' // <-- ADDED 19th Feature (Gender)
    ],
];

$dataset = new Unlabeled($newStudents);

$predictions = $estimator->predict($dataset);

echo "Predicted Exam Score for Student 1: " . round($predictions[0], 2) . "\n";
echo "Predicted Exam Score for Student 2: " . round($predictions[1], 2) . "\n";