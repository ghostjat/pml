<?php
require_once __DIR__ . '/DataFrame.php';

// 1. Generate Mock Data (Price, Volume, Signals)
$csvData = "price,volume,signal\n100.5,10000,1.2\n102.1,12000,0.8\n99.8,8000,1.5\n105.0,20000,-0.5\n101.2,11000,0.1\n";
file_put_contents('data.csv', $csvData);

// 2. Load Zero-Copy DataFrame
$df = DataFrame::fromCSV('data.csv');
echo "Original Data:\n";
$df->head();

// 3. Chain high-performance C operations natively
echo "Applying Z-Score Standardization (AVX2)...\n";
$df->standardize()->head();

// 4. Reload and test Robust Scaler
$df2 = DataFrame::fromCSV('data.csv');
echo "Applying Robust Scaler (O(N) QuickSelect Median/IQR)...\n";
$df2->robustScale()->head();

// 5. MinMax Scaler
$df3 = DataFrame::fromCSV('data.csv');
echo "Applying Min-Max Scaler...\n";
$df3->minMaxScale()->head();

// Memory strictly cleared at script termination using __destruct() chain