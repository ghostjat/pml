<?php

require_once __DIR__ . '/DataFrame.php'; 
require_once __DIR__ . '/Xgboost.php';    

$csvPath = __DIR__ . '/datasets/stocks/ml_ready_features.csv';
$modelPath = __DIR__ . '/models/xgboost_stock_model.bin';

echo "Initiating High-Performance ML Pipeline...\n";

// 1. Define active features (Exclude 'Date', 'Ticker', 'Label')
$features = ['VCP_Consolidation', 'Breakout_8W', 'RS_Rating', 'Stage_2_Filter', 'Vol_Spike_Ratio', 'Vol_Dry_Up', 'ATR_Ratio', 'Log_Return'];

// 2. Load and Standardize using Fast C-Engine
echo "Loading and Standardizing Data...\n";
$df = DataFrame::fromCSV($csvPath);
$df->standardize(); 

// 3. Prepare DMatrix (All ugly C-pointers are hidden inside XGBHelper)
echo "Preparing Zero-Copy Memory Buffers...\n";
$data = XGBHelper::prepareDataset(
    $csvPath, 
    $df, 
    $features, 
    'Label', 
    ['-1' => 0, 'SHORT' => 0, '0' => 1, 'AVOID' => 1, '1' => 2, 'LONG' => 2]
);

// 4. Define Grid Search Parameters
// We will test shallow vs medium trees, and light vs heavy regularization.
// The algorithm will find the exact combination that breaks the 61% threshold.
$paramGrid = [
    'max_depth' => [3, 5, 7],
    'eta' => [0.05, 0.1],
    'lambda' => [1.0, 5.0],
    'min_child_weight' => [1, 5]
];

// 5. Run Automated GridSearch
$result = GridSearchCV::search($paramGrid, $data['dTrain'], $data['dVal'], $data['yVal'], 300);

$bestBooster = $result['booster'];
$bestAcc = $result['accuracy'];
$bestParams = $result['params'];

// 6. Save and Report
echo "\n==================================================\n";
echo "           XGBOOST OPTIMIZED REPORT\n";
echo "==================================================\n";
echo "Best Val Accuracy : " . number_format($bestAcc * 100, 2) . "%\n";
echo "Optimal Params    : " . json_encode($bestParams) . "\n";

if ($bestAcc >= 0.60) {
    $bestBooster->save($modelPath);
    echo "Model Status      : SAVED TO DISK\n\n";
} else {
    echo "Model Status      : DISCARDED (Below 60% standard)\n\n";
}

echo "[ TOP FEATURES IMPORTANCE ]\n";
$imp = $bestBooster->getFeatureImportance($features);
$rank = 1;
foreach (array_slice($imp, 0, 10, true) as $feature => $weight) {
    printf("%2d. %-20s : %d splits\n", $rank++, $feature, $weight);
}
echo str_repeat("=", 50) . "\n";