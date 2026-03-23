<?php

declare(strict_types=1);

/**
 * examples/djia_forecasting.php
 * * Masterclass: DJIA Stock Time Series Forecasting using Pml\Classic.
 * * QUANTITATIVE NOTE: STATIONARITY
 * We do not predict raw stock prices (e.g., predicting $150.50). Financial markets 
 * are non-stationary (they drift upwards over decades). Tree-based models like XGBoost 
 * cannot extrapolate outside their training range. If AAPL never hit $200 in 2010-2018, 
 * the model will never predict $200 for 2022.
 * * Instead, we predict Percentage Returns (e.g., +1.5%). Returns are "Stationary"—they 
 * hover around a constant mean (0%) with a consistent variance, regardless of the decade.
 * This allows the model to find underlying mathematical patterns rather than memorizing prices.
 */

require_once __DIR__ . '/../vendor/autoload.php';

use Pml\Classic\Datasets\DataLoader;
use Pml\Classic\Preprocess\StandardScaler;
use Pml\Classic\Ensemble\XGBRegressor;
use Pml\Classic\Pipeline\Pipeline;
use Pml\Classic\Metrics\Metrics;
use Pml\Tensor;

function banner(string $title, string $phase): void
{
    $line = str_repeat('═', 70);
    echo "\n{$line}\n  {$phase}: {$title}\n{$line}\n";
}



banner("DJIA Stock Forecasting with Engineered Financial Features", "Phase 1");

// 1. Load the raw dataset
// CSV Format: date, open, high, low, close, volume, name
$csvPath = $argv[1] ?? __DIR__ . '/../datasets/stocks/APPL.csv';

if (!file_exists($csvPath)) {
    die("Error: Dataset not found at $csvPath. Please ensure the DJIA data is present.\n");
}

$dataset = DataLoader::load_csv($csvPath, header: true);
$raw_data = $dataset['data'];

$engineered_X = [];
$target_y = [];

$window_size = 5; // Lookback window for lagged returns
$sma_period = 14; // 14-day Simple Moving Average

echo "⚙️  Calculating Returns, Volatility, SMAs, and Momentum...\n";

// 2. Feature Engineering Loop
// Start at the 14th day so we have enough historical data to calculate the first SMA.
// End at count - 1 so we have a "Tomorrow" to predict.
for ($i = $sma_period; $i < count($raw_data) - 1; $i++) {
    
    // Extract core price points
    $today_close = (float) $raw_data[$i][4];
    $today_high  = (float) $raw_data[$i][2];
    $today_low   = (float) $raw_data[$i][3];
    $today_vol   = (float) $raw_data[$i][5];
    
    $yesterday_close = (float) $raw_data[$i-1][4];
    $yesterday_vol   = (float) $raw_data[$i-1][5];
    
    $tomorrow_close  = (float) $raw_data[$i+1][4];

    // Protect against zero division in dirty data
    if ($today_close <= 0 || $yesterday_close <= 0 || $yesterday_vol <= 0) {
        continue; 
    }

    // ── TARGET (y) ─────────────────────────────────────────────────────────
    // Tomorrow's percentage return
    $tomorrow_return = ($tomorrow_close - $today_close) / $today_close;
    
    // ── FEATURE 1: Intraday Volatility ─────────────────────────────────────
    $intraday_volatility = ($today_high - $today_low) / $today_close;

    // ── FEATURE 2: 14-Day SMA Ratio ────────────────────────────────────────
    // Identifies if the stock is overbought (>1) or oversold (<1) relative to its recent average
    $sum_14 = 0.0;
    for ($j = 0; $j < $sma_period; $j++) {
        $sum_14 += (float) $raw_data[$i - $j][4];
    }
    $sma_14 = $sum_14 / $sma_period;
    $sma_ratio = $today_close / $sma_14;

    // ── FEATURE 3: Volume Momentum ─────────────────────────────────────────
    // Surges in volume often precede sharp price movements
    $volume_change = ($today_vol - $yesterday_vol) / $yesterday_vol;

    // ── FEATURE 4: Lagged Returns (Sliding Window) ─────────────────────────
    $lagged_returns = [];
    for ($lag = 0; $lag < $window_size; $lag++) {
        $lag_today = (float) $raw_data[$i - $lag][4];
        $lag_yest  = (float) $raw_data[$i - $lag - 1][4];
        
        $lagged_returns[] = $lag_yest > 0 ? ($lag_today - $lag_yest) / $lag_yest : 0.0;
    }

    // Assemble the row
    $row_features = [
        $intraday_volatility,
        $sma_ratio,
        $volume_change
    ];
    $row_features = array_merge($row_features, $lagged_returns);

    $engineered_X[] = $row_features;
    $target_y[] = $tomorrow_return;
}

$num_samples = count($engineered_X);
$num_features = count($engineered_X[0]);
echo "✓  Engineered Matrix: [$num_samples rows, $num_features features]\n";


banner("Building Tensors and Sequential Train/Test Split", "Phase 2");

// CRITICAL: Use explicit shape arrays to build the Tensors from 1D/2D flat data
$X_tensor = Tensor::fromArray($engineered_X, [$num_samples, $num_features]);
$y_tensor = Tensor::fromArray($target_y, [$num_samples]);

// STRICT RULE: Never shuffle Time Series Data. 
// Train on the past (First 80%), Test on the future (Last 20%).
$split_idx = (int) ($num_samples * 0.8);
$test_size = $num_samples - $split_idx;

echo "🔪 Sequential Split -> Train: $split_idx days | Test: $test_size days\n";

// Slice the tensors sequentially
$X_train = $X_tensor->slice(0, $split_idx);
$y_train = $y_tensor->slice(0, $split_idx);

$X_test = $X_tensor->slice($split_idx, $test_size);
$y_test = $y_tensor->slice($split_idx, $test_size);

// Build the Pipeline
// We scale the moving averages and volume, then apply a constrained XGBoost.
$pipeline = new Pipeline([
    ['scaler', new StandardScaler()],
    ['xgb', new XGBRegressor(
        n_estimators: 150, 
        max_depth: 4,         // Shallow depth prevents overfitting high-noise market data
        learning_rate: 0.05,  // Slow learning rate for a smoother convergence
        subsample: 0.8        // Drop 20% of data per tree to generalize better
    )]
]);

echo "🚀 Fitting Pipeline (StandardScaler -> XGBRegressor)...\n";
$pipeline->fit($X_train, $y_train);


banner("Evaluating Model Performance with MAE, RMSE, and Directional Accuracy", "Phase 3");

$predictions = $pipeline->predict($X_test);

$mae = Metrics::mean_absolute_error($y_test, $predictions);
$mse = Metrics::mean_squared_error($y_test, $predictions);
$rmse = sqrt($mse);

// THE QUANT METRIC: Directional Accuracy
// In quantitative trading, predicting the exact return amount (e.g., 1.52%) is nearly impossible.
// However, predicting the DIRECTION (Positive or Negative) provides a statistical edge for trading.
$correct_direction = 0;

for ($i = 0; $i < $test_size; $i++) {
    $actual = $y_test->buffer[$i];
    $predicted = $predictions->buffer[$i];
    
    // Check if the signs match (both > 0 or both < 0)
    // We treat 0.0 as a missed direction for strictness
    if (($actual > 0 && $predicted > 0) || ($actual < 0 && $predicted < 0)) {
        $correct_direction++;
    }
}

$directional_accuracy = ($correct_direction / $test_size) * 100;

echo sprintf("  Mean Absolute Error (Returns) : %.5f\n", $mae);
echo sprintf("  Root Mean Squared (RMSE)      : %.5f\n", $rmse);
echo sprintf("  Directional Accuracy          : %.2f%%\n", $directional_accuracy);

echo "\n──────────────────────────────────────────────────────────────────────\n";
if ($directional_accuracy > 52.0) {
    echo " [SUCCESS] Model shows a statistically significant market edge.\n";
    echo " In high-frequency finance, >52% directional accuracy is profitable.\n";
} else {
    echo " [WARNING] Model performs like a coin flip (or worse).\n";
    echo " Consider adding Macro-economic features or increasing the window size.\n";
}
echo "──────────────────────────────────────────────────────────────────────\n";