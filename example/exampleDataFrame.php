<?php
require_once __DIR__ . '/DataFrame.php';

// 1. Setup mock data
$csvData = "date,open,close,volume\n2023-01-01,100.0,105.0,1000\n2023-01-02,105.0,102.0,1200\n2023-01-03,102.0,110.0,1500\n2023-01-04,110.0,108.0,800\n";
file_put_contents('data.csv', $csvData);

echo "Starting High-Performance FFI Engine...\n\n";

// 2. Load directly into C memory space (zero PHP array allocations)
$df = DataFrame::fromCSV("data.csv");

// 3. Vectorized Memory-safe C-filtering 
// This spawns a new CData struct with tight-loop copied elements,
// old DataFrame is preserved and automatically cleaned by GC.
$df = $df->filter("close > open");

// 4. Custom PHP Logic over C Memory
// Here we pre-load C pointers and loop, directly injecting closure returns into a C array buffer.
$df->addColumn("returns", function($row) {
    return ($row['close'] - $row['open']) / $row['open'];
});

// 5. C-Side statistical operation
$mean = $df->column("returns")->mean();

echo "Average return on up-days: " . number_format($mean * 100, 2) . "%\n";

// 6. Memory cleans itself up perfectly upon script exit or variable unset.