<?php
declare(strict_types=1);
/**
 * IoT SENSOR ANOMALY DETECTION — Predictive Maintenance
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Detect failing industrial equipment from vibration,
 *            temperature and pressure sensor readings.
 * Model    : RobustZScore for fast streaming detection +
 *            IsolationForest for multi-variate anomalies.
 * Business : Unplanned downtime on a factory line costs $10,000–
 *            $50,000 per hour. Detecting bearing failure 2 weeks
 *            before breakdown enables planned maintenance.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Estimators\AnomalyDetectors\IsolationForest;
use Pml\Estimators\AnomalyDetectors\RobustZScore;

section('IoT Predictive Maintenance — Anomaly Detection');

// ── 1. Sensor stream ──────────────────────────────────────────────────────────
// Features: vibration_rms_g, temp_bearing_c, temp_motor_c,
//           pressure_bar, rpm, current_a, oil_quality_index

mt_srand(17);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

$normalRows  = [];
$faultRows   = [];

// 10,000 normal operating readings
for ($i = 0; $i < 10000; $i++) {
    $rpm = $rng(2900, 3100);
    $normalRows[] = [
        $rng(0.1, 0.8),    // vibration (g)
        $rng(55, 75),      // bearing temp (°C)
        $rng(60, 80),      // motor temp (°C)
        $rng(4.5, 5.5),    // pressure (bar)
        $rpm,
        $rng(10, 14),      // current (A)
        $rng(80, 100),     // oil quality index
    ];
}

// Simulated fault progression over 200 readings
for ($i = 0; $i < 200; $i++) {
    $t = $i / 200;  // fault severity 0→1
    $faultRows[] = [
        $rng(0.1, 0.8) + $t * $rng(2, 8),     // growing vibration
        $rng(55, 75) + $t * $rng(20, 40),      // heating bearing
        $rng(60, 80) + $t * $rng(5, 15),
        $rng(4.5, 5.5) - $t * $rng(0.5, 1.5), // dropping pressure
        $rng(2800, 3100) - $t * $rng(100, 300),
        $rng(10, 14) + $t * $rng(2, 6),        // increasing current draw
        $rng(80, 100) - $t * $rng(30, 60),     // oil degrading
    ];
}

$trainDs = Dataset::fromArray($normalRows);
metric('Normal readings (train)', count($normalRows));
metric('Fault-progression readings', count($faultRows));

// ── 2. Train both detectors ───────────────────────────────────────────────────
section('Training Detectors');
$t0 = microtime(true);

$isoForest = new IsolationForest(nEstimators: 100, sampleSize: 256, contamination: 0.02);
$isoForest->train($trainDs);

$robustZ = new RobustZScore(threshold: 3.5);
$robustZ->train($trainDs);

metric('Training time', elapsed($t0));

// ── 3. Score fault progression ────────────────────────────────────────────────
section('Fault Progression Scoring');

printf("  %5s | %12s | %12s | %s\n", 'Step', 'IsoForest', 'RobustZScore', 'Status');
printf("  %s\n", str_repeat('-', 55));

$milestones = [0, 40, 80, 120, 160, 199];
foreach ($milestones as $step) {
    $sample = Dataset::fromArray([$faultRows[$step]]);
    $isoScore = $isoForest->predict($sample)->toFlatArray()[0] ?? 0.0;
    $zsScore  = $robustZ->predict($sample)->toFlatArray()[0] ?? 0.0;

    $pct    = round($step / 200 * 100);
    $status = match(true) {
        $isoScore > 0.7 || $zsScore > 0.7 => '🚨 CRITICAL — shutdown now',
        $isoScore > 0.5 || $zsScore > 0.5 => '⚠️  WARNING  — schedule maintenance',
        $isoScore > 0.3 || $zsScore > 0.3 => '📊 WATCH    — monitor closely',
        default                            => '✅ NORMAL',
    };
    printf("  %3d%%  | %12.4f | %12.4f | %s\n", $pct, $isoScore, $zsScore, $status);
}

// ── 4. Streaming simulation ───────────────────────────────────────────────────
section('Streaming Alert Counts');
$faultDs  = Dataset::fromArray($faultRows);
$isoPreds = $isoForest->predict($faultDs)->toFlatArray();
$zsPreds  = $robustZ->predict($faultDs)->toFlatArray();

$isoAlerts = count(array_filter($isoPreds, fn($s) => $s > 0.5));
$zsAlerts  = count(array_filter($zsPreds,  fn($s) => $s > 0.5));

metric('IsolationForest alerts / 200 fault readings', $isoAlerts);
metric('RobustZScore alerts / 200 fault readings',    $zsAlerts);
metric('First ISO alert at step', array_search(true, array_map(fn($s) => $s > 0.5, $isoPreds)));

echo "\n✓ Done\n";
