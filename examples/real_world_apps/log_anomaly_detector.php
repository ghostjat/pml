<?php
declare(strict_types=1);
/**
 * AI LOG ANOMALY DETECTOR — AIOps / SRE Tooling
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Detect anomalous server log patterns (error spikes,
 *            latency regressions, traffic anomalies) without labels.
 * Method   : Extract numeric features from log lines →
 *            RobustZScore (univariate, fast) + IsolationForest
 *            (multivariate, catches interaction anomalies).
 * Business : Mean time to detect (MTTD) for production incidents
 *            is 197 minutes without automation (IBM 2022).
 *            AI-driven log analysis cuts it to under 5 minutes.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Estimators\AnomalyDetectors\IsolationForest;
use Pml\Estimators\AnomalyDetectors\RobustZScore;

section('AI Log Anomaly Detector — AIOps');

// ── 1. Log feature extraction ──────────────────────────────────────────────────
// Each log "window" (1 minute) produces a feature vector:
//   error_rate, warn_rate, avg_latency_ms, p99_latency_ms,
//   req_per_sec, db_query_count, cache_miss_rate,
//   cpu_pct, memory_pct, active_connections

mt_srand(23);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

function logWindow(callable $rng, string $type = 'normal'): array
{
    return match($type) {
        'normal'      => [
            $rng(0.001, 0.01), $rng(0.01, 0.05), $rng(80, 200),  $rng(200, 500),
            $rng(100, 400),    $rng(50, 200),     $rng(0.05, 0.2),
            $rng(30, 65),      $rng(40, 70),      $rng(50, 150),
        ],
        'error_spike' => [
            $rng(0.15, 0.40), $rng(0.1, 0.3),  $rng(500, 2000), $rng(2000, 8000),
            $rng(50, 150),    $rng(200, 500),   $rng(0.3, 0.8),
            $rng(70, 95),     $rng(70, 95),     $rng(200, 500),
        ],
        'traffic_flood' => [
            $rng(0.005, 0.02), $rng(0.02, 0.08), $rng(300, 800), $rng(800, 2000),
            $rng(2000, 8000),  $rng(1000, 3000), $rng(0.4, 0.8),
            $rng(85, 100),     $rng(80, 98),     $rng(800, 2000),
        ],
        'memory_leak'   => [
            $rng(0.02, 0.10), $rng(0.05, 0.15), $rng(200, 600), $rng(600, 2000),
            $rng(80, 200),    $rng(100, 300),   $rng(0.2, 0.5),
            $rng(50, 80),     $rng(88, 100),    $rng(100, 300),
        ],
        'db_slowdown'   => [
            $rng(0.02, 0.08), $rng(0.03, 0.10), $rng(500, 3000), $rng(3000, 10000),
            $rng(60, 150),    $rng(500, 2000),  $rng(0.5, 0.9),
            $rng(40, 70),     $rng(60, 80),     $rng(80, 200),
        ],
    };
}

// Training data: 2 weeks of normal 1-minute windows
$normalWindows = [];
for ($i = 0; $i < 20160; $i++) {   // 14 days × 24h × 60min
    $normalWindows[] = logWindow($rng, 'normal');
}
$trainDs = Dataset::fromArray($normalWindows);
metric('Training windows (2 weeks)', count($normalWindows));

// ── 2. Train detectors ────────────────────────────────────────────────────────
section('Training Anomaly Detectors');
$t0 = microtime(true);

$isoForest = new IsolationForest(nEstimators: 100, sampleSize: 256, contamination: 0.01);
$isoForest->train($trainDs);

$robustZ = new RobustZScore(threshold: 3.5);
$robustZ->train($trainDs);

metric('Training time', elapsed($t0));

// ── 3. Simulate live monitoring feed ─────────────────────────────────────────
section('Live Monitoring — 24h Simulation');

$incidents = [
    ['time' => '02:15', 'type' => 'normal',       'expected' => false],
    ['time' => '09:00', 'type' => 'normal',       'expected' => false],
    ['time' => '11:30', 'type' => 'db_slowdown',  'expected' => true],
    ['time' => '13:00', 'type' => 'normal',       'expected' => false],
    ['time' => '14:45', 'type' => 'traffic_flood','expected' => true],
    ['time' => '16:00', 'type' => 'normal',       'expected' => false],
    ['time' => '19:22', 'type' => 'error_spike',  'expected' => true],
    ['time' => '22:10', 'type' => 'memory_leak',  'expected' => true],
    ['time' => '23:55', 'type' => 'normal',       'expected' => false],
];

printf("\n  %-8s | %-16s | %-10s | %-10s | %s\n",
       'Time', 'Event', 'IsoForest', 'RobustZ', 'Alert');
printf("  %s\n", str_repeat('-', 70));

$detected = $falsePos = 0;

foreach ($incidents as $inc) {
    $window = logWindow($rng, $inc['type']);
    $ds     = Dataset::fromArray([$window]);

    $isoScore = $isoForest->predict($ds)->toFlatArray()[0] ?? 0.0;
    $zsScore  = $robustZ->predict($ds)->toFlatArray()[0] ?? 0.0;

    $isAnomaly = $isoScore > 0.5 || $zsScore > 0.5;
    $isActualAnomaly = $inc['expected'];

    if ($isAnomaly && $isActualAnomaly)  $detected++;
    if ($isAnomaly && !$isActualAnomaly) $falsePos++;

    $alertIcon = match(true) {
        $isAnomaly && $isActualAnomaly  => '🚨 INCIDENT',
        $isAnomaly && !$isActualAnomaly => '⚠️  FALSE POS',
        !$isAnomaly && $isActualAnomaly => '❌ MISSED',
        default                          => '✅ OK',
    };

    printf("  %-8s | %-16s | %-10.4f | %-10.4f | %s\n",
           $inc['time'], $inc['type'], $isoScore, $zsScore, $alertIcon);
}

$trueIncidents = count(array_filter($incidents, fn($i) => $i['expected']));
printf("\n  Detection rate : %d/%d incidents\n", $detected, $trueIncidents);
printf("  False positives: %d\n", $falsePos);

// ── 4. Alert integration ──────────────────────────────────────────────────────
section('Alert Integration Pattern');
echo <<<TXT
  // In production: call this per 1-minute log aggregate
  function checkWindow(array \$features, Pipeline \$detector): void {
      \$score = \$detector->predict(Dataset::fromArray([\$features]))->toFlatArray()[0];
      if (\$score > 0.6) {
          // POST to PagerDuty / OpsGenie / Slack
          \$this->alerting->fire(severity: 'P1', score: \$score, features: \$features);
      }
  }
TXT;

echo "\n✓ Done\n";
