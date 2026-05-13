<?php
declare(strict_types=1);
/**
 * SERVER INTRUSION DETECTION — Network Flow Anomaly Detection
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Identify malicious network connections (port scans,
 *            lateral movement, C2 beaconing) from flow features.
 * Model    : IsolationForest — no labels required, trains on normal
 *            traffic only, scores anomalous flows.
 * Business : The average cost of a data breach is $4.5 M (IBM 2023).
 *            Detecting intrusions within minutes vs. days limits
 *            blast radius and regulatory exposure.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Estimators\AnomalyDetectors\IsolationForest;
use Pml\Metrics\Classification\RocAuc;
use Pml\Metrics\Classification\Precision;
use Pml\Metrics\Classification\Recall;

section('Network Intrusion Detection — Isolation Forest');

// ── 1. Network flow features ──────────────────────────────────────────────────
// duration_s, bytes_sent, bytes_recv, packets_sent, packets_recv,
// unique_ports, avg_packet_size, inter_arrival_ms, flags_syn_ratio, ttl_variance

mt_srand(42);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

$normalRows    = [];
$anomalyRows   = [];

// Normal web / app traffic
for ($i = 0; $i < 4500; $i++) {
    $normalRows[] = [
        $rng(0.1, 60),       // duration
        $rng(500, 50000),    // bytes sent
        $rng(1000, 200000),  // bytes recv (download > upload)
        $rng(5, 200),        // packets sent
        $rng(10, 500),       // packets recv
        $rng(1, 3),          // unique ports (http=80, https=443)
        $rng(200, 1400),     // avg packet size
        $rng(20, 500),       // inter-arrival ms
        $rng(0, 0.05),       // low SYN ratio (established sessions)
        $rng(0, 5),          // low TTL variance
    ];
}

// Anomalous traffic: port scan, C2 beacon, data exfil
$attackTypes = ['port_scan', 'c2_beacon', 'exfil'];

for ($i = 0; $i < 500; $i++) {
    $type = $attackTypes[$i % 3];
    $anomalyRows[] = match($type) {
        'port_scan' => [
            $rng(0.001, 0.5), $rng(40, 200), $rng(0, 50),
            $rng(50, 500), $rng(0, 50),
            $rng(200, 65535),  // scanning many ports
            $rng(40, 80),
            $rng(1, 10),       // very fast inter-arrival
            $rng(0.8, 1.0),    // high SYN ratio (no handshakes completing)
            $rng(20, 50),
        ],
        'c2_beacon' => [
            $rng(0.09, 0.11), $rng(64, 128), $rng(64, 128),  // tiny, periodic
            $rng(1, 3), $rng(1, 3),
            $rng(1, 2),
            $rng(64, 128),
            $rng(298, 302),    // suspiciously regular timing
            $rng(0, 0.1),
            $rng(0, 1),
        ],
        'exfil' => [
            $rng(60, 3600),    // long session
            $rng(500000, 2000000),  // massive upload
            $rng(100, 500),
            $rng(5000, 20000),
            $rng(5, 20),
            $rng(1, 2),
            $rng(1400, 1500),  // jumbo frames
            $rng(1, 5),
            $rng(0, 0.02),
            $rng(0, 2),
        ],
    };
}

// Build labelled test set (model trains ONLY on normal)
$trainDs = Dataset::fromArray($normalRows);
$allRows = array_merge($normalRows, $anomalyRows);
$allLbls = array_merge(
    array_fill(0, count($normalRows), 0.0),
    array_fill(0, count($anomalyRows), 1.0)
);

// Shuffle test set
$idx = range(0, count($allRows) - 1);
shuffle($idx);
$testRows = array_map(fn($i) => $allRows[$i], $idx);
$testLbls = array_map(fn($i) => $allLbls[$i], $idx);
$testDs   = Dataset::fromArray($testRows, $testLbls);

metric('Normal flows (train)', count($normalRows));
metric('Anomalous flows',      count($anomalyRows));

// ── 2. Train on normal traffic only ──────────────────────────────────────────
section('Training (unsupervised — normal traffic only)');
$t0 = microtime(true);

$model = new IsolationForest(nEstimators: 200, sampleSize: 256, contamination: 0.1);
$model->train($trainDs);

metric('Training time', elapsed($t0));

// ── 3. Evaluate on labelled test set ─────────────────────────────────────────
section('Evaluation');
$t1   = microtime(true);
$pred = $model->predict($testDs);
metric('Inference time', elapsed($t1));

$labels = $testDs->labels();
metric('ROC-AUC',   (new RocAuc())->score($pred, $labels));
metric('Precision', (new Precision())->score($pred, $labels));
metric('Recall',    (new Recall())->score($pred, $labels));

// ── 4. Real-time alert simulation ────────────────────────────────────────────
section('Real-Time Alert Simulation');

$liveFlows = [
    'Normal HTTPS session' => [12.0, 8500, 45000, 60, 200, 2, 800, 150, 0.01, 1],
    'Port scan detected'   => [0.002, 60, 0, 200, 0, 1024, 60, 2, 0.99, 35],
    'C2 beacon detected'   => [0.1, 96, 96, 2, 2, 1, 96, 300, 0.02, 0],
    'Data exfiltration'    => [900, 1500000, 200, 12000, 10, 1, 1480, 3, 0.01, 1],
];

foreach ($liveFlows as $desc => $features) {
    $score = $model->predict(Dataset::fromArray([$features]))->toFlatArray()[0] ?? 0.0;
    $alert = $score > 0.5 ? '🚨 ALERT' : '  OK  ';
    printf("  [%s] %-28s anomaly_score=%.3f\n", $alert, $desc, $score);
}

echo "\n✓ Done\n";

/*
 * PRODUCTION NOTES
 * ────────────────
 * • Consume from Kafka network-flow topic at line rate.
 * • Alert pipeline: score > 0.7 → PagerDuty / SIEM integration.
 * • Retrain weekly on the previous week's verified-clean traffic.
 * • Combine with rule-based detection for known-bad signatures.
 */
