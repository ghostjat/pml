<?php
declare(strict_types=1);
/**
 * CUSTOMER SEGMENTATION — RFM + KMeans
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Group customers into behavioural segments for targeted
 *            marketing: VIP, at-risk, new, dormant.
 * Model    : KMeans on RFM (Recency, Frequency, Monetary) features.
 * Business : Personalised campaigns outperform batch-and-blast by
 *            3–5× on conversion rate. Segmentation is the foundation
 *            of every retention and upsell strategy.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Tensor;
use Pml\Estimators\Clusterers\KMeans;
use Pml\Transformers\StandardScaler;
use Pml\Metrics\Clustering\SilhouetteScore;

section('Customer Segmentation — RFM KMeans');

// ── 1. Generate RFM dataset ───────────────────────────────────────────────────
// Recency  : days since last purchase (lower = better)
// Frequency: number of orders in last 12 months
// Monetary : total spend in last 12 months ($)

mt_srand(55);
$rng  = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

$rows = [];
$n    = 5000;

// Four archetypes with noise
$archetypes = [
    // [recency, frequency, monetary]  — Champions
    [7,   24, 2800],
    // Loyal
    [20,  14, 1200],
    // At-risk (was loyal, going cold)
    [90,  8,  900],
    // Hibernating / dormant
    [300, 2,  150],
];

for ($i = 0; $i < $n; $i++) {
    $arch   = $archetypes[mt_rand(0, 3)];
    $rows[] = [
        max(1.0, $arch[0] + $rng(-$arch[0] * 0.4, $arch[0] * 0.6)),
        max(1.0, $arch[1] + $rng(-$arch[1] * 0.4, $arch[1] * 0.6)),
        max(10.0, $arch[2] + $rng(-$arch[2] * 0.4, $arch[2] * 0.6)),
    ];
}

$raw     = Dataset::fromArray($rows);

// ── 2. Normalise before clustering ───────────────────────────────────────────
$scaler  = new StandardScaler();
$scaler->fit($raw);
$scaled  = $scaler->transform($raw);

// ── 3. KMeans — 4 segments ───────────────────────────────────────────────────
section('Clustering (k=4)');
$t0 = microtime(true);

$model = new KMeans(k: 4, maxIter: 500);
$model->train($scaled);

metric('Training time', elapsed($t0));

$assignments = $model->predict($scaled);

// Silhouette score measures cluster quality (−1 to +1, higher is better)
$sil = (new SilhouetteScore())->score($assignments, $scaled->samples());
metric('Silhouette Score', $sil);

// ── 4. Profile each cluster ───────────────────────────────────────────────────
section('Cluster Profiles');

$clusterIds = $assignments->toFlatArray();
$clusters   = [];

foreach ($rows as $idx => $row) {
    $c = (int)$clusterIds[$idx];
    $clusters[$c][] = $row;
}

$labels = [];
ksort($clusters);

foreach ($clusters as $cid => $members) {
    $n   = count($members);
    $avgR = array_sum(array_column($members, 0)) / $n;
    $avgF = array_sum(array_column($members, 1)) / $n;
    $avgM = array_sum(array_column($members, 2)) / $n;

    // Assign a business label based on RFM profile
    $label = match(true) {
        $avgR < 30  && $avgF > 15 => 'Champions',
        $avgR < 60  && $avgF > 8  => 'Loyal Customers',
        $avgR < 150 && $avgF > 4  => 'At-Risk',
        default                    => 'Dormant / Lost',
    };
    $labels[$cid] = $label;

    printf("  Cluster %d — %-20s | n=%4d | Recency=%5.1fd | Freq=%4.1f | Spend=$%6.0f\n",
           $cid, $label, $n, $avgR, $avgF, $avgM);
}

// ── 5. Campaign recommendations ───────────────────────────────────────────────
section('Campaign Recommendations');
$campaigns = [
    'Champions'       => 'Early access to new products, referral programme',
    'Loyal Customers' => 'Loyalty rewards, upsell to premium tier',
    'At-Risk'         => 'Win-back email: "We miss you" + 20% discount',
    'Dormant / Lost'  => 'Reactivation SMS or suppress from budget',
];
foreach ($campaigns as $seg => $action) {
    printf("  %-22s → %s\n", $seg, $action);
}

// ── 6. Score new customers ────────────────────────────────────────────────────
section('Scoring New Customers');
$newCustomers = Dataset::fromArray([
    [5, 28, 3200],   // should be Champion
    [180, 3, 200],   // should be Dormant
]);
$newScaled  = $scaler->transform($newCustomers);
$newClusters = $model->predict($newScaled)->toFlatArray();

foreach ($newClusters as $i => $cid) {
    printf("  Customer %d → Cluster %d (%s)\n", $i + 1, (int)$cid, $labels[(int)$cid] ?? 'Unknown');
}

echo "\n✓ Done\n";
