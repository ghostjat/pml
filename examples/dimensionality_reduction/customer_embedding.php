<?php
declare(strict_types=1);
/**
 * CUSTOMER EMBEDDING & VISUALISATION — PCA + t-SNE
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Compress high-dimensional customer behaviour vectors
 *            into 2D for visualisation and 6D for downstream ML.
 * Models   : PCA (fast, linear, production-grade compression) +
 *            t-SNE (nonlinear, for cluster discovery / dashboards).
 * Business : Analysts use 2D customer maps to identify organic
 *            segments, discover anomalous cohorts, and validate
 *            that marketing segments are actually separable.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Tensor;
use Pml\Estimators\Decomposition\PrincipalComponentAnalysis;
use Pml\Transformers\StandardScaler;

section('Customer Embedding — PCA Dimensionality Reduction');

// ── 1. High-dimensional customer behaviour matrix ─────────────────────────────
// 50 features per customer: page_views_by_category (10), purchase_history (10),
//   engagement_scores (10), timing_patterns (10), device/channel (10)

mt_srand(99);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

$nCustomers = 2000;
$nFeatures  = 50;
$rows = [];
$trueSegment = [];   // hidden ground truth for validation

// 4 customer archetypes
$archetypes = [
    // 'power_users'   : high engagement, high purchase, desktop
    array_merge(array_fill(0, 10, 8.0), array_fill(0, 10, 7.0),
                array_fill(0, 10, 9.0), array_fill(0, 10, 5.0), array_fill(0, 10, 1.0)),
    // 'mobile_casual' : medium engagement, low purchase, mobile
    array_merge(array_fill(0, 10, 4.0), array_fill(0, 10, 2.0),
                array_fill(0, 10, 5.0), array_fill(0, 10, 8.0), array_fill(0, 10, 0.0)),
    // 'bargain_hunters': low engagement, purchase on discounts, mixed
    array_merge(array_fill(0, 10, 2.0), array_fill(0, 10, 5.0),
                array_fill(0, 10, 3.0), array_fill(0, 10, 4.0), array_fill(0, 10, 0.5)),
    // 'window_shoppers': high views, zero purchase
    array_merge(array_fill(0, 10, 9.0), array_fill(0, 10, 0.0),
                array_fill(0, 10, 6.0), array_fill(0, 10, 6.0), array_fill(0, 10, 0.3)),
];

for ($i = 0; $i < $nCustomers; $i++) {
    $seg  = mt_rand(0, 3);
    $arch = $archetypes[$seg];
    $row  = array_map(fn($v) => max(0.0, $v + $rng(-2.5, 2.5)), $arch);
    $rows[] = $row;
    $trueSegment[] = $seg;
}

$ds = Dataset::fromArray($rows);
metric('Customers', $nCustomers);
metric('Original dimensions', $nFeatures);

// ── 2. Scale ──────────────────────────────────────────────────────────────────
$scaler = new StandardScaler();
$scaler->fit($ds);
$scaled = $scaler->transform($ds);

// ── 3. PCA — production embedding (6 components) ─────────────────────────────
section('PCA Compression');
$t0 = microtime(true);

$pca6 = new PrincipalComponentAnalysis(nComponents: 6);
$pca6->train($scaled);
$emb6 = $pca6->predict($scaled);

metric('Compressed to', 6 . ' dimensions');
metric('Compression ratio', '8.3×');
metric('PCA time', elapsed($t0));

// ── 4. PCA — 2D for visualisation ─────────────────────────────────────────────
$pca2 = new PrincipalComponentAnalysis(nComponents: 2);
$pca2->train($scaled);
$emb2 = $pca2->predict($scaled);

metric('2D embedding computed', 'YES');

// ── 5. ASCII scatter plot of the 2D embedding ─────────────────────────────────
section('2D Customer Map (ASCII preview)');

$flat2d = $emb2->toFlatArray();
$xs = []; $ys = [];
for ($i = 0; $i < $nCustomers; $i++) {
    $xs[] = $flat2d[$i * 2];
    $ys[] = $flat2d[$i * 2 + 1];
}

$minX = min($xs); $maxX = max($xs);
$minY = min($ys); $maxY = max($ys);
$W = 60; $H = 20;

$grid = array_fill(0, $H, array_fill(0, $W, ' '));
$chars = ['P', 'M', 'B', 'W'];  // Power, Mobile, Bargain, Window

foreach ($xs as $i => $x) {
    $col = (int)(($x - $minX) / ($maxX - $minX + 1e-9) * ($W - 1));
    $row = (int)(($ys[$i] - $minY) / ($maxY - $minY + 1e-9) * ($H - 1));
    $grid[$row][$col] = $chars[$trueSegment[$i]];
}

echo "\n";
foreach ($grid as $row) {
    echo '  |' . implode('', $row) . "|\n";
}
echo "  Legend: P=PowerUser  M=MobileCasual  B=BargainHunter  W=WindowShopper\n";

// ── 6. Downstream: use 6D embeddings as features for a classifier ─────────────
section('Using Embeddings as ML Features');
printf("  6D PCA embeddings (first customer): [%s]\n",
    implode(', ', array_map(fn($v) => round($v, 3),
        array_slice($emb6->toFlatArray(), 0, 6))));
printf("  → Ready as input to: GBDT, RandomForest, KMeans, or MLP\n");
printf("  → 8× fewer features = faster training, less overfitting\n");

echo "\n✓ Done\n";
