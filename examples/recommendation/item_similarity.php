<?php
declare(strict_types=1);
/**
 * PRODUCT RECOMMENDATION ENGINE — Item Similarity + Clustering
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Recommend similar products to an active user based on
 *            their purchase/view history using item embeddings.
 * Method   : Build item feature vectors → PCA → KMeans clusters →
 *            cosine similarity within cluster for fast lookup.
 * Business : Amazon attributes 35 % of revenue to recommendations.
 *            Even a 1 % CTR lift on recommendations translates to
 *            millions of dollars in incremental revenue.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Tensor;
use Pml\Estimators\Clusterers\KMeans;
use Pml\Estimators\Decomposition\PrincipalComponentAnalysis;
use Pml\Transformers\StandardScaler;

section('Product Recommendation — PCA + KMeans + Cosine Similarity');

// ── 1. Product feature catalogue ─────────────────────────────────────────────
// Each product has features: price_tier(1-5), avg_rating, review_count,
//   category(0-9), brand_tier(1-3), purchase_frequency, return_rate,
//   seasonal_index, margin_pct, weight_kg

mt_srand(42);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

$nProducts = 2000;
$productNames = [];
$rows = [];

$categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports',
               'Beauty', 'Toys', 'Garden', 'Food', 'Automotive'];

for ($i = 0; $i < $nProducts; $i++) {
    $cat           = mt_rand(0, 9);
    $priceTier     = mt_rand(1, 5);
    $productNames[] = $categories[$cat] . "_prod_{$i}";

    $rows[] = [
        (float)$priceTier,
        $rng(1.0, 5.0),       // avg rating
        $rng(0, 5000),        // review count
        (float)$cat,
        (float)mt_rand(1, 3), // brand tier
        $rng(0.01, 0.5),      // purchase frequency
        $rng(0.0, 0.3),       // return rate
        $rng(0.5, 2.0),       // seasonal index
        $rng(0.1, 0.6),       // margin %
        $rng(0.1, 20.0),      // weight kg
    ];
}

$ds = Dataset::fromArray($rows);

// ── 2. Scale → PCA → Cluster ──────────────────────────────────────────────────
section('Building Item Index');
$t0 = microtime(true);

$scaler = new StandardScaler();
$scaler->fit($ds);
$scaled = $scaler->transform($ds);

$pca = new PrincipalComponentAnalysis(nComponents: 6);
$pca->train($scaled);
$reduced = $pca->predict($scaled);  // [N x 6] Tensor

// Wrap PCA output in Dataset so KMeans can consume it
$reducedFlat = $reduced->toFlatArray();
$nComp = 6;
$reducedRows = array_chunk($reducedFlat, $nComp);
$reducedDs = Dataset::fromArray($reducedRows);

$kmeans = new KMeans(k: 20, maxIter: 300);
$kmeans->train($reducedDs);
$clusters = $kmeans->predict($reducedDs)->toFlatArray();

metric('Products indexed', $nProducts);
metric('PCA components', 6);
metric('KMeans clusters', 20);
metric('Index build time', elapsed($t0));

// ── 3. Cosine similarity recommendation ───────────────────────────────────────
function cosineSim(array $a, array $b): float
{
    $dot = $normA = $normB = 0.0;
    foreach ($a as $i => $v) {
        $dot   += $v * $b[$i];
        $normA += $v * $v;
        $normB += $b[$i] * $b[$i];
    }
    return ($normA * $normB > 0) ? $dot / (sqrt($normA) * sqrt($normB)) : 0.0;
}

function recommend(int $productId, array $reducedVectors, array $clusters, array $names, int $topK = 5): array
{
    $targetCluster = $clusters[$productId];
    $targetVec     = $reducedVectors[$productId];
    $candidates    = [];

    foreach ($clusters as $i => $c) {
        if ($i === $productId || $c !== $targetCluster) continue;
        $candidates[] = ['id' => $i, 'score' => cosineSim($targetVec, $reducedVectors[$i])];
    }

    usort($candidates, fn($a, $b) => $b['score'] <=> $a['score']);
    return array_slice($candidates, 0, $topK);
}

// Extract reduced vectors (already computed above)
$reducedVectors = [];
for ($i = 0; $i < $nProducts; $i++) {
    $reducedVectors[$i] = array_slice($reducedFlat, $i * $nComp, $nComp);
}

// ── 4. Demo recommendations ────────────────────────────────────────────────────
section('Recommendation Results');

$queryProducts = [42, 100, 500, 1337];

foreach ($queryProducts as $qid) {
    printf("\n  Customers who viewed \"%s\" also liked:\n", $productNames[$qid]);
    $recs = recommend($qid, $reducedVectors, (array)$clusters, $productNames, topK: 5);
    foreach ($recs as $rank => $rec) {
        printf("    %d. %-30s  similarity=%.4f\n",
               $rank + 1, $productNames[$rec['id']], $rec['score']);
    }
}

// ── 5. User-based session recommendation ─────────────────────────────────────
section('Session-Based Recommendations');

// User viewed products: 10, 15, 20 — blend their embeddings
$userHistory = [10, 15, 20];
$blended = array_fill(0, $nComp, 0.0);
foreach ($userHistory as $pid) {
    foreach ($reducedVectors[$pid] as $j => $v) $blended[$j] += $v;
}
foreach ($blended as &$v) $v /= count($userHistory);

// Find most similar products across ALL clusters
$allScores = [];
foreach ($reducedVectors as $i => $vec) {
    if (in_array($i, $userHistory)) continue;
    $allScores[$i] = cosineSim($blended, $vec);
}
arsort($allScores);
$topRecs = array_slice($allScores, 0, 5, preserve_keys: true);

printf("\n  User session history: %s\n",
       implode(', ', array_map(fn($id) => $productNames[$id], $userHistory)));
printf("  Recommended for you:\n");
foreach ($topRecs as $pid => $score) {
    printf("    · %-30s  score=%.4f\n", $productNames[$pid], $score);
}

echo "\n✓ Done\n";
