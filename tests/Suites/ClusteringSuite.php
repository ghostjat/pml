<?php

declare(strict_types=1);

namespace Pml\Tests\Suites;

use Pml\Tensor;
use Pml\Tests\Core\TestRunner;
use Pml\Tests\Datasets\DatasetLoader;
use Pml\Classic\Cluster\KMeans;
use Pml\Classic\Neighbors\KNeighborsClassifier;
use Pml\Classic\ModelSelection\DataSplit;
use Pml\Classic\Metrics\Metrics;

// ═══════════════════════════════════════════════════════════════════════════
//  ClusteringSuite — Distance-based algorithm correctness tests
//
//  Tests:
//    1. KMeans(k=3) on make_blobs(centers=3):
//         • cluster_centers_ has shape [3, n_features]
//         • inertia_ is finite and strictly positive
//         • Labels cover all 3 cluster IDs {0, 1, 2}
//
//    2. KNeighborsClassifier(k=5) on Iris:
//         • Accuracy > 90% on a held-out 20% test split
//         • _fit_X and _fit_y shapes match training data
//         • Prediction tensor shape equals [n_test_samples]
//
//  Why these thresholds?
//    KMeans on well-separated blobs (5 σ between cluster centers):
//      The inertia must be positive and finite — 0 would mean all samples
//      collapsed to their centroid (impossible with Gaussian noise), and
//      infinity/NaN would indicate numerical instability in the distance
//      matrix computation.
//
//    KNN on Iris (k=5, 80/20 split):
//      KNN is a non-parametric instance-based learner.  With Euclidean
//      distance on normalised Iris features it reliably achieves ≥ 93% on
//      any standard split.  We require > 90% to leave margin for unlucky
//      random_state while still proving the implementation is correct.
// ═══════════════════════════════════════════════════════════════════════════

final class ClusteringSuite
{
    public static function run(TestRunner $r): void
    {
        $r->suite('Clustering & Distance-Based Models', function(TestRunner $r) {

            // ── Test 1: KMeans ─────────────────────────────────────────────
            $r->test('KMeans(k=3) on make_blobs(centers=3): shape, inertia, label coverage', function() use ($r) {

                // Well-separated blobs — KMeans should converge in < 10 iterations
                // and assign each sample to its ground-truth cluster.
                $data = DatasetLoader::make_blobs(
                    n_samples:   150,
                    centers:     3,
                    n_features:  2,
                    cluster_std: 0.9,
                    seed:        42,
                );

                $km = new KMeans(
                    n_clusters:   3,
                    max_iter:     300,
                    n_init:       10,
                    random_state: 1,
                );
                $km->fit($data['X']);

                // ── cluster_centers_ shape: [k, n_features] ───────────────
                //
                // For k=3, n_features=2 this must be exactly [3, 2].
                // The shape proves KMeans initialised and updated k centroids,
                // not fewer (merged clusters) or more (split clusters).
                $r->assertShape(
                    $km->cluster_centers_,
                    [3, 2],
                    'cluster_centers_ shape = [3, 2]'
                );

                // ── inertia_ is finite and positive ───────────────────────
                //
                // inertia = Σ_i ||x_i − c_{label_i}||²
                // With Gaussian noise std=0.9 it cannot be 0 (that would
                // require every point to coincide exactly with its centroid).
                // NaN/Inf would indicate a BLAS or distance-matrix bug.
                $r->assertEq(
                    is_finite($km->inertia_) && $km->inertia_ > 0.0,
                    true,
                    sprintf('inertia_=%.4f is finite and positive', $km->inertia_)
                );

                // ── labels_ shape matches input ────────────────────────────
                $r->assertShape($km->labels_, [150], 'labels_ shape = [150]');

                // ── All k=3 cluster IDs must appear in labels_ ─────────────
                //
                // If any cluster is empty the centroid update would produce
                // NaN centroids (division by zero).  Verifying all IDs appear
                // confirms KMeans produced a valid non-degenerate partition.
                $seenIds = [];
                for ($i = 0; $i < 150; $i++) {
                    $seenIds[(int)(float)$km->labels_->buffer[$i]] = true;
                }
                $r->assertEq(count($seenIds), 3, 'all 3 cluster IDs present in labels_');
            });

            // ── Test 2: KNeighborsClassifier ───────────────────────────────
            $r->test('KNeighborsClassifier(k=5) on Iris: accuracy > 90%', function() use ($r) {

                $iris = DatasetLoader::iris();

                // 80/20 stratified split — same random_state as ClassicSuite
                // to ensure comparability across runs.
                [$Xtrain, $Xtest, $ytrain, $ytest] = DataSplit::train_test_split(
                    $iris['X'], $iris['y'],
                    test_size:    0.2,
                    random_state: 1,
                );

                $knn = new KNeighborsClassifier(n_neighbors: 5);
                $knn->fit($Xtrain, $ytrain);

                // ── _fit_X and _fit_y store the training data ──────────────
                //
                // KNN is lazy: fit() merely memorises the training set.
                // We verify the stored shapes to confirm no silent truncation.
                $r->assertShape(
                    $knn->_fit_X,
                    $Xtrain->shape,
                    '_fit_X shape matches training X'
                );
                $r->assertShape(
                    $knn->_fit_y,
                    $ytrain->shape,
                    '_fit_y shape matches training y'
                );

                // ── Prediction shape ───────────────────────────────────────
                $yPred = $knn->predict($Xtest);
                $r->assertShape(
                    $yPred,
                    [$Xtest->shape[0]],
                    'prediction shape = [n_test_samples]'
                );

                // ── Accuracy > 90% ─────────────────────────────────────────
                //
                // Iris is near-perfectly separable in petal-feature space.
                // KNN (k=5) consistently achieves 93–97% on any 80/20 split —
                // the 90% floor leaves ample margin for unlucky splits.
                $acc = Metrics::accuracy_score($ytest, $yPred);
                $r->assertGreaterThan(
                    $acc,
                    0.90,
                    sprintf('KNN accuracy=%.4f > 0.90', $acc)
                );
            });

        });
    }
}
