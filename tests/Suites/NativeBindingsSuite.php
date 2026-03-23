<?php

declare(strict_types=1);

namespace Pml\Tests\Suites;

use Pml\Tensor;
use Pml\Tests\Core\TestRunner;
use Pml\Tests\Datasets\DatasetLoader;
use Pml\Classic\SVM\{SVC, SVR, LibSVMBridge};
use Pml\Classic\Ensemble\{XGBClassifier, XGBoostBridge};
use Pml\Classic\Cluster\DBSCAN;
use Pml\Classic\ModelSelection\DataSplit;
use Pml\Classic\Metrics\Metrics;

// ═══════════════════════════════════════════════════════════════════════════
//  NativeBindingsSuite — Phase 6: C-library FFI bindings + DBSCAN
//
//  Tests:
//    1. SVC(kernel='rbf', C=10) on Iris (3-class OvO):
//         • n_classes_ = 3  (OvO handles multiclass natively in libsvm)
//         • gamma_ is finite and positive  (resolved from gamma='scale')
//         • Accuracy > 90% on 20% held-out set
//         • Prediction shape = [n_test]
//         SKIP if libsvm.so is not on the library search path.
//
//    2. SVR(kernel='rbf', C=1, epsilon=0.1) on synthetic_regression:
//         • n_features_in_ = 2
//         • gamma_ is finite and positive
//         • R² > 0.85 on training set
//         SKIP if libsvm.so is not on the library search path.
//
//    3. XGBClassifier(n_estimators=20, max_depth=3) on Iris:
//         • n_classes_ = 3
//         • objective_ = 'multi:softprob'  (auto-selected for K > 2)
//         • Accuracy > 90% on 20% held-out set
//         • Prediction shape = [n_test]
//         SKIP if libxgboost.so is not on the library search path.
//
//    4. DBSCAN(eps=2.0, min_samples=5) on make_blobs(centers=3):
//         Pure PHP + BLAS — no external C library required.
//         • labels_ shape = [n_samples]
//         • At least 3 distinct non-noise cluster IDs present
//         • core_sample_indices_ is non-empty
//         • Noise fraction < 10%  (well-separated blobs should be nearly all core)
//
//  ── Why these thresholds? ─────────────────────────────────────────────────
//
//  SVC (RBF, C=10) on Iris:
//    RBF-SVM on the raw 4-feature Iris dataset achieves 97–100% accuracy in
//    sklearn with default gamma.  We require > 90% to leave generous margin
//    for any floating-point divergence in the float32 → double widening path
//    while still proving the C-binding is working (random baseline = 33%).
//
//  SVR (RBF) on synthetic y=3x₁−2x₂+5+ε:
//    The same bar as Ridge/Lasso/ElasticNet (R²>0.85).  SVR with an RBF kernel
//    can model this nearly-linear signal, so the bar is achievable even with
//    the ε-insensitive tube; any lower threshold would not distinguish a
//    functioning SVR from a trivial mean-predictor (R²=0).
//
//  XGBClassifier on Iris:
//    XGBoost with 20 trees and max_depth=3 handily solves this near-linearly-
//    separable 3-class problem.  The > 90% bar proves that:
//      (a) The DMatrix zero-copy path marshalled the data correctly.
//      (b) n_estimators rounds of XGBoosterUpdateOneIter ran without error.
//      (c) The multi:softprob output was correctly argmax-ed in predict().
//
//  DBSCAN eps=2.0, min_samples=5 on make_blobs(cluster_std=0.9):
//    Cluster centres are placed diagonally: k·(5,5), so the minimum
//    inter-cluster distance is 5√2 ≈ 7.07.  With cluster_std=0.9, points
//    lie within ≈ 3σ = 2.7 units of their centre; eps=2.0 captures ≈ 2.2σ
//    (≈98% of each cluster) with no cross-cluster connections.
//    min_samples=5 with 50 samples per cluster trivially designates every
//    in-cluster point as a core point.
// ═══════════════════════════════════════════════════════════════════════════

final class NativeBindingsSuite
{
    public static function run(TestRunner $r): void
    {
        $r->suite('Phase 6: Native C-Bindings & DBSCAN', function(TestRunner $r) {

            // ── Shared datasets — loaded once for the suite ────────────────
            $iris    = DatasetLoader::iris();
            $regData = DatasetLoader::synthetic_regression(n: 500, noise_std: 0.5, seed: 42);

            // ── Iris 80/20 split — reused by SVC and XGBClassifier ─────────
            [$Xtrain, $Xtest, $ytrain, $ytest] = DataSplit::train_test_split(
                $iris['X'], $iris['y'],
                test_size:    0.2,
                random_state: 1,
            );

            // ── Test 1: SVC on Iris ────────────────────────────────────────
            $r->test('SVC(RBF, C=10) on Iris: n_classes_=3, gamma_ finite, accuracy > 90%', function() use ($r, $Xtrain, $Xtest, $ytrain, $ytest) {

                // Probe for libsvm availability.
                // LibSVMBridge::get() throws if libsvm.so is not on LD_LIBRARY_PATH.
                // SkipException propagates out of the catch block normally.
                try {
                    LibSVMBridge::get();
                } catch (\Throwable $e) {
                    $r->skip('libsvm.so unavailable: ' . $e->getMessage());
                }

                // ── Fit ───────────────────────────────────────────────────
                //
                // C=10 allows fewer margin violations → tighter fit on Iris.
                // RBF kernel with gamma='scale' adapts to the feature variance.
                // libsvm uses OvO internally: 3 binary classifiers for 3 classes.
                $svc = new SVC(C: 10.0, kernel: 'rbf', gamma: 'scale');
                
                $svc->fit($Xtrain, $ytrain);

                // ── n_classes_ must reflect all 3 Iris classes ────────────
                $r->assertEq($svc->n_classes_, 3, 'n_classes_ = 3 (Iris has 3 classes)');

                // ── n_features_in_ must match training feature count ───────
                $r->assertEq(
                    $svc->n_features_in_,
                    $Xtrain->shape[1],
                    'n_features_in_ matches training feature count'
                );

                // ── gamma_ must be finite and positive ─────────────────────
                //
                // gamma='scale' → γ = 1 / (d · Var(X)).  With 4 Iris features
                // and typical feature variance, this is a small positive float.
                // NaN or Inf would indicate the variance computation failed.
                $r->assertEq(
                    is_finite($svc->gamma_) && $svc->gamma_ > 0.0,
                    true,
                    sprintf('gamma_=%.6f is finite and positive', $svc->gamma_)
                );

                // ── Accuracy > 90% ─────────────────────────────────────────
                $yPred = $svc->predict($Xtest);
                $acc   = Metrics::accuracy_score($ytest, $yPred);
                $r->assertGreaterThan(
                    $acc,
                    0.90,
                    sprintf('SVC accuracy=%.4f > 0.90', $acc)
                );

                // ── Prediction shape = [n_test] ────────────────────────────
                $r->assertShape($yPred, [$Xtest->shape[0]], 'prediction shape = [n_test]');
            });

            // ── Test 2: SVR on synthetic regression ────────────────────────
            $r->test('SVR(RBF, C=1, ε=0.1) on synthetic regression: n_features_in_=2, R² > 0.85', function() use ($r, $regData) {

                // Probe for libsvm availability (same bridge as SVC).
                try {
                    LibSVMBridge::get();
                } catch (\Throwable $e) {
                    $r->skip('libsvm.so unavailable: ' . $e->getMessage());
                }

                // ── Fit ───────────────────────────────────────────────────
                //
                // Default SVR: RBF kernel, C=1, ε=0.1, gamma='scale'.
                // The synthetic dataset has 2 features (x₁, x₂) and a
                // near-linear signal y=3x₁−2x₂+5+ε.  SVR can model this.
                $svr = new SVR(C: 1.0, epsilon: 0.1, kernel: 'rbf', gamma: 'scale');
                $svr->fit($regData['X'], $regData['y']);

                // ── n_features_in_ must be 2 (synthetic dataset has 2 cols) ─
                $r->assertEq($svr->n_features_in_, 2, 'n_features_in_ = 2');

                // ── gamma_ must be finite and positive ─────────────────────
                $r->assertEq(
                    is_finite($svr->gamma_) && $svr->gamma_ > 0.0,
                    true,
                    sprintf('gamma_=%.6f is finite and positive', $svr->gamma_)
                );

                // ── R² > 0.85 ──────────────────────────────────────────────
                //
                // Train-set R² is the correct metric here because:
                //  • We are testing that the C-binding returned a valid model
                //    (predictions correlated with the target).
                //  • Generalisation on a held-out set is irrelevant to the FFI
                //    correctness proof; we already test generalisation in Lasso/
                //    Ridge/ElasticNet using the same dataset.
                $yPred = $svr->predict($regData['X']);
                $r2    = Metrics::r2_score($regData['y'], $yPred);
                $r->assertGreaterThan(
                    $r2,
                    0.85,
                    sprintf('SVR R²=%.4f > 0.85', $r2)
                );

                // ── Prediction shape = [n_samples] ────────────────────────
                $r->assertShape(
                    $yPred,
                    [$regData['X']->shape[0]],
                    'prediction shape = [n_samples]'
                );
            });

            // ── Test 3: XGBClassifier on Iris ─────────────────────────────
            $r->test('XGBClassifier(n_estimators=20, max_depth=3) on Iris: objective_=multi:softprob, accuracy > 90%', function() use ($r, $Xtrain, $Xtest, $ytrain, $ytest) {

                // Probe for libxgboost availability.
                try {
                    XGBoostBridge::get();
                } catch (\Throwable $e) {
                    $r->skip('libxgboost.so unavailable: ' . $e->getMessage());
                }

                // ── Fit ───────────────────────────────────────────────────
                //
                // n_estimators=20: 20 boosting rounds — fast, sufficient for Iris.
                // max_depth=3: each tree can model depth-3 interactions.
                // objective='auto': XGBClassifier detects K=3 at fit() time and
                //   selects 'multi:softprob', which outputs K probabilities per
                //   sample.  predict() then takes argmax.
                $xgb = new XGBClassifier(
                    n_estimators: 20,
                    max_depth:    3,
                    random_state: 42,
                );
                $xgb->fit($Xtrain, $ytrain);

                // ── n_classes_ must be 3 ──────────────────────────────────
                $r->assertEq($xgb->n_classes_, 3, 'n_classes_ = 3');

                // ── objective_ must be auto-resolved to multi:softprob ─────
                //
                // This assertion verifies the auto-dispatch logic inside fit():
                //   K=2 → 'binary:logistic'
                //   K>2 → 'multi:softprob'   ← expected for Iris (K=3)
                $r->assertEq(
                    $xgb->objective_,
                    'multi:softprob',
                    "objective_ = 'multi:softprob'"
                );

                // ── n_features_in_ must match the training feature count ───
                $r->assertEq(
                    $xgb->n_features_in_,
                    $Xtrain->shape[1],
                    'n_features_in_ matches training feature count'
                );

                // ── Accuracy > 90% ─────────────────────────────────────────
                $yPred = $xgb->predict($Xtest);
                $acc   = Metrics::accuracy_score($ytest, $yPred);
                $r->assertGreaterThan(
                    $acc,
                    0.90,
                    sprintf('XGBClassifier accuracy=%.4f > 0.90', $acc)
                );

                // ── Prediction shape = [n_test] ────────────────────────────
                $r->assertShape($yPred, [$Xtest->shape[0]], 'prediction shape = [n_test]');
            });

            // ── Test 4: DBSCAN on make_blobs ──────────────────────────────
            $r->test('DBSCAN(eps=2.0, min_samples=5) on make_blobs(centers=3): ≥3 clusters, noise < 10%', function() use ($r) {

                // Pure PHP + BLAS — no FFI library skip needed.
                //
                // Blobs: 3 centres at (0,0), (5,5), (10,10) in 2D space.
                // cluster_std=0.9: points lie within ~2.7 units of their centre.
                // Inter-cluster gap: 5√2 ≈ 7.07 >> eps=2.0 → no cross-cluster edges.
                $data = DatasetLoader::make_blobs(
                    n_samples:   150,
                    centers:     3,
                    n_features:  2,
                    cluster_std: 0.9,
                    seed:        42,
                );

                $db = new DBSCAN(eps: 2.0, min_samples: 5);
                $db->fit($data['X']);

                // ── labels_ shape = [n_samples] ───────────────────────────
                $r->assertShape($db->labels_, [150], 'labels_ shape = [150]');

                // ── At least 3 distinct non-noise cluster IDs ─────────────
                //
                // We scan labels_ for unique values that are ≥ 0 (noise = -1).
                // At minimum the 3 blobs must be discovered as 3 clusters.
                // DBSCAN may split a blob if eps is very tight, so we require
                // count(clusterIds) ≥ 3, not exactly 3.
                $clusterIds = [];
                $noiseCount = 0;
                for ($i = 0; $i < 150; $i++) {
                    $label = (int)(float)$db->labels_->buffer[$i];
                    if ($label === -1) {
                        $noiseCount++;
                    } else {
                        $clusterIds[$label] = true;
                    }
                }

                $r->assertGreaterThan(
                    (float)count($clusterIds),
                    2.0,   // strictly > 2 ≡ at least 3
                    sprintf('found %d distinct cluster IDs (≥3 required)', count($clusterIds))
                );

                // ── Noise fraction < 10% ───────────────────────────────────
                //
                // With well-separated blobs and eps=2.0 capturing ≈98% of each
                // cluster, fewer than 15 of 150 points should be classified as
                // noise (label = -1).  A high noise fraction would indicate eps
                // is too small or the distance matrix has a BLAS arithmetic bug.
                $noiseFrac = $noiseCount / 150;
                $r->assertLessThan(
                    $noiseFrac,
                    0.10,
                    sprintf('noise fraction=%.3f < 0.10 (noisy points: %d/150)', $noiseFrac, $noiseCount)
                );

                // ── core_sample_indices_ must be non-empty ─────────────────
                //
                // With 50 tightly-packed samples per cluster and min_samples=5,
                // virtually every in-cluster point qualifies as a core point.
                // An empty core_sample_indices_ means DBSCAN found no density
                // above the threshold — the algorithm did nothing useful.
                $r->assertGreaterThan(
                    (float)count($db->core_sample_indices_),
                    0.0,
                    'core_sample_indices_ is non-empty'
                );
            });

        });
    }
}
