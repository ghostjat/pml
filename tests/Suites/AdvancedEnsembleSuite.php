<?php

declare(strict_types=1);

namespace Pml\Tests\Suites;

use Pml\Tensor;
use Pml\Tests\Core\TestRunner;
use Pml\Tests\Datasets\DatasetLoader;
use Pml\Classic\LinearModel\ElasticNet;
use Pml\Classic\Ensemble\AdaBoostClassifier;
use Pml\Classic\Ensemble\BaggingClassifier;
use Pml\Classic\Tree\DecisionTreeClassifier;
use Pml\Classic\ModelSelection\DataSplit;
use Pml\Classic\Metrics\Metrics;

// ═══════════════════════════════════════════════════════════════════════════
//  AdvancedEnsembleSuite — Ensemble models and regularised linear regression
//
//  Tests:
//    1. ElasticNet(α=0.01, l1_ratio=0.5):
//         On the synthetic y=3x₁−2x₂+5+ε dataset.
//         R² > 0.85 (same bar as Ridge/Lasso in ClassicSuite).
//         Verifies coordinate descent converges for a combined L1+L2 penalty.
//
//    2. AdaBoostClassifier(n_estimators=50):
//         On make_classification(n_classes=2) — well-separated binary blobs.
//         Accuracy > 85% on a 20% held-out set.
//         Proves the SAMME weight update and estimator weighting are correct:
//         a random baseline achieves 50%, so 85% requires real learning.
//
//    3. BaggingClassifier(base=DecisionTreeClassifier, n_estimators=20):
//         On the Iris dataset (3-class).
//         Accuracy > 90% on a 20% held-out set.
//         Verifies bootstrap sampling, majority-vote aggregation, and
//         the ability to wrap an arbitrary base estimator.
// ═══════════════════════════════════════════════════════════════════════════

final class AdvancedEnsembleSuite
{
    public static function run(TestRunner $r): void
    {
        $r->suite('Advanced Ensembles & Regularised Linear Models', function(TestRunner $r) {

            // ── Shared datasets — loaded once for the suite ────────────────
            $regData = DatasetLoader::synthetic_regression(n: 500, noise_std: 0.5, seed: 42);
            $classData = DatasetLoader::make_classification(
                n_samples:  200,
                n_features: 4,
                n_classes:  2,
                seed:       42,
            );
            $iris = DatasetLoader::iris();

            // ── Test 1: ElasticNet ─────────────────────────────────────────
            $r->test('ElasticNet(α=0.01, l1_ratio=0.5) on synthetic regression: R² > 0.85', function() use ($r, $regData) {

                // ElasticNet mixes L1 (Lasso) and L2 (Ridge) penalties:
                //   objective = (1/2n)||y − Xw||² + α·l1_ratio·||w||₁
                //                                 + α·(1−l1_ratio)/2·||w||²²
                //
                // With α=0.01 and l1_ratio=0.5 the regularisation is mild
                // enough that the coordinate descent solution tracks the OLS
                // solution closely.  On y=3x₁−2x₂+5+ε the true signal-to-noise
                // is very high (σ_noise=0.5, coefficient magnitude=3), so
                // R²>0.85 is an easily achievable lower bound.
                $en = new ElasticNet(
                    alpha:         0.01,
                    l1_ratio:      0.5,
                    fit_intercept: true,
                    max_iter:      2000,
                    tol:           1e-5,
                );
                $en->fit($regData['X'], $regData['y']);

                $yPred = $en->predict($regData['X']);
                $r2    = Metrics::r2_score($regData['y'], $yPred);

                $r->assertGreaterThan($r2, 0.85, sprintf('ElasticNet R²=%.4f > 0.85', $r2));

                // ── coef_ shape: [n_features] ─────────────────────────────
                // The synthetic dataset has 2 features.
                $r->assertShape($en->coef_, [2], 'coef_ shape = [2]');

                // ── intercept_ is finite ───────────────────────────────────
                $r->assertEq(
                    is_finite($en->intercept_),
                    true,
                    sprintf('intercept_=%.4f is finite', $en->intercept_)
                );
            });

            // ── Test 2: AdaBoostClassifier ─────────────────────────────────
            $r->test('AdaBoostClassifier(50 rounds) on binary make_classification: accuracy > 85%', function() use ($r, $classData) {

                // make_classification produces two Gaussian blobs centred at
                // (0,0,0,0) and (5,5,5,5) with cluster_std=0.8 — easily
                // linearly separable.  A single decision stump (depth-1 tree)
                // achieves ~95% accuracy on this data, so AdaBoost with 50
                // rounds should achieve effectively perfect accuracy.
                //
                // We set the threshold at 85% to leave room for any sampling
                // variance in the train/test split while still proving that
                // the SAMME weight update is working (random baseline = 50%).
                [$Xtrain, $Xtest, $ytrain, $ytest] = DataSplit::train_test_split(
                    $classData['X'], $classData['y'],
                    test_size:    0.2,
                    random_state: 7,
                );

                $ada = new AdaBoostClassifier(
                    n_estimators:  50,
                    learning_rate: 1.0,
                    random_state:  42,
                );
                $ada->fit($Xtrain, $ytrain);

                // ── estimators_ populated ─────────────────────────────────
                // AdaBoost may early-stop if an estimator has 0 error (perfect
                // weak learner), so n_estimators_fitted ≤ 50 is expected.
                $r->assertGreaterThan(
                    (float)count($ada->estimators_),
                    0.0,
                    'at least one estimator was fitted'
                );

                // ── Accuracy > 85% ─────────────────────────────────────────
                $yPred = $ada->predict($Xtest);
                $acc   = Metrics::accuracy_score($ytest, $yPred);
                $r->assertGreaterThan(
                    $acc,
                    0.85,
                    sprintf('AdaBoost accuracy=%.4f > 0.85', $acc)
                );

                // ── Prediction shape ───────────────────────────────────────
                $r->assertShape($yPred, [$Xtest->shape[0]], 'prediction shape = [n_test]');
            });

            // ── Test 3: BaggingClassifier ──────────────────────────────────
            $r->test('BaggingClassifier(DecisionTree base, 20 estimators) on Iris: accuracy > 90%', function() use ($r, $iris) {

                // Bagging wraps a base estimator and trains each copy on a
                // bootstrap sample (sampling with replacement).  Majority vote
                // across 20 trees reduces variance substantially versus a single
                // tree, and the 90% bar is the same as the Pipeline test.
                //
                // Using DecisionTreeClassifier explicitly (not the default None)
                // also tests that BaggingClassifier correctly clones arbitrary
                // Estimator&Predictor instances.
                [$Xtrain, $Xtest, $ytrain, $ytest] = DataSplit::train_test_split(
                    $iris['X'], $iris['y'],
                    test_size:    0.2,
                    random_state: 1,
                );

                $bag = new BaggingClassifier(
                    estimator:    new DecisionTreeClassifier(max_depth: 10),
                    n_estimators: 20,
                    random_state: 42,
                );
                $bag->fit($Xtrain, $ytrain);

                // ── All 20 estimators were fitted ─────────────────────────
                $r->assertEq(
                    $bag->n_estimators_fitted_,
                    20,
                    'n_estimators_fitted_ = 20'
                );

                // ── n_classes_ detected correctly ─────────────────────────
                $r->assertEq($bag->n_classes_, 3, 'n_classes_ = 3 (Iris has 3 classes)');

                // ── Accuracy > 90% ─────────────────────────────────────────
                $yPred = $bag->predict($Xtest);
                $acc   = Metrics::accuracy_score($ytest, $yPred);
                $r->assertGreaterThan(
                    $acc,
                    0.90,
                    sprintf('BaggingClassifier accuracy=%.4f > 0.90', $acc)
                );
            });

        });
    }
}
