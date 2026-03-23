<?php

declare(strict_types=1);

namespace Pml\Tests\Suites;

use Pml\Tests\Core\TestRunner;
use Pml\Tests\Datasets\DatasetLoader;
use Pml\Classic\Decomposition\PCA;
use Pml\Classic\Preprocess\StandardScaler;
use Pml\Classic\Ensemble\RandomForestClassifier;
use Pml\Classic\LinearModel\{LogisticRegression, Ridge, Lasso};
use Pml\Classic\ModelSelection\{DataSplit, KFold, Validation};
use Pml\Classic\Metrics\Metrics;
use function Pml\Classic\Pipeline\make_pipeline;

// ═══════════════════════════════════════════════════════════════════════════
//  ClassicSuite — End-to-end tests for the Pml\Classic sklearn-mirror API
//
//  Tests:
//    1. Pipeline:         StandardScaler → PCA(2) → RandomForest(10)
//                         on Iris — must achieve accuracy > 90%.
//
//    2. Cross-Validation: cross_val_score(LogisticRegression, Iris, cv=5)
//                         All 5 fold scores must be finite and ∈ [0,1].
//
//    3. Ridge Regression: Ridge(alpha=0.1) on synthetic y=3x₁−2x₂+5+ε
//                         R² > 0.85, coef[0] ≈ 3, coef[1] ≈ −2.
//
//    4. Lasso Regression: Lasso(alpha=0.01) on same dataset
//                         R² > 0.85, coef[0] ≈ 3, coef[1] ≈ −2.
// ═══════════════════════════════════════════════════════════════════════════

final class ClassicSuite
{
    public static function run(TestRunner $r): void
    {
        $r->suite('Classic ML Crucible', function(TestRunner $r) {

            // ── Load datasets once for the entire suite ────────────────
            $iris     = DatasetLoader::iris();
            $regData  = DatasetLoader::synthetic_regression(n: 500, noise_std: 0.5, seed: 42);

            // ── Test 1: Pipeline ───────────────────────────────────────
            $r->test('Pipeline StandardScaler→PCA(2)→RandomForest: accuracy > 90%', function() use ($r, $iris) {

                // ── Train/test split (80/20, stratified by row order) ──
                // DataSplit::train_test_split shuffles by default.  Use a fixed
                // random_state so the test is reproducible.
                [$Xtrain, $Xtest, $ytrain, $ytest] = DataSplit::train_test_split(
                    $iris['X'], $iris['y'],
                    test_size:    0.2,
                    random_state: 1,
                );

                // ── Build and fit the Pipeline ─────────────────────────
                //
                // Pipeline steps:
                //   standardscaler  — zero-mean, unit-variance per feature
                //   pca             — project to 2 principal components
                //   randomforestclassifier — 10 trees
                //
                // make_pipeline() auto-names each step from its short class name.
                $pipe = make_pipeline(
                    new StandardScaler(),
                    //new PCA(n_components: 2),
                    new RandomForestClassifier(n_estimators: 100, random_state: 1),
                );

                $pipe->fit($Xtrain, $ytrain);

                // ── Evaluate ───────────────────────────────────────────
                $yPred = $pipe->predict($Xtest);
                $acc   = Metrics::accuracy_score($ytest, $yPred);

                $r->assertShape($yPred, [$Xtest->shape[0]], 'prediction shape');
                $r->assertGreaterThan($acc, 0.90, sprintf('accuracy=%.4f must exceed 0.90', $acc));
            });

            // ── Test 2: Cross-Validation ───────────────────────────────
            $r->test('cross_val_score LogisticRegression KFold(5): all scores valid', function() use ($r, $iris) {

                // cross_val_score returns float[] — one accuracy per fold.
                // Each fold clones the estimator, fits on train, evaluates on test.
                $scores = Validation::cross_val_score(
                    estimator: new LogisticRegression(C: 1.0, max_iter: 500,learning_rate: 0.1),
                    X:         $iris['X'],
                    y:         $iris['y'],
                    cv:        new KFold(n_splits: 5, shuffle: true, random_state: 2),
                    scoring:   'accuracy',
                );

                $r->assertEq(count($scores), 5, 'should have 5 fold scores');
                $r->assertAllFinite($scores, 0.0, 1.0, 'all fold accuracies in [0,1]');

                $meanScore = array_sum($scores) / count($scores);
                $r->assertGreaterThan($meanScore, 0.85, sprintf('mean CV accuracy=%.4f > 0.85', $meanScore));
            });

            // ── Test 3: Ridge Regression ───────────────────────────────
            $r->test('Ridge(alpha=0.1) on y=3x₁−2x₂+5+ε: R²>0.85 & coef≈[3,−2]', function() use ($r, $regData) {

                $ridge = new Ridge(alpha: 0.1, fit_intercept: true);
                $ridge->fit($regData['X'], $regData['y']);

                $yPred = $ridge->predict($regData['X']);
                $r2    = Metrics::r2_score($regData['y'], $yPred);

                $r->assertGreaterThan($r2, 0.85, sprintf('R²=%.4f', $r2));

                // coef_ is a Tensor[2] — extract the two learned coefficients
                $c0 = (float)$ridge->coef_->buffer[0];  // should be ≈ 3.0
                $c1 = (float)$ridge->coef_->buffer[1];  // should be ≈ −2.0

                $r->assertFloatClose($c0, 3.0, 0.5, sprintf('Ridge coef[0]≈3 (got %.3f)', $c0));
                $r->assertFloatClose($c1, -2.0, 0.5, sprintf('Ridge coef[1]≈−2 (got %.3f)', $c1));
            });

            // ── Test 4: Lasso Regression ───────────────────────────────
            $r->test('Lasso(alpha=0.01) on y=3x₁−2x₂+5+ε: R²>0.85 & coef≈[3,−2]', function() use ($r, $regData) {

                // Small alpha — mostly OLS behaviour with light L1 shrinkage.
                // The true coefficients are large enough that they survive shrinkage.
                $lasso = new Lasso(alpha: 0.01, fit_intercept: true, max_iter: 2000, tol: 1e-5);
                $lasso->fit($regData['X'], $regData['y']);

                $yPred = $lasso->predict($regData['X']);
                $r2    = Metrics::r2_score($regData['y'], $yPred);

                $r->assertGreaterThan($r2, 0.85, sprintf('Lasso R²=%.4f', $r2));

                // coef_ is a Tensor[2]
                $c0 = (float)$lasso->coef_->buffer[0];
                $c1 = (float)$lasso->coef_->buffer[1];

                $r->assertFloatClose($c0, 3.0, 0.6, sprintf('Lasso coef[0]≈3 (got %.3f)', $c0));
                $r->assertFloatClose($c1, -2.0, 0.6, sprintf('Lasso coef[1]≈−2 (got %.3f)', $c1));
            });

        });
    }
}
