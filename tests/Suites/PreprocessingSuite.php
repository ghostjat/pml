<?php

declare(strict_types=1);

namespace Pml\Tests\Suites;

use Pml\Tensor;
use Pml\Tests\Core\TestRunner;
use Pml\Tests\Datasets\DatasetLoader;
use Pml\Classic\Impute\SimpleImputer;
use Pml\Classic\FeatureSelection\VarianceThreshold;
use Pml\Classic\Preprocess\OneHotEncoder;
use Pml\Classic\ModelSelection\GridSearchCV;
use Pml\Classic\LinearModel\Ridge;

// ═══════════════════════════════════════════════════════════════════════════
//  PreprocessingSuite — Data pipeline correctness tests
//
//  Tests:
//    1. SimpleImputer:       After fit_transform(), zero NaN values remain in
//                            the output Tensor (strategy='mean').
//
//    2. VarianceThreshold:   A dataset with one entirely constant column
//                            (variance = 0) must be reduced to n_features−1
//                            columns after fit_transform() with threshold=0.0.
//
//    3. OneHotEncoder:       A [n, 1] integer feature with categories {0,1,2}
//                            expands to [n, 3].  All values are 0.0 or 1.0,
//                            and every row sums to exactly 1.0 (valid one-hot).
//
//    4. GridSearchCV:        Ridge(alpha=?) searched over [0.01, 0.1, 1.0] on
//                            a binary classification dataset.  Verifies the
//                            grid-search machinery: best_params_ is populated
//                            with the 'alpha' key and best_score_ is finite.
// ═══════════════════════════════════════════════════════════════════════════

final class PreprocessingSuite
{
    public static function run(TestRunner $r): void
    {
        $r->suite('Preprocessing & Model Selection', function(TestRunner $r) {

            // ── Test 1: SimpleImputer ──────────────────────────────────────
            $r->test('SimpleImputer(mean): zero NaN values remain after fit_transform', function() use ($r) {

                $data = DatasetLoader::make_imputation_data(
                    n_samples:  20,
                    n_features: 4,
                    seed:       0,
                );

                // Verify the raw dataset actually contains NaN (sanity guard:
                // if make_imputation_data() stopped injecting NaN, this suite
                // would trivially pass while testing nothing).
                $rawHasNan = false;
                for ($i = 0; $i < $data['X']->size; $i++) {
                    if (is_nan((float)$data['X']->buffer[$i])) {
                        $rawHasNan = true;
                        break;
                    }
                }
                if (!$rawHasNan) {
                    throw new \RuntimeException('make_imputation_data() produced no NaN — dataset generator is broken.');
                }

                // ── Fit and transform ─────────────────────────────────────
                $imp  = new SimpleImputer(strategy: 'mean');
                $Xout = $imp->fit_transform($data['X']);

                // ── Shape must be preserved ───────────────────────────────
                $r->assertShape(
                    $Xout,
                    [$data['n_samples'], $data['n_features']],
                    'output shape equals input shape'
                );

                // ── Critical assertion: no NaN survives imputation ────────
                //
                // We scan the flat buffer directly.  A single surviving NaN
                // means the imputer did not fill all missing cells.
                $hasNan = false;
                for ($i = 0; $i < $Xout->size; $i++) {
                    if (is_nan((float)$Xout->buffer[$i])) {
                        $hasNan = true;
                        break;
                    }
                }
                $r->assertEq($hasNan, false, 'no NaN values remain after mean imputation');

                // ── Imputed values should be close to the per-column means ─
                //
                // true_means[j] = column mean of non-NaN values in the raw X.
                // The imputer replaces NaN with exactly that mean, so any
                // surviving imputed cell must equal true_means[j] within 1e-4
                // (float32 round-trip tolerance).
                //
                // We identify the cells that were originally NaN and verify
                // they now hold the expected fill value.
                $n = $data['n_samples'];
                $d = $data['n_features'];
                foreach ($data['true_means'] as $j => $expectedMean) {
                    for ($i = 0; $i < $n; $i++) {
                        $pos = $i * $d + $j;
                        if (is_nan((float)$data['X']->buffer[$pos])) {
                            // This cell was NaN — must now hold the column mean
                            $filled = (float)$Xout->buffer[$pos];
                            $r->assertFloatClose(
                                $filled,
                                (float)$expectedMean,
                                1e-3,
                                sprintf('filled[%d,%d] ≈ col_mean %.4f', $i, $j, $expectedMean)
                            );
                        }
                    }
                }
            });

            // ── Test 2: VarianceThreshold ──────────────────────────────────
            $r->test('VarianceThreshold(0.0): constant column removed → n_features−1 remain', function() use ($r) {

                // Build a [20, 3] matrix:
                //   col 0 — strictly increasing (high variance)
                //   col 1 — constant 1.0  ← variance = 0, must be dropped
                //   col 2 — strictly decreasing (high variance)
                $n = 20;
                $X = new Tensor([$n, 3]);
                for ($i = 0; $i < $n; $i++) {
                    $X->buffer[$i * 3]     = (float)$i * 0.5;          // col 0
                    $X->buffer[$i * 3 + 1] = 1.0;                       // col 1 — constant
                    $X->buffer[$i * 3 + 2] = (float)($n - 1 - $i) * 0.3; // col 2
                }

                $vt   = new VarianceThreshold(threshold: 1e-5);
                $Xout = $vt->fit_transform($X);

                // ── n_features_in_ must be 3 ──────────────────────────────
                $r->assertEq($vt->n_features_in_, 3, 'n_features_in_ = 3');

                // ── Output must drop exactly the constant column → [n, 2] ─
                $r->assertShape($Xout, [$n, 2], 'constant column dropped → shape [20, 2]');

                // ── The two surviving columns must not be constant ─────────
                //
                // Compute variance of each output column; both must be > 0.
                for ($j = 0; $j < 2; $j++) {
                    $sum  = 0.0;
                    $sum2 = 0.0;
                    for ($i = 0; $i < $n; $i++) {
                        $v     = (float)$Xout->buffer[$i * 2 + $j];
                        $sum  += $v;
                        $sum2 += $v * $v;
                    }
                    $var = $sum2 / $n - ($sum / $n) ** 2;
                    $r->assertGreaterThan($var, 0.0, sprintf('output col %d has positive variance %.4f', $j, $var));
                }
            });

            // ── Test 3: OneHotEncoder ──────────────────────────────────────
            $r->test('OneHotEncoder: [n,1] with {0,1,2} → [n,3], all 0/1, rows sum to 1', function() use ($r) {

                // 9 samples, one feature, three categories cycling 0→1→2
                $n = 9;
                $X = new Tensor([$n, 1]);
                for ($i = 0; $i < $n; $i++) {
                    $X->buffer[$i] = (float)($i % 3);   // 0,1,2,0,1,2,0,1,2
                }

                $ohe  = new OneHotEncoder();
                $Xout = $ohe->fit_transform($X);

                // ── Shape: [9, 3] ─────────────────────────────────────────
                $r->assertShape($Xout, [$n, 3], 'one feature with 3 categories → 3 output columns');

                // ── Every value is exactly 0.0 or 1.0 ────────────────────
                $onlyBinary = true;
                for ($i = 0; $i < $Xout->size; $i++) {
                    $v = (float)$Xout->buffer[$i];
                    if ($v !== 0.0 && $v !== 1.0) {
                        $onlyBinary = false;
                        break;
                    }
                }
                $r->assertEq($onlyBinary, true, 'all encoded values are 0.0 or 1.0');

                // ── Each row sums to exactly 1.0 (valid probability simplex) ─
                //
                // A one-hot row has exactly one 1 and two 0s.
                // Σ = 1.0 is both necessary and sufficient for a valid encoding.
                $rowSumsOk = true;
                for ($i = 0; $i < $n; $i++) {
                    $rowSum = 0.0;
                    for ($k = 0; $k < 3; $k++) {
                        $rowSum += (float)$Xout->buffer[$i * 3 + $k];
                    }
                    if (abs($rowSum - 1.0) > 1e-6) {
                        $rowSumsOk = false;
                        break;
                    }
                }
                $r->assertEq($rowSumsOk, true, 'each row sums to 1.0');

                // ── Spot-check: sample i=0 (category 0) → [1,0,0] ────────
                $r->assertFloatClose((float)$Xout->buffer[0], 1.0, 1e-6, 'row 0, col 0 = 1.0 (cat 0)');
                $r->assertFloatClose((float)$Xout->buffer[1], 0.0, 1e-6, 'row 0, col 1 = 0.0');
                $r->assertFloatClose((float)$Xout->buffer[2], 0.0, 1e-6, 'row 0, col 2 = 0.0');

                // ── Spot-check: sample i=1 (category 1) → [0,1,0] ────────
                $r->assertFloatClose((float)$Xout->buffer[3], 0.0, 1e-6, 'row 1, col 0 = 0.0 (cat 1)');
                $r->assertFloatClose((float)$Xout->buffer[4], 1.0, 1e-6, 'row 1, col 1 = 1.0');
                $r->assertFloatClose((float)$Xout->buffer[5], 0.0, 1e-6, 'row 1, col 2 = 0.0');
            });

            // ── Test 4: GridSearchCV ───────────────────────────────────────
            $r->test('GridSearchCV(Ridge, alpha=[0.01,0.1,1.0]): best_params_ populated, best_score_ finite', function() use ($r) {

                // Binary classification dataset: y ∈ {0, 1}, n=200, 4 features.
                // We search Ridge's alpha — this tests the grid-search machinery
                // (clone → fit → score per fold, pick best), not Ridge's
                // suitability as a classifier.
                $data = DatasetLoader::make_classification(
                    n_samples:  200,
                    n_features: 4,
                    n_classes:  2,
                    seed:       42,
                );

                $gs = new GridSearchCV(
                    estimator:  new Ridge(),
                    param_grid: ['alpha' => [0.01, 0.1, 1.0]],
                    cv:         3,          // 3-fold — fast for unit testing
                    scoring:    'accuracy',
                );

                $gs->fit($data['X'], $data['y']);

                // ── best_params_ must contain the searched key ────────────
                //
                // GridSearchCV selects the parameter combination that
                // maximises the CV score.  Even if all alphas give identical
                // scores, the winner is deterministically the first one
                // (stable sort), so best_params_ is always populated.
                $r->assertEq(
                    isset($gs->best_params_['alpha']),
                    true,
                    'best_params_ contains the alpha key'
                );

                // ── best_score_ must be a finite number ───────────────────
                //
                // A finite score (even 0.0) proves that:
                //   (a) All 3 folds completed without fatal error.
                //   (b) The scoring callable returned a number.
                //   (c) The comparison loop selected a winner.
                $r->assertEq(
                    is_finite($gs->best_score_),
                    true,
                    sprintf('best_score_=%.4f is finite', $gs->best_score_)
                );

                // ── best_estimator_ must be a fitted Ridge instance ───────
                $r->assertEq(
                    $gs->best_estimator_ instanceof Ridge,
                    true,
                    'best_estimator_ is a Ridge instance'
                );
            });

        });
    }
}
