<?php

declare(strict_types=1);

namespace Pml\Classic\ModelSelection;

use Pml\Tensor;
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  GridSearchCV — sklearn.model_selection.GridSearchCV
//
//  Exhaustive search over a parameter grid using K-Fold cross-validation.
//  Mirrors sklearn's interface: constructor takes the estimator and param_grid,
//  fit() discovers the best parameters and re-fits on the full dataset.
//
//  ── Parameter Grid ───────────────────────────────────────────────────────
//
//  $param_grid is an associative array: ['param_name' => [val1, val2, ...]].
//  GridSearchCV computes the full Cartesian product of all parameter lists,
//  evaluating every combination via cross_val_score.
//
//  Example:
//    $grid = ['max_depth' => [3, 5, 10], 'min_samples_split' => [2, 5]];
//    → 6 combinations: {3,2}, {3,5}, {5,2}, {5,5}, {10,2}, {10,5}
//
//  ── Estimator Cloning ────────────────────────────────────────────────────
//
//  For each parameter combination, a fresh estimator clone is built via
//  Reflection: constructor parameters are read from the base estimator's
//  properties and overridden with the grid combination values.
//  This mirrors sklearn's clone() + set_params() pipeline.
//
//  ── Evaluation ───────────────────────────────────────────────────────────
//
//  Each combination is scored via Validation::cross_val_score() using the
//  provided $cv (int → KFold) and $scoring string.  The mean of per-fold
//  scores is the combination's CV score.
//
//  ── Best Estimator ───────────────────────────────────────────────────────
//
//  After evaluating all combinations, a fresh estimator is cloned with the
//  best parameters and fitted on the ENTIRE (X, y) dataset.  This matches
//  sklearn's refit=True behaviour (the default).
//
//  ── cv_results_ ──────────────────────────────────────────────────────────
//
//  A PHP array of associative rows, one per parameter combination:
//    ['params' => [...], 'mean_test_score' => float, 'std_test_score' => float,
//     'split_scores' => float[]]
//  Mirrors sklearn's cv_results_ dict-of-arrays structure.
// ═══════════════════════════════════════════════════════════════════════════

final class GridSearchCV implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Best estimator: a fresh clone of $estimator fitted on the FULL (X, y)
     * with the parameters that achieved the highest cross-validation score.
     */
    public readonly object $best_estimator_;

    /**
     * Parameter combination that produced the best CV score.
     * Associative array: ['param_name' => value, ...]
     *
     * @var array<string, mixed>
     */
    public readonly array $best_params_;

    /** Mean CV score of the best parameter combination. */
    public readonly float $best_score_;

    /**
     * Results for every combination, sorted by rank.
     * Each entry: ['params' => [...], 'mean_test_score' => float,
     *              'std_test_score' => float, 'split_scores' => float[]].
     *
     * @var array<int, array<string, mixed>>
     */
    public readonly array $cv_results_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param object              $estimator   An UNFITTED estimator.
     * @param array<string,array> $param_grid  Associative grid of parameters.
     * @param int|KFold           $cv          Number of folds or a KFold instance.
     * @param string|callable     $scoring     Metric name or callable.
     *                                         Passed to cross_val_score().
     */
    public function __construct(
        private readonly object    $estimator,
        private readonly array     $param_grid,
        private readonly int|KFold $cv      = 5,
        /** @var string|callable */ private readonly mixed $scoring = 'accuracy',
    ) {
        if (empty($param_grid)) {
            throw new \InvalidArgumentException('GridSearchCV: param_grid must not be empty.');
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Run exhaustive grid search with cross-validation.
     *
     * For each parameter combination:
     *   1. Clone the base estimator with those parameters.
     *   2. Evaluate via cross_val_score → mean CV score.
     *   3. Track the best-scoring combination.
     *
     * After all combinations are evaluated, fit a final clone with the
     * best parameters on the ENTIRE dataset (X, y).
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Target vector  [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        // ── Generate all parameter combinations (Cartesian product) ────
        $combinations = $this->cartesianProduct($this->param_grid);
        $nCombos      = count($combinations);

        if ($nCombos === 0) {
            throw new \RuntimeException('GridSearchCV: param_grid produced zero combinations.');
        }

        $results   = [];
        $bestScore = -INF;
        $bestIdx   = 0;

        // ── Evaluate each combination ──────────────────────────────────
        foreach ($combinations as $i => $params) {
            // Build a clone with this parameter combination
            $est = $this->buildEstimator($params);

            // cross_val_score handles cloning internally per fold
            $scores    = Validation::cross_val_score($est, $X, $y, $this->cv, $this->scoring);
            $meanScore = count($scores) > 0 ? array_sum($scores) / count($scores) : 0.0;

            // Compute standard deviation of fold scores for cv_results_
            $stdScore = 0.0;
            if (count($scores) > 1) {
                $variance = 0.0;
                foreach ($scores as $s) {
                    $variance += ($s - $meanScore) * ($s - $meanScore);
                }
                $stdScore = sqrt($variance / count($scores));
            }

            $results[$i] = [
                'params'          => $params,
                'mean_test_score' => $meanScore,
                'std_test_score'  => $stdScore,
                'split_scores'    => $scores,
            ];

            if ($meanScore > $bestScore) {
                $bestScore = $meanScore;
                $bestIdx   = $i;
            }
        }

        // ── Sort cv_results_ by descending mean score (rank_test_score) ─
        usort($results, fn($a, $b) => $b['mean_test_score'] <=> $a['mean_test_score']);
        $this->cv_results_ = $results;

        // ── Refit best estimator on the full dataset ───────────────────
        //
        // sklearn's refit=True: after CV, create a fresh estimator with the
        // best params and fit it on all of (X, y).  This is what is returned
        // by predict() and is accessible via best_estimator_.
        $this->best_params_    = $combinations[$bestIdx];
        $this->best_score_     = $bestScore;
        $bestEst               = $this->buildEstimator($this->best_params_);
        $bestEst->fit($X, $y);
        $this->best_estimator_ = $bestEst;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Delegate to best_estimator_->predict().
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples]
     */
    public function predict(Tensor $X): Tensor
    {
        if (!isset($this->best_estimator_)) {
            throw new \RuntimeException('GridSearchCV is not fitted. Call fit() first.');
        }
        if (!($this->best_estimator_ instanceof Predictor)) {
            throw new \RuntimeException(
                'GridSearchCV: best_estimator_ does not implement Predictor.'
            );
        }
        return $this->best_estimator_->predict($X);
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /**
     * Compute the Cartesian product of the parameter grid.
     *
     * Starting from [[]]] (one empty combination), each parameter's value
     * list is "exploded" — every existing combination is duplicated once per
     * value, with the new (param, value) pair appended.
     *
     * Example: {'a': [1,2], 'b': [10,20]}
     *   Start:   [[]]
     *   After a: [[a=1], [a=2]]
     *   After b: [[a=1,b=10], [a=1,b=20], [a=2,b=10], [a=2,b=20]]
     *
     * @param array<string, array> $paramGrid
     * @return array<int, array<string, mixed>>
     */
    private function cartesianProduct(array $paramGrid): array
    {
        $combinations = [[]];

        foreach ($paramGrid as $param => $values) {
            $expanded = [];
            foreach ($combinations as $existing) {
                foreach ($values as $value) {
                    $expanded[] = array_merge($existing, [$param => $value]);
                }
            }
            $combinations = $expanded;
        }

        return $combinations;
    }

    /**
     * Build an estimator clone with specific parameter values overridden.
     *
     * Algorithm (mirrors sklearn's clone() + set_params()):
     *   1. Inspect the base estimator's constructor parameters via Reflection.
     *   2. For each constructor parameter:
     *        a. If it appears in $params, use the grid value.
     *        b. Otherwise, read the current value from the base estimator's
     *           same-named property (constructor-promoted readonly pattern).
     *   3. Instantiate a fresh object via ReflectionClass::newInstance().
     *
     * This ensures the clone starts unfitted with identical hyperparameters
     * to the base estimator except for the overridden grid values.
     *
     * @param array<string, mixed> $params  Parameter overrides from the grid.
     * @return Estimator                    Fresh unfitted estimator clone.
     */
    private function buildEstimator(array $params): object
    {
        $rc   = new \ReflectionClass($this->estimator);
        $ctor = $rc->getConstructor();

        if ($ctor === null) {
            return $rc->newInstance();
        }

        $args = [];
        foreach ($ctor->getParameters() as $param) {
            $name = $param->getName();

            if (array_key_exists($name, $params)) {
                // ── Override from the grid ─────────────────────────────
                $args[] = $params[$name];
            } else {
                // ── Preserve from base estimator ───────────────────────
                try {
                    $prop = $rc->getProperty($name);
                    $prop->setAccessible(true);
                    if ($prop->isInitialized($this->estimator)) {
                        $args[] = $prop->getValue($this->estimator);
                    } elseif ($param->isDefaultValueAvailable()) {
                        $args[] = $param->getDefaultValue();
                    } else {
                        $args[] = null;
                    }
                } catch (\ReflectionException) {
                    $args[] = $param->isDefaultValueAvailable()
                        ? $param->getDefaultValue()
                        : null;
                }
            }
        }

        return $rc->newInstance(...$args);
    }
}
