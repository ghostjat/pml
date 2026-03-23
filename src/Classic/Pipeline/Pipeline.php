<?php

declare(strict_types=1);

namespace Pml\Classic\Pipeline;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor, Transformer};

// ═══════════════════════════════════════════════════════════════════════════
//  Pipeline — sklearn.pipeline.Pipeline
//
//  Chains a sequence of transformers followed by a final estimator.
//  Intermediate steps must implement Transformer (fit_transform / transform).
//  The final step must implement Estimator and optionally Predictor.
//
//  ── Fit ─────────────────────────────────────────────────────────────────
//
//  For steps 0 … N-2 (intermediate transformers):
//    X = step.fit_transform(X, y)     — fits and transforms in one pass
//
//  For step N-1 (final estimator):
//    step.fit(X, y)                   — uses the transformed X from step N-2
//
//  ── Predict ─────────────────────────────────────────────────────────────
//
//  For steps 0 … N-2:
//    X = step.transform(X)            — must be already fitted
//
//  For step N-1:
//    return step.predict(X)           — or predict_proba / transform
//
//  ── Step Format ─────────────────────────────────────────────────────────
//
//  Steps are passed as an array of 2-element tuples: [string $name, object $estimator].
//  Names must be unique (sklearn constraint).
//  Use make_pipeline(...$estimators) for auto-naming.
//
//  ── Named Access ─────────────────────────────────────────────────────────
//
//  pipeline['scaler']      → access step by name (via ArrayAccess)
//  pipeline->named_steps   → associative array [name => estimator]
// ═══════════════════════════════════════════════════════════════════════════

final class Pipeline implements Estimator, Predictor, Transformer
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Associative map name → fitted estimator (mirrors sklearn's named_steps).
     * Populated after fit() — intermediate values are the fitted transformers.
     *
     * @var array<string, object>
     */
    public readonly array $named_steps;

    // ── Internal state ────────────────────────────────────────────────────

    /**
     * Ordered steps: [[name, estimator], ...].  Stored as received.
     *
     * @var array<int, array{0: string, 1: object}>
     */
    private readonly array $steps;

    /**
     * Ordered step names (parallel to $steps for O(1) lookup).
     * @var string[]
     */
    private readonly array $stepNames;

    /**
     * @param array<int, array{0: string, 1: object}> $steps
     *   Ordered list of [name, estimator] tuples.
     *   All intermediate steps must implement Transformer.
     *   The final step must implement Estimator.
     */
    public function __construct(array $steps)
    {
        if (count($steps) === 0) {
            throw new \InvalidArgumentException('Pipeline: steps must be non-empty.');
        }

        // Validate: each entry must be a 2-element [name, object] tuple
        $names = [];
        foreach ($steps as $i => $step) {
            if (!is_array($step) || count($step) < 2 || !is_string($step[0]) || !is_object($step[1])) {
                throw new \InvalidArgumentException(
                    "Pipeline: step {$i} must be a [string, object] tuple."
                );
            }
            $name = $step[0];
            if (isset($names[$name])) {
                throw new \InvalidArgumentException(
                    "Pipeline: duplicate step name '{$name}'."
                );
            }
            $names[$name] = true;
        }

        // Validate intermediate steps implement Transformer
        $lastIdx = count($steps) - 1;
        for ($i = 0; $i < $lastIdx; $i++) {
            if (!($steps[$i][1] instanceof Transformer)) {
                $cls = get_class($steps[$i][1]);
                throw new \InvalidArgumentException(
                    "Pipeline: intermediate step '{$steps[$i][0]}' ({$cls}) must implement Transformer."
                );
            }
        }

        // Final step must implement Estimator
        if (!($steps[$lastIdx][1] instanceof Estimator)) {
            $cls = get_class($steps[$lastIdx][1]);
            throw new \InvalidArgumentException(
                "Pipeline: final step '{$steps[$lastIdx][0]}' ({$cls}) must implement Estimator."
            );
        }

        $this->steps     = $steps;
        $this->stepNames = array_column($steps, 0);
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Fit all steps sequentially.
     *
     * Intermediate transformers are fitted and transform X in place.
     * The final estimator is fitted on the transformed X.
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Target vector [n_samples] (passed to every step)
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        $lastIdx = count($this->steps) - 1;

        // ── Fit-transform all intermediate steps ───────────────────────────
        for ($i = 0; $i < $lastIdx; $i++) {
            /** @var Transformer $transformer */
            $transformer = $this->steps[$i][1];
            // fit_transform = fit(X, y) → transform(X): may be more efficient than two calls
            $X = $transformer->fit_transform($X, $y);
        }

        // ── Fit the final estimator on the transformed X ───────────────────
        $this->steps[$lastIdx][1]->fit($X, $y);

        // ── Build named_steps map for introspection ────────────────────────
        $named = [];
        foreach ($this->steps as [$name, $estimator]) {
            $named[$name] = $estimator;
        }
        $this->named_steps = $named;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Transform through intermediate steps then predict with the final step.
     *
     * @param Tensor $X  Feature matrix [n_samples, n_features]
     * @return Tensor    Predictions [n_samples]
     */
    public function predict(Tensor $X): Tensor
    {
        $this->checkFitted();

        $X = $this->transformIntermediateSteps($X);

        $final = $this->steps[count($this->steps) - 1][1];
        if (!($final instanceof Predictor)) {
            $cls = get_class($final);
            throw new \RuntimeException(
                "Pipeline::predict() — final step '{$cls}' does not implement Predictor."
            );
        }

        return $final->predict($X);
    }

    /**
     * Transform through all intermediate steps then call predict_proba on the final step.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples, n_classes]
     */
    public function predict_proba(Tensor $X): Tensor
    {
        $this->checkFitted();

        $X     = $this->transformIntermediateSteps($X);
        $final = $this->steps[count($this->steps) - 1][1];

        if (!method_exists($final, 'predict_proba')) {
            $cls = get_class($final);
            throw new \RuntimeException(
                "Pipeline::predict_proba() — final step '{$cls}' has no predict_proba() method."
            );
        }

        return $final->predict_proba($X);
    }

    // ── Transformer ────────────────────────────────────────────────────────

    /**
     * Transform through ALL steps (used when the pipeline itself is an intermediate step,
     * or when the final step is a Transformer).
     *
     * @param Tensor $X  [n_samples, n_features_in]
     * @return Tensor    [n_samples, n_features_out]
     */
    public function transform(Tensor $X): Tensor
    {
        $this->checkFitted();

        $lastIdx = count($this->steps) - 1;

        // Run intermediate transformers
        $X = $this->transformIntermediateSteps($X);

        // Run final step if it is a Transformer
        $final = $this->steps[$lastIdx][1];
        if ($final instanceof Transformer) {
            $X = $final->transform($X);
        } else {
            $cls = get_class($final);
            throw new \RuntimeException(
                "Pipeline::transform() — final step '{$cls}' does not implement Transformer."
            );
        }

        return $X;
    }

    /**
     * Fit then transform.  Equivalent to fit($X, $y)->transform($X).
     */
    public function fit_transform(Tensor $X, ?Tensor $y = null): Tensor
    {
        return $this->fit($X, $y)->transform($X);
    }

    // ── Scoring ────────────────────────────────────────────────────────────

    /**
     * Delegate to the final step's score() method (if available).
     */
    public function score(Tensor $X, Tensor $y): float
    {
        $this->checkFitted();

        $X     = $this->transformIntermediateSteps($X);
        $final = $this->steps[count($this->steps) - 1][1];

        if (!method_exists($final, 'score')) {
            $cls = get_class($final);
            throw new \RuntimeException(
                "Pipeline::score() — final step '{$cls}' has no score() method."
            );
        }

        return $final->score($X, $y);
    }

    // ── Named step access ─────────────────────────────────────────────────

    /**
     * Get a fitted step by name.
     *
     * @return object  The estimator/transformer for that step.
     */
    public function getStep(string $name): object
    {
        $this->checkFitted();
        if (!isset($this->named_steps[$name])) {
            throw new \InvalidArgumentException("Pipeline: no step named '{$name}'.");
        }
        return $this->named_steps[$name];
    }

    /**
     * Expose the ordered steps list as-is.
     *
     * @return array<int, array{0: string, 1: object}>
     */
    public function getSteps(): array
    {
        return $this->steps;
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /**
     * Pass X through all intermediate steps (index 0 … N-2) via transform().
     * The final step is NOT called here.
     */
    private function transformIntermediateSteps(Tensor $X): Tensor
    {
        $lastIdx = count($this->steps) - 1;
        for ($i = 0; $i < $lastIdx; $i++) {
            /** @var Transformer $transformer */
            $transformer = $this->steps[$i][1];
            $X = $transformer->transform($X);
        }
        return $X;
    }

    private function checkFitted(): void
    {
        if (!isset($this->named_steps)) {
            throw new \RuntimeException('Pipeline is not fitted. Call fit() first.');
        }
    }
}
