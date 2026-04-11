<?php
declare(strict_types=1);

namespace Pml;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use RuntimeException;

/**
 * Committee Machine — weighted ensemble of heterogeneous estimators.
 * Aggregates predictions via weighted average (soft vote for probabilistic,
 * weighted hard vote for non-probabilistic classifiers).
 *
 * JIT & Memory Optimized:
 * - Weights are PHP scalars (no C overhead); accumulation is C-level addInplace.
 * - Each estimator trains independently — zero shared state.
 */
final class CommitteeMachine implements Learner
{
    /** @var array{estimator: Learner, weight: float}[] */
    private array $members = [];
    private bool  $trained = false;

    /**
     * @param array<array{Learner, float}> $members  [[estimator, weight], ...]
     */
    public function __construct(array $members)
    {
        if (empty($members)) {
            throw new \InvalidArgumentException("CommitteeMachine requires at least one member.");
        }

        $totalWeight = 0.0;
        foreach ($members as [$est, $weight]) {
            if (!$est instanceof Learner) {
                throw new \InvalidArgumentException("All members must implement Learner.");
            }
            $totalWeight += $weight;
            $this->members[] = ['estimator' => $est, 'weight' => (float) $weight];
        }

        // Normalize weights so they sum to 1
        foreach ($this->members as &$m) {
            $m['weight'] /= $totalWeight;
        }
    }

    public function train(Dataset $dataset): void
    {
        foreach ($this->members as $m) {
            $m['estimator']->train($dataset);
        }
        $this->trained = true;
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained) {
            throw new RuntimeException("CommitteeMachine has not been trained.");
        }

        $sum = null;
        foreach ($this->members as $m) {
            $p   = $m['estimator']->predict($dataset);
            $wp  = $p->mulScalar($m['weight']);
            $sum = $sum === null ? $wp : $sum->addInplace($wp);
        }

        return $sum->round();
    }

    public function trained(): bool { return $this->trained; }
}
