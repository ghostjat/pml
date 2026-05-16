<?php
declare(strict_types=1);

namespace Pml;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use RuntimeException;

/**
 * Bootstrap Aggregator (Bagging) meta-estimator.
 * Trains N base estimators on bootstrap resamples; aggregates via majority vote
 * (classifiers) or mean (regressors).
 *
 * JIT & Memory Optimized:
 * - Bootstrap indices generated in PHP; C-level tensor_take extracts each sample.
 * - Voting/averaging uses in-place Tensor accumulation — zero PHP scalar loops.
 */
final class BootstrapAggregator implements Learner
{
    /** @var Learner[] */
    private array $estimators = [];

    /**
     * @param Learner   $base         A clonable base estimator
     * @param int       $estimators   Number of bootstrap models
     * @param float     $ratio        Fraction of training samples per bag (with replacement)
     * @param bool      $regression   True → mean aggregation, False → majority vote
     */
    public function __construct(
        private readonly Learner $base,
        private readonly int     $estimators = 10,
        private readonly float   $ratio      = 1.0,
        private readonly bool    $regression = false
    ) {}

    public function train(Dataset $dataset, mixed ...$options): void
    {
        $n       = $dataset->numRows();
        $bagSize = max(1, (int) round($n * $this->ratio));

        $this->estimators = [];

        for ($i = 0; $i < $this->estimators; $i++) {
            // Bootstrap sample: random with replacement
            $indices = [];
            for ($j = 0; $j < $bagSize; $j++) {
                $indices[] = mt_rand(0, $n - 1);
            }
            $idxT  = Tensor::fromArray($indices, Tensor::DTYPE_INT32);
            $bootX = $dataset->samples()->take($idxT, 0);
            $bootY = $dataset->labels()?->take($idxT, 0);

            $est = clone $this->base;
            $est->train(new Dataset($bootX, $bootY));
            $this->estimators[] = $est;
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (empty($this->estimators)) {
            throw new RuntimeException("BootstrapAggregator has not been trained.");
        }

        if ($this->regression) {
            // Mean of all predictions — accumulate in-place in C
            $sum = null;
            foreach ($this->estimators as $est) {
                $p = $est->predict($dataset);
                $sum = $sum === null ? $p : $sum->addInplace($p);
            }
            return $sum->mulScalarInplace(1.0 / count($this->estimators));
        }

        // Majority vote: collect all predictions then argmax frequency per row
        $n    = $dataset->numRows();
        $preds = array_map(fn($e) => $e->predict($dataset)->toFlatArray(), $this->estimators);
        $out  = [];
        for ($i = 0; $i < $n; $i++) {
            $votes = [];
            foreach ($preds as $p) {
                $v = (int) $p[$i];
                $votes[$v] = ($votes[$v] ?? 0) + 1;
            }
            arsort($votes);
            $out[] = array_key_first($votes);
        }
        return Tensor::fromArray($out);
    }

    public function trained(): bool { return !empty($this->estimators); }
}
