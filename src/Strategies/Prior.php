<?php
declare(strict_types=1);

namespace Pml\Strategies;

use Pml\Tensor;

/**
 * Imputes by sampling from the empirical prior distribution of observed values.
 * Stores the frequency-weighted discrete CDF and samples via inverse transform.
 */
final class Prior implements Strategy
{
    /** @var float[] */
    private array $values = [];
    /** @var float[] cumulative probabilities */
    private array $cdf    = [];

    public function fit(Tensor $values): void
    {
        $flat = $values->toFlatArray();
        $freq = [];
        foreach ($flat as $v) {
            $key = (string) $v;
            $freq[$key] = ($freq[$key] ?? 0) + 1;
        }
        $total = array_sum($freq);
        $cumulative = 0.0;
        $this->values = [];
        $this->cdf    = [];
        foreach ($freq as $val => $count) {
            $this->values[] = (float) $val;
            $cumulative    += $count / $total;
            $this->cdf[]    = $cumulative;
        }
    }

    public function guess(): float
    {
        if (empty($this->values)) {
            return 0.0;
        }
        $r = mt_rand() / mt_getrandmax();
        foreach ($this->cdf as $i => $prob) {
            if ($r <= $prob) {
                return $this->values[$i];
            }
        }
        return $this->values[array_key_last($this->values)];
    }
}
