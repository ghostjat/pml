<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Interval Discretizer — bins each continuous feature into K equal-width intervals.
 * Replaces float values with integer bin indices [0, K-1].
 *
 * JIT & Memory Optimized:
 * - Min/max per column computed via single C-level reduction.
 * - Bin assignment uses C-level clip + floor arithmetic — no PHP loops over rows.
 */
final class IntervalDiscretizer implements Transformer
{
    private ?Tensor $mins   = null;   // [D]
    private ?Tensor $widths = null;   // [D]  = (max-min)/bins

    public function __construct(private readonly int $bins = 5) {}

    public function fit(Dataset $dataset): void
    {
        $x           = $dataset->samples();
        $this->mins  = $x->minAxis(0);
        $maxes       = $x->maxAxis(0);
        $range       = $maxes->sub($this->mins)->clip(1e-10, INF);
        $this->widths = $range->mulScalar(1.0 / $this->bins);
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("IntervalDiscretizer has not been fitted.");
        }

        $x   = $dataset->samples();                                // [N × D]
        // bin = floor( (x - min) / width ), clamped to [0, bins-1]
        $bin = $x->sub($this->mins)
                  ->div($this->widths)
                  ->floor()
                  ->clip(0.0, (float)($this->bins - 1));

        return new Dataset($bin, $dataset->labels());
    }

    public function fitted(): bool { return $this->mins !== null; }
}
