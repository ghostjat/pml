<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Stateful;
use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Quantile Transformer — maps each feature to a uniform [0, 1] distribution.
 *
 * Fit computes [D, n_quantiles] landmark matrix in C (single qsort per feature).
 * Transform applies binary-search interpolation in C with OpenMP parallelism.
 * Zero PHP array allocations after fit.
 */
final class QuantileTransformer implements Transformer, Stateful
{
    private ?Tensor $landmarks = null;

    public function __construct(private readonly int $nQuantiles = 1000) {}

    public function fit(Dataset $dataset): void
    {
        $this->landmarks = Tensor::quantileFit($dataset->samples(), $this->nQuantiles);
    }

    public function transform(Dataset $dataset): Dataset
    {
        if ($this->landmarks === null) {
            throw new RuntimeException("QuantileTransformer has not been fitted.");
        }
        $out = Tensor::quantileTransform($dataset->samples(), $this->landmarks);
        return new \Pml\Dataset($out, $dataset->labels());
    }

    public function fitted(): bool
    {
        return $this->landmarks !== null;
    }

    public function getStateDict(string $prefix = ''): array
    {
        return $this->landmarks !== null ? ["{$prefix}landmarks" => $this->landmarks] : [];
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->landmarks = $dict["{$prefix}landmarks"] ?? null;
    }
}
