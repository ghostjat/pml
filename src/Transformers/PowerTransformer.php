<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Stateful;
use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Yeo-Johnson Power Transformer — maps features toward normality.
 *
 * Fit finds optimal lambda per feature via 101-step grid search in C (neg log-likelihood).
 * Transform applies the Yeo-Johnson formula column-wise with OpenMP SIMD parallelism.
 * Zero PHP array allocations.
 */
final class PowerTransformer implements Transformer, Stateful
{
    private ?Tensor $lambdas = null;

    public function fit(Dataset $dataset): void
    {
        $this->lambdas = Tensor::yjFit($dataset->samples());
    }

    public function transform(Dataset $dataset): Dataset
    {
        if ($this->lambdas === null) {
            throw new RuntimeException("PowerTransformer has not been fitted.");
        }
        $out = Tensor::yjTransform($dataset->samples(), $this->lambdas);
        return new \Pml\Dataset($out, $dataset->labels());
    }

    public function fitted(): bool
    {
        return $this->lambdas !== null;
    }

    public function getStateDict(string $prefix = ''): array
    {
        return $this->lambdas !== null ? ["{$prefix}lambdas" => $this->lambdas] : [];
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->lambdas = $dict["{$prefix}lambdas"] ?? null;
    }
}
