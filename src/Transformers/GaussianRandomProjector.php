<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Gaussian Random Projection — Johnson-Lindenstrauss dimensionality reduction.
 * Projects X @ R where R ~ N(0, 1/nComponents) — preserves pairwise distances.
 *
 * JIT & Memory Optimized:
 * - Projection matrix R is allocated once in C and reused for all transforms.
 * - Transform is a single BLAS matmul — no PHP-loop arithmetic.
 */
final class GaussianRandomProjector implements Transformer
{
    private ?Tensor $R = null;    // [D × nComponents]

    public function __construct(private readonly int $nComponents = 100) {}

    public function fit(Dataset $dataset): void
    {
        $d       = $dataset->numColumns();
        $scale   = 1.0 / sqrt((float) $this->nComponents);
        $this->R = Tensor::randomNormal([$d, $this->nComponents], 0.0, $scale);
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("GaussianRandomProjector has not been fitted.");
        }
        return new Dataset(
            $dataset->samples()->matmul($this->R),                // [N × nComponents]
            $dataset->labels()
        );
    }

    public function fitted(): bool { return $this->R !== null; }
}
