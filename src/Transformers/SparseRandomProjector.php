<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Sparse Random Projection — Achlioptas (2003) ±1/sqrt(s) sparse projection matrix.
 * Memory-efficient alternative to Gaussian RP; density defaults to 1/sqrt(D).
 *
 * JIT & Memory Optimized:
 * - Sparse matrix constructed as a dense Tensor in C (suitable for moderate D).
 * - Transform is a single BLAS matmul.
 */
final class SparseRandomProjector implements Transformer
{
    private ?Tensor $R   = null;
    private ?float  $density = null;

    public function __construct(
        private readonly int    $nComponents = 100,
        private readonly ?float $densityHint = null   // null = auto: 1/sqrt(D)
    ) {}

    public function fit(Dataset $dataset): void
    {
        $d       = $dataset->numColumns();
        $density = $this->densityHint ?? (1.0 / sqrt((float) $d));
        $density = max(1e-4, min(1.0, $density));
        $scale   = sqrt(1.0 / ($density * $this->nComponents));

        // Build dense matrix filled with 0, then assign ±scale with probability density
        $flat = [];
        for ($i = 0; $i < $d * $this->nComponents; $i++) {
            $r = mt_rand() / mt_getrandmax();
            if ($r < $density / 2.0) {
                $flat[] = $scale;
            } elseif ($r < $density) {
                $flat[] = -$scale;
            } else {
                $flat[] = 0.0;
            }
        }

        $this->R = Tensor::fromArray(array_chunk($flat, $this->nComponents));
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("SparseRandomProjector has not been fitted.");
        }
        return new Dataset(
            $dataset->samples()->matmul($this->R),
            $dataset->labels()
        );
    }

    public function fitted(): bool { return $this->R !== null; }
}
