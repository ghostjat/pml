<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Truncated SVD (LSA) — projects data onto the top-K right singular vectors.
 * Unlike PCA, does NOT centre the data first — suitable for sparse/TF-IDF matrices.
 *
 * JIT & Memory Optimized:
 * - Full SVD via LAPACKE (tensor_svd); only the Vt slice is retained.
 * - Transform is a single BLAS matmul — zero PHP arithmetic.
 */
final class TruncatedSVD implements Transformer
{
    private ?Tensor $Vt = null;   // [nComponents × D] right singular vectors

    public function __construct(private readonly int $nComponents = 2) {}

    public function fit(Dataset $dataset): void
    {
        $x = $dataset->samples();                                  // [N × D]

        ['U' => $U, 'S' => $S, 'Vt' => $Vt] = $x->svd();

        // Keep only top nComponents rows of Vt — zero-copy slice
        $nComp    = min($this->nComponents, $Vt->shape()[0]);
        $this->Vt = $Vt->slice(0, 0, $nComp);                    // [nComp × D]
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("TruncatedSVD has not been fitted.");
        }
        // X @ Vt^T  →  [N × nComp]
        $projected = $dataset->samples()->matmul($this->Vt->transpose());
        return new Dataset($projected, $dataset->labels());
    }

    public function fitted(): bool { return $this->Vt !== null; }
}
