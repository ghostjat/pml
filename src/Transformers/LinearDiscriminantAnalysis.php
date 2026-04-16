<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Stateful;
use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Linear Discriminant Analysis (LDA) — supervised dimensionality reduction.
 * Finds the axes that maximise between-class scatter relative to within-class scatter.
 *
 * JIT & Memory Optimized:
 * - Scatter matrices built via C-level matmul on mean-centred class slices.
 * - Generalised eigendecomposition via LAPACKE (tensor_eigen_sym on S_W^{-1} S_B).
 * - Projection is a single BLAS matmul — zero PHP-loop arithmetic.
 */
final class LinearDiscriminantAnalysis implements Transformer, Stateful
{
    private ?Tensor $W       = null;   // [D × n_components] projection matrix
    private ?Tensor $means   = null;   // overall mean [1 × D]

    public function __construct(private readonly ?int $nComponents = null) {}

    public function fit(Dataset $dataset): void
    {
        $labels = $dataset->labels();
        if ($labels === null) {
            throw new \InvalidArgumentException("LDA requires labeled data.");
        }

        $x       = $dataset->samples();                            // [N × D]
        $d       = $x->shape()[1];
        $flat    = $labels->toFlatArray();
        $classes = array_values(array_unique($flat));
        $k       = count($classes);
        $nComp   = min($this->nComponents ?? ($k - 1), $d, $k - 1);

        $this->means = $x->meanAxis(0);                           // [D]

        // Within-class scatter S_W = sum_c ( X_c - mu_c )^T (X_c - mu_c )
        $SW = Tensor::zeros($d, $d);
        $SB = Tensor::zeros($d, $d);

        $n = $x->shape()[0];
        foreach ($classes as $c) {
            // Boolean mask — C-level
            $cScalar = (float) $c;
            $mask    = $labels->equal(
                Tensor::zeros($n)->addScalarInplace($cScalar)
            );
            $xc      = $x->booleanIndex($mask);                   // [n_c × D]
            $nc      = $xc->shape()[0];
            $muC     = $xc->meanAxis(0);                           // [D]

            // Within-class: (X_c - mu_c)^T (X_c - mu_c)
            $centered = $xc->sub($muC->expandDims(0));
            $SW->addInplace($centered->transpose()->matmul($centered));

            // Between-class: n_c * (mu_c - mu)(mu_c - mu)^T
            $diff = $muC->sub($this->means)->expandDims(0);       // [1 × D]
            $SB->addInplace($diff->transpose()->matmul($diff)->mulScalarInplace((float) $nc));
        }

        // S_W^{-1} S_B
        $SWinv = $SW->inverse();
        $M     = $SWinv->matmul($SB);

        // Eigen decomposition — ascending eigenvalues
        ['values' => $vals, 'vectors' => $vecs] = $M->eigenSym();

        // Take the top nComp eigenvectors (last columns — ascending sort)
        $totalVecs = $vecs->shape()[1];
        $startCol  = max(0, $totalVecs - $nComp);

        // Slice columns [startCol .. end] — zero-copy view
        $idxT     = Tensor::fromArray(range($startCol, $totalVecs - 1));
        $this->W  = $vecs->take($idxT, 1);                        // [D × nComp]
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("LDA has not been fitted.");
        }
        $projected = $dataset->samples()->matmul($this->W);       // [N × nComp]
        return new Dataset($projected, $dataset->labels());
    }

    public function fitted(): bool { return $this->W !== null; }

    public function getStateDict(string $prefix = ''): array
    {
        $dict = [];
        if ($this->W     !== null) { $dict[$prefix . 'W']     = $this->W; }
        if ($this->means !== null) { $dict[$prefix . 'means'] = $this->means; }
        return $dict;
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->W     = $dict[$prefix . 'W']     ?? null;
        $this->means = $dict[$prefix . 'means'] ?? null;
    }
}
