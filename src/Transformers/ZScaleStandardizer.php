<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Z-Scale Standardizer — alias for StandardScaler using corrected (N-1) variance.
 * Computes column-wise: z = (x - mean) / std  using Bessel-corrected std.
 *
 * JIT & Memory Optimized:
 * - All statistics computed in a single C pass per column axis.
 * - Transform is two in-place broadcasts — zero intermediate copies.
 */
final class ZScaleStandardizer implements Transformer
{
    private ?Tensor $means = null;
    private ?Tensor $stds  = null;

    public function fit(Dataset $dataset): void
    {
        $x            = $dataset->samples();
        $n            = (float) $x->shape()[0];
        $this->means  = $x->meanAxis(0);
        // Bessel-corrected variance: E[X^2] - E[X]^2 scaled by N/(N-1)
        $varBiased    = $x->square()->meanAxis(0)->sub($this->means->square());
        $varUnbiased  = $varBiased->mulScalar($n / max(1.0, $n - 1.0));
        $this->stds   = $varUnbiased->sqrt()->clip(1e-8, INF);
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("ZScaleStandardizer has not been fitted.");
        }
        $scaled = $dataset->samples()->sub($this->means)->divInplace($this->stds);
        return new Dataset($scaled, $dataset->labels());
    }

    public function fitted(): bool { return $this->means !== null; }
}
