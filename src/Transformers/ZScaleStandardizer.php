<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Stateful;
use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Z-Scale Standardizer
 * Now updated with a $center parameter to safely handle sparse TF-IDF matrices!
 */
final class ZScaleStandardizer implements Transformer, Stateful
{
    private bool $center;
    private ?Tensor $means = null;
    private ?Tensor $stds  = null;

    /**
     * @param bool $center If false, data is only scaled by std deviation (preserves sparse 0s).
     */
    public function __construct(bool $center = true)
    {
        $this->center = $center;
    }

    public function fit(Dataset $dataset): void
    {
        $x            = $dataset->samples();
        $n            = (float) $x->shape()[0];
        
        $this->means  = $x->meanAxis(0);
        
        // Bessel-corrected variance
        $varBiased    = $x->square()->meanAxis(0)->sub($this->means->square());
        $varUnbiased  = $varBiased->mulScalar($n / max(1.0, $n - 1.0));
        
        // C-Level clipping prevents division by zero
        $this->stds   = $varUnbiased->sqrt()->clip(1e-8, INF);
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new RuntimeException("ZScaleStandardizer has not been fitted.");
        }

        $samples = $dataset->samples();

        // Broadcast-safe: if samples is 2D [N,D] and stats are 1D [D],
        // reshape stats to [1,D] so C broadcast engine sees matching ndim.
        $ndim  = count($samples->shape());
        $means = ($ndim > 1 && count($this->means->shape()) === 1)
            ? $this->means->reshape(1, ...$this->means->shape())
            : $this->means;
        $stds  = ($ndim > 1 && count($this->stds->shape()) === 1)
            ? $this->stds->reshape(1, ...$this->stds->shape())
            : $this->stds;

        if ($this->center) {
            $samples = $samples->sub($means);
        }

        $scaled = $samples->divInplace($stds);

        return new Dataset($scaled, $dataset->labels());
    }

    public function fitted(): bool { return $this->means !== null && $this->stds !== null; }

    public function getStateDict(string $prefix = ''): array
    {
        $dict = [];
        if ($this->means !== null) { $dict[$prefix . 'means'] = $this->means; }
        if ($this->stds  !== null) { $dict[$prefix . 'stds']  = $this->stds; }
        return $dict;
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $rawMeans = $dict[$prefix . 'means'] ?? null;
        $rawStds  = $dict[$prefix . 'stds']  ?? null;
        // copy() converts mmap-backed (read-only) tensors to normal heap tensors
        $this->means = $rawMeans?->copy();
        $this->stds  = $rawStds?->copy();
    }
}