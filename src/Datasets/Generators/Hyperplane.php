<?php
declare(strict_types=1);

namespace Pml\Datasets\Generators;

use Pml\Tensor;
use Pml\Dataset;

/**
 * Hyperplane — generates a binary-classification dataset separated by a linear hyperplane.
 * Points are sampled uniformly in [-1, 1]^D; label = sign(w·x + noise).
 *
 * JIT & Memory Optimized:
 * - All sampling and dot products are C-level ops.
 * - Label assignment uses C-level sign() — no PHP loops.
 */
final class Hyperplane implements Generator
{
    private Tensor $normal;

    /**
     * @param int   $dimensions   Feature space dimensionality
     * @param float $noise        Gaussian noise magnitude added to margins
     */
    public function __construct(
        private readonly int   $dimensions = 2,
        private readonly float $noise      = 0.1
    ) {
        // Random unit normal vector in C memory — defines the separating hyperplane
        $w            = Tensor::randomNormal([$this->dimensions], 0.0, 1.0);
        $norm         = $w->std() * sqrt($this->dimensions) + 1e-8;
        $this->normal = $w->mulScalar(1.0 / $norm);
    }

    public function generate(int $n): Dataset
    {
        // X ~ Uniform[-1, 1]^(n×D)
        $x      = Tensor::randomUniform([$n, $this->dimensions], -1.0, 1.0);

        // Margin: m = X * w  [N]
        $margin = $x->matmul($this->normal->expandDims(1))->squeeze();

        // Add Gaussian noise
        if ($this->noise > 0.0) {
            $noiseT = Tensor::randomNormal([$n], 0.0, $this->noise);
            $margin = $margin->add($noiseT);
        }

        // Label: 0 if margin < 0, else 1  (using sign then remap)
        $labels = $margin->greaterScalar(0.0);                        // [N] 0.0 / 1.0

        return new Dataset($x, $labels);
    }
}
