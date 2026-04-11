<?php

declare(strict_types=1);

namespace Pml\Datasets\Generators;

use Pml\Tensor;
use Pml\Dataset;

/**
 * Swiss Roll Generator.
 * Generates a 3D dataset shaped like a Swiss roll, a classic test for Manifold Learning (e.g., t-SNE, UMAP).
 * * JIT & Memory Optimized:
 * - Employs chained in-place memory mutations to generate the complex 3D spiral natively.
 * - Prevents massive GC spikes by isolating mathematical temporaries in C-pointers.
 */
final class SwissRoll implements Generator
{
    private float $noise;

    public function __construct(float $noise = 0.0)
    {
        $this->noise = $noise;
    }

    public function generate(int $n): Dataset
    {
        // t = 1.5 * pi * (1 + 2 * random) -> Range [1.5pi, 4.5pi]
        $t = Tensor::randomUniform([$n, 1], 0.0, 1.0)
            ->mulScalarInplace(2.0)
            ->addScalarInplace(1.0)
            ->mulScalarInplace(1.5 * M_PI);

        // X = t * cos(t)
        $tCos = $t->copy()->cos();
        $x = $t->copy()->mulInplace($tCos);

        // Y = random[0, 21] (The height/width of the roll)
        $y = Tensor::randomUniform([$n, 1], 0.0, 21.0);

        // Z = t * sin(t)
        $tSin = $t->copy()->sin();
        $z = $t->copy()->mulInplace($tSin);

        // Pack into [N, 3]
        $samples = Tensor::concat([$x, $y, $z], 1);

        // Inject hardware noise
        if ($this->noise > 0.0) {
            $noiseT = Tensor::randomNormal([$n, 3], 0.0, $this->noise);
            $samples->addInplace($noiseT);
        }

        // Return the continuous manifold parameter (t) as the target label for regression tasks
        return new Dataset($samples, $t->squeeze());
    }
}