<?php

declare(strict_types=1);

namespace Pml\Datasets\Generators;

use Pml\Tensor;
use Pml\Dataset;

/**
 * Gaussian Blob Generator.
 * Generates an isotropic Gaussian blob for clustering and classification tasks.
 * * JIT & Memory Optimized:
 * - Employs pure C-Level `randomNormal()` to generate memory blocks instantly.
 * - Broadcasts the center offset natively via AVX2 without loops.
 */
final class Blob implements Generator
{
    private array $center;
    private float $stddev;

    public function __construct(array $center = [0.0, 0.0], float $stddev = 1.0)
    {
        $this->center = $center;
        $this->stddev = $stddev;
    }

    public function generate(int $n): Dataset
    {
        $dims = count($this->center);
        
        // Generate raw Gaussian distribution [N, D] natively in C
        $samples = Tensor::randomNormal([$n, $dims], 0.0, $this->stddev);
        
        // Extract zero-copy view of the center offset [1, D]
        $centerT = Tensor::fromArray([$this->center]);
        
        // Broadcast the offset across the entire matrix instantly
        $samples->addInplace($centerT);

        return new Dataset($samples);
    }
}