<?php

declare(strict_types=1);

namespace Pml\Datasets\Generators;

use Pml\Tensor;
use Pml\Dataset;

/**
 * Half Moon Generator.
 * Generates two interleaving half circles, an excellent dataset to test non-linear classifiers.
 * * JIT & Memory Optimized:
 * - 100% Vectorized geometric generation.
 * - C-level trigonometric broadcasting (`sin`, `cos`) over thousands of points in microseconds.
 */
final class HalfMoon implements Generator
{
    private float $noise;

    public function __construct(float $noise = 0.1)
    {
        $this->noise = $noise;
    }

    public function generate(int $n): Dataset
    {
        $nOut = (int) ceil($n / 2);
        $nIn = $n - $nOut;

        // 1. Outer Moon (Class 0)
        $angleOut = Tensor::randomUniform([$nOut, 1], 0.0, M_PI);
        $xOut = $angleOut->copy()->cos();
        $yOut = $angleOut->copy()->sin();
        $labelsOut = Tensor::zeros($nOut, 1);

        // 2. Inner Moon (Class 1) - Shifted geometry
        $angleIn = Tensor::randomUniform([$nIn, 1], 0.0, M_PI);
        $xIn = $angleIn->copy()->cos()->mulScalarInplace(-1.0)->addScalarInplace(1.0);
        $yIn = $angleIn->copy()->sin()->mulScalarInplace(-1.0)->addScalarInplace(0.5);
        $labelsIn = Tensor::ones($nIn, 1);

        // 3. Hardware memory concatenation [N, 1]
        $x = Tensor::concat([$xOut, $xIn], 0);
        $y = Tensor::concat([$yOut, $yIn], 0);
        
        // Pack into final shape [N, 2]
        $samples = Tensor::concat([$x, $y], 1);
        $labels = Tensor::concat([$labelsOut, $labelsIn], 0)->squeeze();

        // 4. Inject Gaussian Noise natively
        if ($this->noise > 0.0) {
            $noise = Tensor::randomNormal([$n, 2], 0.0, $this->noise);
            $samples->addInplace($noise);
        }

        return new Dataset($samples, $labels);
    }
}