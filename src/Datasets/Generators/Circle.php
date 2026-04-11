<?php

declare(strict_types=1);

namespace Pml\Datasets\Generators;

use Pml\Tensor;
use Pml\Dataset;

/**
 * Concentric Circles Generator.
 * Generates a 2D dataset of a large circle containing a smaller inner circle.
 * Perfect for testing non-linear kernels like RBF SVMs or DBSCAN.
 * * JIT & Memory Optimized:
 * - 100% Vectorized geometric generation natively in C using AVX2 `sin()` and `cos()`.
 */
final class Circle implements Generator
{
    private float $noise;
    private float $factor;

    /**
     * @param float $noise Standard deviation of Gaussian noise added to the data.
     * @param float $factor Scale factor between inner and outer circle (0.0 to 1.0).
     */
    public function __construct(float $noise = 0.05, float $factor = 0.5)
    {
        $this->noise = $noise;
        $this->factor = $factor;
    }

    public function generate(int $n): Dataset
    {
        $nOut = (int) ceil($n / 2);
        $nIn = $n - $nOut;

        // 1. Outer Circle (Radius = 1.0)
        $thetaOut = Tensor::randomUniform([$nOut, 1], 0.0, 2.0 * M_PI);
        $xOut = $thetaOut->copy()->cos();
        $yOut = $thetaOut->copy()->sin();
        $labelsOut = Tensor::zeros($nOut, 1);

        // 2. Inner Circle (Radius = factor)
        $thetaIn = Tensor::randomUniform([$nIn, 1], 0.0, 2.0 * M_PI);
        $xIn = $thetaIn->copy()->cos()->mulScalarInplace($this->factor);
        $yIn = $thetaIn->copy()->sin()->mulScalarInplace($this->factor);
        $labelsIn = Tensor::ones($nIn, 1);

        // 3. Hardware Concatenation [N, 1]
        $x = Tensor::concat([$xOut, $xIn], 0);
        $y = Tensor::concat([$yOut, $yIn], 0);
        
        $samples = Tensor::concat([$x, $y], 1);
        $labels = Tensor::concat([$labelsOut, $labelsIn], 0)->squeeze();

        // 4. Inject Gaussian Noise natively
        if ($this->noise > 0.0) {
            $noiseTensor = Tensor::randomNormal([$n, 2], 0.0, $this->noise);
            $samples->addInplace($noiseTensor);
        }

        return new Dataset($samples, $labels);
    }
}