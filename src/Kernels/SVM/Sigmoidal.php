<?php
declare(strict_types=1);

namespace Pml\Kernels\SVM;

/**
 * Sigmoidal (Hyperbolic Tangent) SVM Kernel.
 * K(a, b) = tanh(gamma * a·b + coef0)
 */
final class Sigmoidal implements Kernel
{
    public function __construct(
        private readonly float $gamma = 0.001,
        private readonly float $coef0 = 0.0
    ) {}

    public function compute(array $a, array $b): float
    {
        $dot = 0.0;
        foreach ($a as $i => $v) {
            $dot += $v * ($b[$i] ?? 0.0);
        }
        return tanh($this->gamma * $dot + $this->coef0);
    }
}
