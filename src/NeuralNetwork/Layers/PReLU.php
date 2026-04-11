<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;
use Pml\Dataset;

/**
 * Parametric ReLU (PReLU) — learnable per-unit negative slope.
 * f(z) = z if z > 0, else alpha_i * z   (alpha_i learned via gradient descent)
 *
 * JIT & Memory Optimized:
 * - Alpha vector stays in C memory; gradient update is a single in-place mul.
 * - Forward pass uses C-level where() — one branch per call.
 */
final class PReLU implements Layer
{
    private ?Tensor $alpha = null;   // [units] learnable slopes

    public function __construct(
        private readonly int   $units,
        private readonly float $initAlpha   = 0.25,
        private readonly float $learningRate = 0.001
    ) {}

    public function forward(Tensor $x, bool $training = true): Tensor
    {
        if ($this->alpha === null) {
            $this->alpha = Tensor::zeros($this->units)->addScalarInplace($this->initAlpha);
        }
        $pos  = $x->relu();
        $neg  = $x->mul($this->alpha);
        return $x->greaterScalar(0.0)->where($pos, $neg);
    }

    public function backward(Tensor $dOut, Tensor $x): Tensor
    {
        $mask  = $x->greaterScalar(0.0);
        $ones  = Tensor::ones(...$x->shape());
        $dX    = $mask->where($ones, $this->alpha);

        // Update alpha: d_alpha = sum(dOut * x * (x <= 0)) per unit
        $negX  = $mask->logicalNot()->mul($x)->mul($dOut);
        $dAlpha = $negX->meanAxis(0);
        $this->alpha->subInplace($dAlpha->mulScalarInplace($this->learningRate));

        return $dOut->mul($dX);
    }

    public function params(): array { return $this->alpha !== null ? [$this->alpha] : []; }
}
