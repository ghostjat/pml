<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

use Pml\Tensor;

/**
 * Stochastic Gradient Descent with optional L2 weight decay and momentum.
 * Alias for the existing SGD; provided for RubixML API compatibility.
 *
 * JIT & Memory Optimized:
 * - Momentum accumulator lives in C memory, reused across steps.
 * - All arithmetic is in-place to avoid extra C allocations.
 */
final class Stochastic implements Optimizer
{
    /** @var array<int, Tensor> velocity buffers keyed by param object-id */
    private array $velocity = [];

    public function __construct(
        private readonly float $lr         = 0.01,
        private readonly float $momentum   = 0.0,
        private readonly float $decay      = 0.0
    ) {}

    public function update(Tensor $param, Tensor $grad): void
    {
        $id = spl_object_id($param);

        // Weight decay: g += decay * w
        $g = $this->decay > 0.0
            ? $grad->add($param->mulScalar($this->decay))
            : $grad;

        if ($this->momentum > 0.0) {
            if (!isset($this->velocity[$id])) {
                $this->velocity[$id] = Tensor::zeros(...$param->shape());
            }
            $v = $this->velocity[$id];
            $v->mulScalarInplace($this->momentum)->addInplace($g->mulScalar($this->lr));
            $param->subInplace($v);
        } else {
            $param->subInplace($g->mulScalar($this->lr));
        }
    }

    public function learningRate(): float { return $this->lr; }
}
