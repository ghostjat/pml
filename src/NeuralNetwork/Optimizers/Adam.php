<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

use Pml\NeuralNetwork\Layers\Layer;
use Pml\Tensor;

/**
 * Adam (Adaptive Moment Estimation) Optimizer.
 * Dynamically adapts the learning rate for every parameter using First and Second moments.
 * * JIT & Memory Optimized:
 * - O(1) momentum state lookups via `spl_object_id()`.
 * - Extensive use of In-Place Tensor mutations to prevent PHP Heap fragmentation.
 * - Safely detaches FFI pointers during serialization to prevent memory leaks/crashes.
 */
final class Adam implements Optimizer, LearningRateAware
{
    private float $learningRate;
    private float $beta1;
    private float $beta2;
    private float $epsilon;
    
    private int $t = 0; // Global step counter

    /** @var array<int, Tensor> First moment estimates (m) mapped by object ID */
    private array $m = [];
    
    /** @var array<int, Tensor> Second moment estimates (v) mapped by object ID */
    private array $v = [];

    public function __construct(
        float $learningRate = 0.001,
        float $beta1 = 0.9,
        float $beta2 = 0.999,
        float $epsilon = 1e-8
    ) {
        $this->learningRate = $learningRate;
        $this->beta1 = $beta1;
        $this->beta2 = $beta2;
        $this->epsilon = $epsilon;
    }

    public function step(array $layers): void
    {
        $this->t++;

        foreach ($layers as $layer) {
            $params = $layer->getParameters();
            $grads  = $layer->getGradients();

            foreach ($params as $name => $paramTensor) {
                if (!isset($grads[$name])) {
                    continue;
                }

                $oid = spl_object_id($paramTensor);

                if (!isset($this->m[$oid])) {
                    $shape           = $paramTensor->shape();
                    $this->m[$oid]   = Tensor::zeros(...$shape);
                    $this->v[$oid]   = Tensor::zeros(...$shape);
                }

                // Single C kernel: m/v update + bias correction + param update — zero PHP allocs.
                Tensor::fusedAdamStep(
                    $paramTensor,
                    $grads[$name],
                    $this->m[$oid],
                    $this->v[$oid],
                    $this->learningRate,
                    $this->beta1,
                    $this->beta2,
                    $this->epsilon,
                    $this->t
                );
            }
        }
    }

    // ---- LearningRateAware ------------------------------------------------

    public function getLearningRate(): float { return $this->learningRate; }

    /**
     * Replace the learning rate mid-training (used by LRScheduler / Trainer).
     * Safe to call between steps; takes effect on the very next step().
     * Momentum buffers and the step counter are preserved so Adam's adaptive
     * correction continues from where it left off.
     */
    public function setLearningRate(float $lr): void { $this->learningRate = $lr; }

    // -----------------------------------------------------------------------

    /**
     * Called automatically by PHP when Sequential::save() is triggered.
     * Prevents the FFI C-Tensors ($m and $v) from being serialized, averting Fatal Errors.
     */
    public function __sleep(): array
    {
        return ['learningRate', 'beta1', 'beta2', 'epsilon', 't'];
    }

    /**
     * Called automatically by PHP when Sequential::load() is triggered.
     * Re-establishes the state arrays so training can safely resume.
     */
    public function __wakeup(): void
    {
        $this->m = [];
        $this->v = [];
    }
}