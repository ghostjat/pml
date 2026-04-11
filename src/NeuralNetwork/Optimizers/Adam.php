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
final class Adam implements Optimizer
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
        
        // Calculate bias corrections once globally per step
        $beta1_t = 1.0 - pow($this->beta1, $this->t);
        $beta2_t = 1.0 - pow($this->beta2, $this->t);
        
        // Pre-calculate scaling factors to prevent multiple division operations in C
        $step_size = $this->learningRate / $beta1_t;
        $v_hat_scale = 1.0 / $beta2_t;

        foreach ($layers as $layer) {
            $params = $layer->getParameters();
            $grads = $layer->getGradients();

            foreach ($params as $name => $paramTensor) {
                if (isset($grads[$name])) {
                    
                    // O(1) Cache-friendly lookup linking the parameter to its momentum state
                    $oid = spl_object_id($paramTensor);
                    $g = $grads[$name];

                    // Initialize momentum tensors (zeros) if first time seeing this parameter
                    if (!isset($this->m[$oid])) {
                        $shape = $paramTensor->shape();
                        $this->m[$oid] = Tensor::zeros(...$shape);
                        $this->v[$oid] = Tensor::zeros(...$shape);
                    }

                    $m = $this->m[$oid];
                    $v = $this->v[$oid];

                    // 1. Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
                    $tempG1 = $g->mulScalar(1.0 - $this->beta1);
                    $m->mulScalarInplace($this->beta1)->addInplace($tempG1);

                    // 2. Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * g^2
                    $tempG2 = $g->square()->mulScalarInplace(1.0 - $this->beta2);
                    $v->mulScalarInplace($this->beta2)->addInplace($tempG2);

                    // 3. Compute the Denominator: sqrt(v_hat) + eps
                    // Scaled dynamically natively in C without duplicating the $v tensor
                    $denom = $v->mulScalar($v_hat_scale)->sqrt()->addScalarInplace($this->epsilon);

                    // 4. Compute the Update: (m_hat / denom) * lr
                    // $m already holds the un-corrected momentum, step_size applies the beta1 correction.
                    $update = $m->div($denom)->mulScalarInplace($step_size);

                    // 5. Apply the final gradient descent step IN-PLACE
                    // Guarantees zero memory allocation crossing the FFI boundary
                    $paramTensor->subInplace($update);
                }
            }
        }
    }

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