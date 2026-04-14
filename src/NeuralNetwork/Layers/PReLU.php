<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;
use RuntimeException;

/**
 * Parametric Rectified Linear Unit (PReLU) Layer.
 * * JIT & Memory Optimized:
 * - Caches input reference for zero-copy backward pass.
 * - Complies strictly with the DAG Layer interface (Lazy compilation).
 * - Utilizes in-place math to minimize Zend GC overhead.
 */
final class PReLU implements Layer
{
    private float $initialAlpha;
    private ?Tensor $alphas = null;

    // Cached states for Backpropagation
    private ?Tensor $x = null;
    private ?Tensor $dAlphas = null;

    /**
     * @param float $initialAlpha The starting value for the learnable alphas.
     */
    public function __construct(float $initialAlpha = 0.25)
    {
        $this->initialAlpha = $initialAlpha;
    }

    public function forward(Tensor $input): Tensor
    {
        // 1. Cache the input reference (Zero-Copy)
        $this->x = $input;

        // 2. Lazy Initialization: Allocate alphas dynamically based on feature count
        if ($this->alphas === null) {
            $shape = $input->shape();
            $features = $shape[1] ?? 1;
            
            // Create a [1, features] tensor for the alphas
            $initArray = array_fill(0, $features, $this->initialAlpha);
            $this->alphas = Tensor::fromArray([$initArray]);
        }

        // 3. Math: Y = max(0, X) + alphas * min(0, X)
        // We use safe tensor primitives: relu(X) == max(0, X). 
        // Therefore, min(0, X) == X - relu(X).
        $pos = $input->relu(); 
        $neg = $input->sub($pos); 
        
        // Return pos + (neg * alphas)
        return $pos->addInplace($neg->mul($this->alphas));
    }

    public function backward(Tensor $dY): Tensor
    {
        if ($this->x === null || $this->alphas === null) {
            throw new RuntimeException("Backward pass called before forward pass.");
        }

        // 1. Recompute the negative values: min(0, X)
        $pos = $this->x->relu();
        $neg = $this->x->sub($pos);

        // 2. Compute Gradients w.r.t Alphas
        // dAlphas = sum(dY * min(0, X), axis=0)
        $this->dAlphas = $dY->mul($neg)->sumAxis(0);

        // 3. Compute Gradients w.r.t Input (dX)
        // dX = dY * (X > 0 ? 1 : alphas)
        
        // FIXED: Using your native Scalar comparison methods
        $posMask = $this->x->greaterScalar(0.0);
        // Using logicalNot on the positive mask creates the perfect <= 0 mask
        $negMask = $posMask->logicalNot();

        // dX_pos = dY * 1 (where X > 0)
        $dX_pos = $dY->mul($posMask);
        
        // dX_neg = (dY * alphas) * 1 (where X <= 0)
        $dX_neg = $dY->mul($negMask)->mulInplace($this->alphas);

        // dX = dX_pos + dX_neg
        return $dX_pos->addInplace($dX_neg);
    }

    public function getParameters(): array
    {
        if ($this->alphas === null) return [];
        return ['alphas' => $this->alphas];
    }

    public function getGradients(): array
    {
        if ($this->dAlphas === null) return [];
        return ['alphas' => $this->dAlphas];
    }
}