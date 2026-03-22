<?php

declare(strict_types=1);
namespace Pml\Layers;

use Pml\Layers\SelfAttention;

class TransformerBlock
{
    private SelfAttention $attention;
    private Tensor $norm1Weight;
    // ... MLP weights (w1, w2, w3) would go here ...

    public function __construct(SelfAttention $attn, Tensor $norm1) 
    {
        $this->attention = $attn;
        $this->norm1Weight = $norm1;
    }

    public function forward(Tensor $x): Tensor 
    {
        // Save residual
        $residual = new Tensor($x->shape);
        \FFI::memcpy($residual->buffer, $x->buffer, $x->size * 4);

        // Pre-normalization
        Ops::rmsNorm($x, $this->norm1Weight);

        // Attention
        $attnOut = $this->attention->forward($x);

        // Add residual (x = x + attnOut)
        Ops::addInPlace($residual, $attnOut);

        // ... Feed Forward Network (MLP) would go here, 
        // following the same Norm -> MLP -> Add Residual pattern ...

        return $residual;
    }
}