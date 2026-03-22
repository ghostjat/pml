<?php

declare(strict_types=1);

namespace Pml\Layers;

use Pml\{Tensor, Ops};

// ═══════════════════════════════════════════════════════════════════════════
//  EMBEDDING LAYER
// ═══════════════════════════════════════════════════════════════════════════

class Embedding
{
    /**
     * @param Tensor $weight [vocab_size, d_model]
     */
    public function __construct(private readonly Tensor $weight) {}

    /**
     * @param int[] $tokenIds
     * @return Tensor [seq_len, d_model]
     */
    public function forward(array $tokenIds): Tensor
    {
        return Ops::embedding($this->weight, $tokenIds);
    }
}