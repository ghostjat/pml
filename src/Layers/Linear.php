<?php

declare(strict_types=1);

namespace Pml\Layers;

use Pml\{Tensor, Ops, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  LINEAR LAYER (fully-connected / dense)
// ═══════════════════════════════════════════════════════════════════════════

class Linear
{
    /**
     * @param Tensor      $weight  [out_features, in_features]
     * @param Tensor|null $bias    [out_features]
     */
    public function __construct(
        private readonly Tensor  $weight,
        private readonly ?Tensor $bias = null,
    ) {}

    /**
     * y = x * W^T + b
     * $x: [*, in_features] → out: [*, out_features]
     */
    public function forward(Tensor $x): Tensor
    {
        $out = Ops::matmul($x, $this->weight, false, true);
        if ($this->bias !== null) {
            Ops::addBiasInPlace($out, $this->bias);
        }
        return $out;
    }
}
